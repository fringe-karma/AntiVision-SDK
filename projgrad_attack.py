"""
V12: Manual gradient projection — clean, no custom autograd.

Pipeline:
  1. nvdiffrast render (beautiful, YOLO detects person)
  2. Compute 2D image gradient using differentiable proxy (Faster R-CNN)
  3. nvdiffrast also gives us UV map — map each screen pixel to texture UV
  4. Project image gradient to texture: dL/dTex[u,v] = sum(dL/dI[x,y] for pixels (x,y) that map to (u,v))
  5. Update texture ← step in direction of projected gradient
  6. Repeat

Key insight: we don't need backward through rendering.
We just need to know WHICH texture pixel became WHICH screen pixel.
nvdiffrast gives us exactly this via the interpolated UVs.
"""
import torch, torch.nn.functional as F, numpy as np, math, time, sys, os, json
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import nvdiffrast.torch as dr

device = torch.device("cuda")
print(f"GPU: {torch.cuda.get_device_name(0)}")
glctx = dr.RasterizeCudaContext(device=device)

# =====================================================================
# 1. OBJ Parser (same as v5/v6)
# =====================================================================
def parse_obj(path):
    with open(path) as f: lines = f.readlines()
    all_verts, all_uvs = [], []; faces_dict = defaultdict(list); cur = None
    for line in lines:
        p = line.strip().split()
        if not p: continue
        if p[0] == "v": all_verts.append([float(x) for x in p[1:4]])
        elif p[0] == "vt": all_uvs.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl": cur = p[1]
        elif p[0] == "f" and cur:
            face = [(int(t.split("/")[0])-1, int(t.split("/")[1])-1 if len(t.split("/"))>1 and t.split("/")[1] else -1) for t in p[1:]]
            if len(face) == 4: faces_dict[cur].extend([[face[0],face[1],face[2]],[face[0],face[2],face[3]]])
            elif len(face) >= 3: faces_dict[cur].append(face[:3])
    return np.array(all_verts,np.float32), np.array(all_uvs,np.float32), dict(faces_dict)

MAT_TEX = {"Arm.001":"Arm_Base_color.png","Chest.001":"Chest_Base_color.png","Helmet.001":"Helmet_Base_color.png","Legs.001":"Legs_Base_color.png","Mask.002":"Mask_Base_color.png"}

def load_tex(path):
    img = Image.open(path).convert("RGB").resize((512,512),Image.LANCZOS)
    return (torch.from_numpy(np.array(img)).float()/255.0).permute(2,0,1).to(device).contiguous()

# =====================================================================
# 2. nvdiffrast renderer with UV map output
# =====================================================================
def build_proj(fov=60.0):
    f=1.0/math.tan(fov*math.pi/180/2); P=torch.zeros(4,4,device=device); P[0,0]=P[1,1]=f; P[2,2]=-0.01; P[3,2]=-1; return P

def rand_cam():
    e=(torch.rand(1,device=device)*2-1)*0.4; a=torch.rand(1,device=device)*2*math.pi; d=1.5+torch.rand(1,device=device)*1.0
    return torch.cat([d*torch.cos(e)*torch.sin(a), d*torch.sin(e)+1.0, d*torch.cos(e)*torch.cos(a)])

def nvdiff_render(tex, verts, faces_t, uv_arr, get_uv_map=False):
    """Render with nvdiffrast. If get_uv_map=True, also return per-pixel UV coordinates."""
    proj=build_proj(); eye=rand_cam()
    at=torch.tensor([-0.15,1.0,0.0],device=device); up=torch.tensor([0.,1.,0.],device=device)
    z=F.normalize(eye-at,dim=0); x=F.normalize(torch.linalg.cross(up,z),dim=0); y=torch.linalg.cross(z,x)
    V=torch.eye(4,device=device); V[0,:3],V[1,:3],V[2,:3]=x,y,z; V[:3,3]=-torch.tensor([x@eye,y@eye,z@eye],device=device)
    MVP=proj@V
    vh=torch.cat([verts,torch.ones(verts.shape[0],1,device=device)],1); vc=vh@MVP.T; vc=vc[:,:4]/vc[:,3:4].clamp(min=0.01)
    rast, rast_db = dr.rasterize(glctx,vc.unsqueeze(0),faces_t,(640,640),grad_db=get_uv_map)
    uvi, _ = dr.interpolate(uv_arr.unsqueeze(0),rast,faces_t)
    col = dr.texture(tex.unsqueeze(0),uvi,filter_mode='linear')[:,:,:,:3]
    alpha = (rast[...,-1:]>0).float()
    yg=torch.linspace(0.2,0.9,640,device=device).view(-1,1,1)
    bg=yg.repeat(1,640,3)*0.4+0.3
    result = (col*alpha+bg.unsqueeze(0)*(1-alpha)).squeeze(0).permute(2,0,1)
    if get_uv_map:
        uv_map = uvi.squeeze(0)[:,:,:2]  # [H, W, 2] — UV at each pixel
        return result, uv_map
    return result

def yolo_detect(batch, yolo_model):
    total=0.0; dets=0
    for i in range(batch.shape[0]):
        img=batch[i].permute(1,2,0).detach().cpu().numpy(); img=(img.clip(0,1)*255).astype(np.uint8)
        r=yolo_model(img,conf=0.10,verbose=False)
        if r[0].boxes is not None:
            for b in r[0].boxes:
                if r[0].names.get(int(b.cls.item()),"?")=="person": total+=float(b.conf.item()); dets+=1
    return total,dets

# =====================================================================
# 3. Main: gradient projection attack
# =====================================================================
def main():
    print("Loading OBJ + textures...")
    verts_np, all_uvs_np, fdict = parse_obj("/root/ancient_character/1.obj")
    verts = torch.tensor(verts_np, dtype=torch.float32, device=device)
    all_uvs_t = torch.tensor(all_uvs_np, dtype=torch.float32, device=device)

    textures={}
    for mat,fname in MAT_TEX.items():
        path=f"/root/ancient_character/textures/{fname}"
        if os.path.exists(path): textures[mat]=load_tex(path)

    # Optimize Arm texture
    tex_adv = textures["Arm.001"].clone()
    tex_orig = textures["Arm.001"].clone()

    # Combine all faces
    all_faces = []; [all_faces.extend(flist) for flist in fdict.values()]
    faces_t = torch.tensor([[f[0] for f in tri] for tri in all_faces], dtype=torch.int32, device=device)

    # Build vertex-aligned UV
    uv_arr = torch.zeros(verts.shape[0],2,dtype=torch.float32,device=device)
    uv_cnt = torch.zeros(verts.shape[0],dtype=torch.int32,device=device)
    for tri in all_faces:
        for f in tri:
            vi,uvi=f[0],f[1]
            if uvi>=0 and uvi<all_uvs_t.shape[0]: uv_arr[vi]+=all_uvs_t[uvi]; uv_cnt[vi]+=1
    uv_arr[uv_cnt>0]=uv_arr[uv_cnt>0]/uv_cnt[uv_cnt>0].float().unsqueeze(-1)

    # Detectors
    yolo = YOLO("yolov8n.pt")
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

    # Baseline
    print("\n=== BASELINE (16 views) ===")
    with torch.no_grad():
        bl=[nvdiff_render(tex_adv,verts,faces_t,uv_arr) for _ in range(16)]
        bl_batch=torch.stack(bl,0)
    bl_c,bl_d=yolo_detect(bl_batch,yolo)
    print(f"Baseline: {bl_d} dets, conf={bl_c:.4f}")
    if bl_d==0: print("No person detected!"); return

    # === GRADIENT PROJECTION ATTACK ===
    print(f"\n=== GRADIENT PROJECTION (200 steps) ===")
    eps=0.039; lr=0.003; steps=400; tex_size=512
    best_tex,best_conf=tex_adv.clone(),float('inf')
    t0=time.time()

    N_VIEWS = 6  # EOT: 6 random views per step, average gradients

    for step in range(steps):
        # Accumulate texture gradients over multiple views
        tex_grad_total = torch.zeros(3, tex_size, tex_size, device=device)
        total_loss = 0.0

        for _ in range(N_VIEWS):
            # 1. Render image from random camera
            img, uv_map = nvdiff_render(tex_adv, verts, faces_t, uv_arr, get_uv_map=True)

            # 2. Compute 2D image gradient via proxy
            img_batch = img.unsqueeze(0).detach().clone().requires_grad_(True)
            outputs = frcnn([img_batch[0]])
            loss = torch.tensor(0.0, device=device)
            for out in outputs:
                if (out["labels"]==1).any(): loss=loss+out["scores"][out["labels"]==1].sum()
            if loss==0: continue
            loss.backward()
            img_grad = img_batch.grad.squeeze(0)
            total_loss += loss.item()

            # 3. Project image gradient → texture gradient via UV map
            uv_screen = uv_map.clone()
            uv_floor = uv_screen * (tex_size-1)
            uv_floor_int = uv_floor.long()
            uv_frac = uv_floor - uv_floor_int.float()

            u0 = uv_floor_int[:,:,0].clamp(0,tex_size-1)
            v0 = uv_floor_int[:,:,1].clamp(0,tex_size-1)
            u1 = (u0+1).clamp(0,tex_size-1)
            v1 = (v0+1).clamp(0,tex_size-1)
            w00 = (1-uv_frac[:,:,0])*(1-uv_frac[:,:,1])
            w10 = uv_frac[:,:,0]*(1-uv_frac[:,:,1])
            w01 = (1-uv_frac[:,:,0])*uv_frac[:,:,1]
            w11 = uv_frac[:,:,0]*uv_frac[:,:,1]

            for c in range(3):
                gc = img_grad[c]
                tex_grad_total[c].index_put_((u0,v0), gc*w00, accumulate=True)
                tex_grad_total[c].index_put_((u1,v0), gc*w10, accumulate=True)
                tex_grad_total[c].index_put_((u0,v1), gc*w01, accumulate=True)
                tex_grad_total[c].index_put_((u1,v1), gc*w11, accumulate=True)

        if total_loss == 0: continue

        # 5. Update texture with averaged gradient
        tex_adv = tex_adv - lr * tex_grad_total / N_VIEWS
        tex_adv = tex_adv.clamp(tex_orig-eps, tex_orig+eps).clamp(0,1)

        if step%20==0 or step==steps-1:
            with torch.no_grad():
                tv=[nvdiff_render(tex_adv,verts,faces_t,uv_arr) for _ in range(8)]
                tb=torch.stack(tv,0)
            yc,yd=yolo_detect(tb,yolo)
            red=(bl_c-yc)/(bl_c+1e-8)*100; dt=(tex_adv-tex_orig).abs().max().item()
            print(f"  Step {step:3d}: YOLO={yc:.4f}({yd}d) {red:+.1f}% delta={dt:.4f} loss={total_loss/N_VIEWS:.4f}")
            if yc<best_conf and yc>0: best_conf,best_tex=yc,tex_adv.clone()

    elapsed=time.time()-t0; print(f"Elapsed: {elapsed:.0f}s")

    with torch.no_grad():
        fv=[nvdiff_render(best_tex,verts,faces_t,uv_arr) for _ in range(16)]
        fb=torch.stack(fv,0)
    fc,fd=yolo_detect(fb,yolo)
    red=(bl_c-fc)/(bl_c+1e-8)*100; md=(best_tex-tex_orig).abs().max().item()
    print(f"\nBaseline:    {bl_d} dets, conf={bl_c:.4f}")
    print(f"Adversarial: {fd} dets, conf={fc:.4f}")
    print(f"Reduction:   {red:+.1f}%")
    print(f"Max delta:   {md:.4f}")
    status = "EFFECTIVE!" if red>50 else ("PARTIAL" if red>20 else "WEAK")
    print(f"Verdict:     {status}")

    adv_np=best_tex.permute(1,2,0).cpu().numpy()
    Image.fromarray((adv_np.clip(0,1)*255).astype(np.uint8)).save("/root/v12_adv.png")
    with open("/root/v12_results.json","w") as f: json.dump({"baseline":bl_c,"adversarial":fc,"reduction":red,"max_delta":md},f)
    print("Saved")

if __name__=="__main__": main()
