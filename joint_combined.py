"""
Joint gradient projection with COMBINED loss — all 6 materials updated together.
One full-body render → per-material UV maps → COMBINED feature loss
→ 6 independent backprojections → 6 simultaneous texture updates.
"""
import torch, torch.nn.functional as F, numpy as np, math, time, os
from PIL import Image
from ultralytics import YOLO
import nvdiffrast.torch as dr
from collections import defaultdict

device = torch.device("cuda")
glctx = dr.RasterizeCudaContext(device=device)
torch.manual_seed(42)
print(f"GPU: {torch.cuda.get_device_name(0)}", flush=True)

# ============================================================
# OBJ + Textures
# ============================================================
def parse_obj(path):
    with open(path) as f: lines = f.readlines()
    av, au = [], []; fd = defaultdict(list); cur = None
    for line in lines:
        p = line.strip().split()
        if not p: continue
        if p[0] == "v": av.append([float(x) for x in p[1:4]])
        elif p[0] == "vt": au.append([float(x) for x in p[1:3]])
        elif p[0] == "usemtl": cur = p[1]
        elif p[0] == "f" and cur:
            face = [(int(t.split("/")[0])-1, int(t.split("/")[1])-1 if len(t.split("/"))>1 and t.split("/")[1] else -1) for t in p[1:]]
            if len(face) == 4: fd[cur].extend([[face[0],face[1],face[2]],[face[0],face[2],face[3]]])
            elif len(face) >= 3: fd[cur].append(face[:3])
    return np.array(av,np.float32), np.array(au,np.float32), dict(fd)

vs,us,fd = parse_obj("/root/quantum_soldier/quantum_character.obj")
av_all = torch.tensor(vs, dtype=torch.float32, device=device)
au_all = torch.tensor(us, dtype=torch.float32, device=device)
TARGET = ["M_Bulletproof_Light","M_Quantum_Arms","M_Drops_Tactical","M_Jeans","M_Holster_Hard","M_Head"]
BP = "/root/quantum_soldier/PBR_Textures"
TF = {
    "M_Bulletproof_Light":"Bulletproof/T_Bulletproof_Bege_BaseColor.png",
    "M_Quantum_Arms":"Arms/T_Quantum_Basemesh_Arms_BaseColor.1003.png",
    "M_Drops_Tactical":"Drops/T_Drops_Tactical_Bege_BaseColor.png",
    "M_Jeans":"Jeans/T_Jeans_Bege_BaseColor.png",
    "M_Holster_Hard":"Holdster_Hard/M_Holster_Hard_Bege_BaseColor.png",
    "M_Head":"Body/T_Head_BaseColor.png",
}
MD = {}
for mn in TARGET:
    if mn not in fd: continue
    tr = fd[mn]; vi = sorted(set(f[0] for t in tr for f in t)); v2 = {v:i for i,v in enumerate(vi)}
    lv = av_all[vi].contiguous()
    lf = torch.tensor([[v2[f[0]] for f in t] for t in tr], dtype=torch.int32, device=device).contiguous()
    lu = torch.zeros(len(vi),2,dtype=torch.float32,device=device)
    lc = torch.zeros(len(vi),dtype=torch.int32,device=device)
    for t in tr:
        for f in t:
            vi2,ui = f[0],f[1]
            if ui>=0 and ui<au_all.shape[0]: li=v2[vi2]; lu[li]+=au_all[ui]; lc[li]+=1
    lu[lc>0] = lu[lc>0]/lc[lc>0].float().unsqueeze(-1)
    MD[mn] = {"v":lv,"f":lf,"u":lu.contiguous()}
TX = {}
for mn,fn in TF.items():
    p = os.path.join(BP,fn)
    if os.path.exists(p):
        tx = Image.open(p).convert("RGB").resize((512,512),Image.LANCZOS)
        TX[mn] = (torch.from_numpy(np.array(tx)).float()/255.0).permute(2,0,1).to(device).contiguous()
MATS = [m for m in TARGET if m in TX]
tex_base = {m: TX[m].clone() for m in MATS}

# ============================================================
# FRCNN with hooks
# ============================================================
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()
IM_MEAN = torch.tensor([0.485,0.456,0.406],device=device).view(1,3,1,1)
IM_STD = torch.tensor([0.229,0.224,0.225],device=device).view(1,3,1,1)
def features(img_batch):
    """Get FPN features from backbone — these are IN the computation graph."""
    x = (img_batch.unsqueeze(0) - IM_MEAN) / IM_STD
    out = frcnn.backbone(x)  # OrderedDict with keys '0','1','2','3','pool'
    return {str(k): v for k, v in out.items()}

def combined_loss(feats):
    loss = torch.tensor(0.0, device=device); count = 0
    for name, feat in feats.items():
        B, C, H, W = feat.shape
        if C < 4: continue
        spatial_var = feat.var(dim=[2,3]).mean()
        active_frac = (feat.abs() > 0.1).float().mean()
        nms_part = -spatial_var * 1.0 - active_frac * 0.5
        # Per-family scale weight
        w = SCALE_WEIGHTS.get(name, 1.0)
        C_thresh = feat.mean(dim=1, keepdim=True)
        sup = feat[feat > C_thresh.expand_as(feat)]
        non = feat[feat <= C_thresh.expand_as(feat)]
        sup_n = sup.norm() if sup.numel()>0 else torch.tensor(0.0,device=device)
        non_n = non.norm() if non.numel()>0 else torch.tensor(0.0,device=device)
        fda_part = (sup_n - non_n) * w
        if torch.isfinite(nms_part) and torch.isfinite(fda_part):
            loss = loss + nms_part * 0.4 + fda_part * 0.6; count += 1
    return loss / max(count,1) if count>0 else torch.tensor(0.0,device=device)

# ============================================================
# Full-body render with per-material UV maps
# ============================================================
def render_full_with_uv(tex_dict, res=512):
    proj = torch.zeros(4,4,device=device); fv=1.0/math.tan(60*math.pi/180/2)
    proj[0,0]=proj[1,1]=fv; proj[2,2]=-0.01; proj[3,2]=-1
    e=(torch.rand(1,device=device)*2-1)*0.35; a=torch.rand(1,device=device)*2*math.pi
    d=1.3+torch.rand(1,device=device)*1.5
    eye=torch.cat([d*torch.cos(e)*torch.sin(a),d*torch.sin(e)+1.0,d*torch.cos(e)*torch.cos(a)])
    at=torch.tensor([-0.15,1.0,0.0],device=device); up=torch.tensor([0.,1.,0.],device=device)
    z=F.normalize(eye-at,dim=0); x=F.normalize(torch.linalg.cross(up,z),dim=0); yv=torch.linalg.cross(z,x)
    V=torch.eye(4,device=device); V[0,:3],V[1,:3],V[2,:3]=x,yv,z
    V[:3,3]=-torch.tensor([x@eye,yv@eye,z@eye],device=device); MVP=proj@V
    dp={}
    for mn in MD:
        md=MD[mn]; lvh=torch.cat([md["v"],torch.ones(md["v"].shape[0],1,device=device)],1).contiguous()
        lvc=lvh@MVP.T; lvc=lvc[:,:4]/lvc[:,3:4].clamp(min=0.01)
        dp[mn]=lvc[:,2].mean().item() if mn in tex_dict else -1e9
    mo=sorted([k for k in dp],key=lambda k:dp[k],reverse=True)
    cv=torch.zeros(1,res,res,4,device=device)
    uv_maps = {}
    for mn in mo:
        md=MD[mn]; tx=tex_dict[mn]
        if tx.dim()==3: tx=tx.unsqueeze(0)
        lvh=torch.cat([md["v"],torch.ones(md["v"].shape[0],1,device=device)],1).contiguous()
        lvc=lvh@MVP.T; lvc=lvc[:,:4]/lvc[:,3:4].clamp(min=0.01)
        rast,_=dr.rasterize(glctx,lvc.unsqueeze(0),md["f"],(res,res),grad_db=True)
        uvi,_=dr.interpolate(md["u"].unsqueeze(0).contiguous(),rast,md["f"])
        uv_maps[mn] = uvi.squeeze(0)[:,:,:2].clone()
        col=dr.texture(tx.contiguous(),uvi,filter_mode="linear")[:,:,:,:3]
        a = (rast[..., -1:] > 0).float()
        cv = torch.cat([col, a], -1) * a + cv * (1 - a)
    bg=torch.ones(1,res,res,3,device=device)*0.45
    ca=cv[...,3:4]; cr=cv[...,:3]
    img = (cr*ca+bg*(1-ca)).squeeze(0).permute(2,0,1)
    return img, uv_maps

def score(td, model, nv=12):
    with torch.no_grad():
        im=torch.stack([render_full_with_uv(td)[0] for _ in range(nv)],0)
    c=0.0
    for i in range(nv):
        np_im=im[i].permute(1,2,0).cpu().numpy(); np_im=(np_im.clip(0,1)*255).astype(np.uint8)
        r=model(np_im,conf=0.12,verbose=False)
        if r[0].boxes is not None:
            for b in r[0].boxes:
                if r[0].names.get(int(b.cls.item()),"")=="person": c+=float(b.conf.item())
    return c

# ============================================================
# Validation models
# ============================================================
import sys
gen_family = sys.argv[1] if len(sys.argv) > 1 else "G_A"
# Per-family loss weights for feature disruption
if gen_family == "G_A":
    SCALE_WEIGHTS = {'3': 3.0, '2': 3.0, '1': 1.5, '0': 1.0, 'pool': 0.5}  # Deep features (v8 family focus)
elif gen_family == "G_B":
    SCALE_WEIGHTS = {'3': 1.0, '2': 2.0, '1': 2.5, '0': 2.5, 'pool': 1.5}  # Mid features (v5+v10 focus)
elif gen_family == "G_D":
    SCALE_WEIGHTS = {'3': 2.0, '2': 3.0, '1': 1.5, '0': 1.0, 'pool': 0.5}  # Deep-med focus (v5s)
else:
    SCALE_WEIGHTS = {'3': 2.0, '2': 2.0, '1': 1.0, '0': 1.0, 'pool': 0.5}

VAL = {}
for n,p in [("v8n","/root/yolov8n.pt"),("v8s","/root/yolov8s.pt"),("v8m","/root/yolov8m.pt"),
             ("v5n","/root/yolov5nu.pt"),("v5s","/root/yolov5su.pt"),
             ("v10n","/root/yolov10n.pt"),("v10s","/root/yolov10s.pt")]:
    VAL[n] = YOLO(p)

# Baseline
bl = {n: score(TX, m, 12) for n,m in VAL.items()}
print("Baseline:", flush=True)
for n in ["v8n","v8s","v8m","v5n","v5s","v10n","v10s"]:
    print(f"  {n}: {bl[n]:.2f}", flush=True)

# ============================================================
# JOINT COMBINED ATTACK: all 6 materials updated together
# ============================================================
EPS = 0.039; STEPS = 400; VIEWS = 6; TEX_SIZE = 512
out_dir = f"/root/{gen_family}_out"
os.makedirs(out_dir, exist_ok=True)
t0 = time.time()

tex_adv = {m: TX[m].clone() for m in MATS}
best_tex = {m: t.clone() for m,t in tex_adv.items()}
best_score = float('inf')

print(f"\n{'='*60}", flush=True)
print(f"{gen_family}: JOINT COMBINED, {len(MATS)} mats, {STEPS} steps, {VIEWS} views", flush=True)
print(f"{'='*60}", flush=True)

for step in range(STEPS):
    # Accumulate tex gradient for ALL materials
    tex_grads = {m: torch.zeros(3, TEX_SIZE, TEX_SIZE, device=device) for m in MATS}
    total_loss = 0.0
    grad_count = 0

    for v in range(VIEWS):
        img, uv_maps = render_full_with_uv(tex_adv)

        # NaN guard: if render produced bad pixels, skip this view
        if not torch.isfinite(img).all():
            continue

        img_copy = img.detach().clone().requires_grad_(True)
        feats = features(img_copy)
        loss = combined_loss(feats)
        if loss == 0 or not torch.isfinite(loss): continue

        loss.backward()
        if img_copy.grad is None: continue
        img_grad = img_copy.grad.squeeze(0)
        if not torch.isfinite(img_grad).all(): continue

        # Backproject image gradient to each material's texture via its UV map
        for mat_name in MATS:
            if mat_name not in uv_maps: continue
            uv_map = uv_maps[mat_name]
            uv_f = uv_map * (TEX_SIZE - 1)
            uv_i = uv_f.long()
            uv_r = uv_f - uv_i.float()
            u0 = uv_i[:,:,0].clamp(0, TEX_SIZE-1)
            v0 = uv_i[:,:,1].clamp(0, TEX_SIZE-1)
            u1 = (u0+1).clamp(0, TEX_SIZE-1)
            v1 = (v0+1).clamp(0, TEX_SIZE-1)
            w00 = (1-uv_r[:,:,0])*(1-uv_r[:,:,1])
            w10 = uv_r[:,:,0]*(1-uv_r[:,:,1])
            w01 = (1-uv_r[:,:,0])*uv_r[:,:,1]
            w11 = uv_r[:,:,0]*uv_r[:,:,1]
            for c in range(3):
                gc = img_grad[c]
                tex_grads[mat_name][c].index_put_((u0,v0), gc*w00, accumulate=True)
                tex_grads[mat_name][c].index_put_((u1,v0), gc*w10, accumulate=True)
                tex_grads[mat_name][c].index_put_((u0,v1), gc*w01, accumulate=True)
                tex_grads[mat_name][c].index_put_((u1,v1), gc*w11, accumulate=True)

        grad_count += 1
        total_loss += loss.item()

    if grad_count == 0: continue

    # Update ALL materials simultaneously
    lr = 0.002 * (1 - step/STEPS) + 0.0005
    for mat_name in MATS:
        tex_grad = tex_grads[mat_name] / grad_count
        if not torch.isfinite(tex_grad).all(): continue
        tex_adv[mat_name] = tex_adv[mat_name] - lr * tex_grad
        tex_adv[mat_name] = tex_adv[mat_name].clamp(tex_base[mat_name]-EPS, tex_base[mat_name]+EPS).clamp(0,1)

    # Logging
    if step % 25 == 0 or step == STEPS - 1:
        vm = score(tex_adv, VAL['v8m'], 8)
        vn = score(tex_adv, VAL['v8n'], 8)
        vs_s = score(tex_adv, VAL['v8s'], 8)
        v5 = score(tex_adv, VAL['v5n'], 8)
        v10 = score(tex_adv, VAL['v10n'], 8)
        el = time.time() - t0
        red_m = (bl['v8m']-vm)/(bl['v8m']+1e-8)*100
        red_n = (bl['v8n']-vn)/(bl['v8n']+1e-8)*100
        print(f"  S{step:3d}: v8m={vm:.2f}({red_m:+4.0f}%) v8n={vn:.2f}({red_n:+4.0f}%) v8s={vs_s:.2f} v5n={v5:.2f} v10n={v10:.2f} {el:.0f}s", flush=True)

        if vm > 0 and vm < best_score:
            best_score = vm
            best_tex = {m: t.clone() for m,t in tex_adv.items()}
            print(f"  [BEST] v8m={vm:.2f}", flush=True)

        # Checkpoint every 25 steps (crash recovery)
        ckpt_dir = f"{out_dir}/checkpoints"
        os.makedirs(ckpt_dir, exist_ok=True)
        for m in MATS:
            torch.save(tex_adv[m], f"{ckpt_dir}/{m}_step{step:04d}.pt")
        torch.save(best_tex[m], f"{ckpt_dir}/{m}_best.pt")

# ============================================================
# FINAL
# ============================================================
print(f"\n=== FINAL ===", flush=True)
td_final = best_tex
for n, m in VAL.items():
    bc = bl[n]
    ac = score(td_final, m, 24)
    red = (bc - ac) / (bc + 1e-8) * 100
    print(f"  {n:6s}: {bc:.2f} -> {ac:.2f} ({red:>+7.0f}%)", flush=True)

for i in range(6):
    with torch.no_grad():
        oi, _ = render_full_with_uv(TX)
        ai, _ = render_full_with_uv(td_final)
    for label, img in [("orig", oi), ("adv", ai)]:
        np_img = img.permute(1, 2, 0).cpu().numpy()
        Image.fromarray((np_img.clip(0, 1) * 255).astype(np.uint8)).save(f"{out_dir}/{label}_{i}.png")

for m in td_final:
    torch.save(td_final[m], f"{out_dir}/{m}_adv.pt")

print(f"\nDONE in {(time.time()-t0)/60:.1f}min. {out_dir}/", flush=True)
