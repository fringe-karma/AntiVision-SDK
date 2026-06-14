"""
Generator Training: Seed → Adversarial Texture Network

Architecture: U-Net (lightweight, 2-5MB target)
Input: seed (64-bit binary) + base_texture (3,H,W)
Output: adversarial_texture (3,H,W)

Training loop:
  1. Generator(seed, base_tex) → adv_tex
  2. Render adv_tex via nvdiffrast (multi-view)
  3. YOLO/FRCNN detect → person confidence loss
  4. Gradient back to Generator (not to texture directly!)
  5. Repeat for thousands of random seeds

Key difference from v1-32: we're training a NETWORK, not optimizing a single texture.
"""
import torch, torch.nn as nn, torch.nn.functional as F
import numpy as np, math, time, secrets, sys, os, json, argparse
from collections import defaultdict
from PIL import Image
from ultralytics import YOLO
import nvdiffrast.torch as dr

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")
if torch.cuda.is_available(): print(f"GPU: {torch.cuda.get_device_name(0)}")

# ============================================================
# 1. Generator Network (U-Net style, lightweight)
# ============================================================
class AdversarialGenerator(nn.Module):
    """
    Seed (64-bit) + Base Texture → Adversarial Texture.

    Architecture: Lightweight U-Net
    - Encoder: 4 downsampling blocks
    - Decoder: 4 upsampling blocks + skip connections
    - Seed injection: 64-bit → linear → reshape → concat with bottleneck

    Target: 1-3M params, 2-5 MB ONNX, <2ms inference.
    """
    def __init__(self, seed_bits=64, base_channels=16):
        super().__init__()
        # Seed encoder → compact latent (smaller = fewer params)
        self.seed_fc = nn.Sequential(
            nn.Linear(seed_bits, 128),
            nn.ReLU(),
            nn.Linear(128, base_channels * 4 * 4),
        )
        self.seed_channels = base_channels

        self.seed_compress = nn.Conv2d(base_channels, 1, 1)  # compress seed map to 1 channel
        # Encoder
        self.enc1 = self._conv_block(3 + 1, base_channels)           # 3 RGB + 1 seed_channel
        self.enc2 = self._conv_block(base_channels, base_channels * 2)
        self.enc3 = self._conv_block(base_channels * 2, base_channels * 4)
        self.enc4 = self._conv_block(base_channels * 4, base_channels * 8)

        # Bottleneck (receives seed injection)
        self.bottleneck = self._conv_block(
            base_channels * 8 + base_channels, base_channels * 8   # +seed_channels
        )

        # Decoder (with skip connections)
        self.up3 = self._upconv_block(base_channels * 8, base_channels * 4)
        self.dec3 = self._conv_block(base_channels * 8, base_channels * 4)  # skip from enc3
        self.up2 = self._upconv_block(base_channels * 4, base_channels * 2)
        self.dec2 = self._conv_block(base_channels * 4, base_channels * 2)
        self.up1 = self._upconv_block(base_channels * 2, base_channels)
        self.dec1 = self._conv_block(base_channels * 2, base_channels)

        # Output
        self.final = nn.Sequential(
            nn.Conv2d(base_channels, 16, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(16, 3, 3, 1, 1),
            nn.Tanh(),  # [-1, 1] perturbation
        )

    def _conv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, 1, 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def _upconv_block(self, in_ch, out_ch):
        return nn.Sequential(
            nn.ConvTranspose2d(in_ch, out_ch, 2, 2),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, seed, base_texture):
        """
        Args:
            seed: [B, 64] binary seed
            base_texture: [B, 3, H, W] original character texture
        Returns:
            adversarial_texture: [B, 3, H, W]
        """
        B, C, H, W = base_texture.shape

        # Seed → spatial feature [B, 32, 4, 4]
        seed_feat = self.seed_fc(seed)  # [B, C_s*4*4]
        C_s = self.seed_channels
        seed_small = seed_feat.view(B, C_s, 4, 4)
        # Upsample to texture size for concat with input
        seed_map = F.interpolate(seed_small, size=(H, W), mode='bilinear')
        # Also upsample to 8×8 for bottleneck injection
        seed_bottleneck = F.interpolate(seed_small, size=(8, 8), mode='bilinear')

        # Compress and concat seed map with base texture
        seed_1ch = self.seed_compress(seed_map)   # [B, 1, H, W]
        x0 = torch.cat([base_texture, seed_1ch], dim=1)  # [B, 4, H, W]

        # Encoder
        e1 = self.enc1(x0)                    # [B, C, H, W]
        e2 = self.enc2(F.avg_pool2d(e1, 2))   # [B, 2C, H/2, W/2]
        e3 = self.enc3(F.avg_pool2d(e2, 2))   # [B, 4C, H/4, W/4]
        e4 = self.enc4(F.avg_pool2d(e3, 2))   # [B, 8C, H/8, W/8]

        # Bottleneck: inject seed at deepest level (match spatial size)
        seed_bn = F.interpolate(seed_bottleneck, size=e4.shape[-2:], mode='bilinear')
        B4 = torch.cat([e4, seed_bn], dim=1)
        bn = self.bottleneck(B4)

        # Decoder
        d3 = self.up3(bn)                     # [B, 4C, H/4, W/4]
        d3 = torch.cat([d3, e3], dim=1)       # skip connection
        d3 = self.dec3(d3)

        d2 = self.up2(d3)                     # [B, 2C, H/2, W/2]
        d2 = torch.cat([d2, e2], dim=1)
        d2 = self.dec2(d2)

        d1 = self.up1(d2)                     # [B, C, H, W]
        d1 = torch.cat([d1, e1], dim=1)
        d1 = self.dec1(d1)

        # Output perturbation
        perturbation = self.final(d1) * 0.039  # [-eps, eps]

        return (base_texture + perturbation).clamp(0, 1)


# ============================================================
# 2. OBJ + Render (same as projgrad_attack.py)
# ============================================================
def parse_obj(path):
    with open(path) as f: lines=f.readlines()
    av,au=[],[]; fd=defaultdict(list); cur=None
    for line in lines:
        p=line.strip().split()
        if not p: continue
        if p[0]=="v": av.append([float(x) for x in p[1:4]])
        elif p[0]=="vt": au.append([float(x) for x in p[1:3]])
        elif p[0]=="usemtl": cur=p[1]
        elif p[0]=="f" and cur:
            face=[(int(t.split("/")[0])-1,int(t.split("/")[1])-1 if len(t.split("/"))>1 and t.split("/")[1] else -1) for t in p[1:]]
            if len(face)==4: fd[cur].extend([[face[0],face[1],face[2]],[face[0],face[2],face[3]]])
            elif len(face)>=3: fd[cur].append(face[:3])
    return np.array(av,np.float32),np.array(au,np.float32),dict(fd)


def train_generator(args):
    """Main training loop."""
    # Setup
    glctx = dr.RasterizeCudaContext(device=device)

    # Load OBJ
    vn,un,fd = parse_obj(args.obj_path)
    verts = torch.tensor(vn, dtype=torch.float32, device=device)
    all_uvs_t = torch.tensor(un, dtype=torch.float32, device=device)
    af = []
    for fl in fd.values(): af.extend(fl)
    ft = torch.tensor([[f[0] for f in tri] for tri in af], dtype=torch.int32, device=device)
    ua = torch.zeros(verts.shape[0], 2, dtype=torch.float32, device=device)
    uc = torch.zeros(verts.shape[0], dtype=torch.int32, device=device)
    for tri in af:
        for f in tri:
            vi, uvi = f[0], f[1]
            if uvi >= 0 and uvi < all_uvs_t.shape[0]:
                ua[vi] += all_uvs_t[uvi]; uc[vi] += 1
    ua[uc > 0] = ua[uc > 0] / uc[uc > 0].float().unsqueeze(-1)
    print(f"Mesh: {verts.shape[0]} verts, {ft.shape[0]} faces")

    # Load base texture
    base_tex_path = args.tex_path or "/root/ancient_character/textures/Arm_Base_color.png"
    base_tex = torch.from_numpy(np.array(
        Image.open(base_tex_path).convert("RGB").resize((args.tex_size, args.tex_size), Image.LANCZOS)
    )).float() / 255.0
    base_tex = base_tex.permute(2, 0, 1).to(device).contiguous()  # [3, H, W]
    base_tex_batch = base_tex.unsqueeze(0)  # [1, 3, H, W] — fixed for training

    # Generator
    generator = AdversarialGenerator(seed_bits=64, base_channels=32).to(device)
    n_params = sum(p.numel() for p in generator.parameters())
    print(f"Generator: {n_params:,} params ({n_params*4//1024//1024}MB FP32)")

    # Proxy detector
    from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
    frcnn = fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT).to(device).eval()

    # Optimizer
    optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)

    # Render function (inlined for speed)
    def render_tex(tex_batch, total_views):
        """Batch render views cycling through available textures."""
        imgs = []
        N_tex = tex_batch.shape[0]
        for i in range(total_views):
            tex = tex_batch[i % N_tex]  # cycle through batch
            fv=1.0/math.tan(60*math.pi/180/2)
            P=torch.zeros(4,4,device=device); P[0,0]=P[1,1]=fv; P[2,2]=-0.01; P[3,2]=-1
            e=(torch.rand(1,device=device)*2-1)*0.4
            a=torch.rand(1,device=device)*2*math.pi
            d=1.5+torch.rand(1,device=device)*1.0
            eye=torch.cat([d*torch.cos(e)*torch.sin(a),d*torch.sin(e)+1.0,d*torch.cos(e)*torch.cos(a)])
            at=torch.tensor([-0.15,1.0,0.0],device=device)
            up=torch.tensor([0.,1.,0.],device=device)
            z=F.normalize(eye-at,dim=0)
            x=F.normalize(torch.linalg.cross(up,z),dim=0)
            y=torch.linalg.cross(z,x)
            V=torch.eye(4,device=device); V[0,:3],V[1,:3],V[2,:3]=x,y,z
            V[:3,3]=-torch.tensor([x@eye,y@eye,z@eye],device=device)
            MVP=P@V
            vh=torch.cat([verts,torch.ones(verts.shape[0],1,device=device)],1)
            vc=vh@MVP.T; vc=vc[:,:4]/vc[:,3:4].clamp(min=0.01)
            rast,_=dr.rasterize(glctx,vc.unsqueeze(0),ft,(640,640))
            uvi,_=dr.interpolate(ua.unsqueeze(0),rast,ft)
            col=dr.texture(tex.unsqueeze(0),uvi,filter_mode="linear")[:,:,:,:3]
            alpha=(rast[...,-1:]>0).float()
            yg=torch.linspace(0.2,0.9,640,device=device).view(-1,1,1)
            bg=yg.repeat(1,640,3)*0.4+0.3
            r=(col*alpha+bg.unsqueeze(0)*(1-alpha)).squeeze(0).permute(2,0,1)
            imgs.append(r)
        return torch.stack(imgs, 0)

    # Training
    print(f"\n=== Training {args.steps} steps, {args.views} views/step ===")
    best_loss = float('inf')
    for step in range(args.steps):
        # Generate random seeds
        seeds = torch.stack([_seed_to_tensor(secrets.randbits(64))
                            for _ in range(args.batch_size)]).to(device)

        # Generate adversarial textures
        adv_tex_batch = generator(seeds, base_tex_batch.expand(args.batch_size, -1, -1, -1))

        # Render
        imgs = render_tex(adv_tex_batch, args.batch_size * args.views)

        # FRCNN loss
        total_loss = torch.tensor(0.0, device=device)
        B_total = imgs.shape[0]
        for i in range(B_total):
            out = frcnn([imgs[i]])
            for o in out:
                if (o["labels"] == 1).any():
                    total_loss = total_loss + o["scores"][o["labels"] == 1].sum()
        total_loss = total_loss / B_total

        # Perceptual loss: penalize green channel (most visible) + total variation (smoothness)
        pert = adv_tex_batch - base_tex_batch.expand(args.batch_size, -1, -1, -1)
        # Green penalty: human eye sees G most (0.587 luminance weight)
        green_loss = pert[:, 1, :, :].abs().mean() * 2.0
        # Total variation: penalize high-frequency noise (visible grain)
        tv_h = (pert[:, :, 1:, :] - pert[:, :, :-1, :]).abs().mean()
        tv_w = (pert[:, :, :, 1:] - pert[:, :, :, :-1]).abs().mean()
        tv_loss = (tv_h + tv_w) * 0.5
        perceptual = green_loss + tv_loss

        total_loss = total_loss + 0.15 * perceptual

        optimizer.zero_grad()
        if total_loss > 0:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(generator.parameters(), 1.0)
            optimizer.step()

        if step % args.log_interval == 0 or step == args.steps - 1:
            print(f"  Step {step:5d}: loss={total_loss.item():.4f}")

        if total_loss.item() < best_loss and total_loss.item() > 0:
            best_loss = total_loss.item()
            torch.save({
                'step': step,
                'generator': generator.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss': best_loss,
            }, os.path.join(args.output_dir, 'best_generator.pth'))

        # Validate every N steps
        if (step + 1) % args.val_interval == 0:
            with torch.no_grad():
                val_seed = _seed_to_tensor(42).unsqueeze(0).to(device)
                val_tex = generator(val_seed, base_tex.unsqueeze(0))
                val_imgs = render_tex(val_tex.expand(args.val_views, -1, -1, -1), args.val_views)
                yolo = YOLO("yolov8n.pt")
                conf_sum = 0.0
                for i in range(args.val_views):
                    img = val_imgs[i].permute(1,2,0).cpu().numpy()
                    img = (img.clip(0,1)*255).astype(np.uint8)
                    r = yolo(img, conf=0.10, verbose=False)
                    if r[0].boxes is not None:
                        for b in r[0].boxes:
                            if r[0].names.get(int(b.cls.item()), "?") == "person":
                                conf_sum += float(b.conf.item())
                print(f"  [Val {step:5d}] YOLOv8 person conf: {conf_sum:.3f}")

    # Save final
    torch.save(generator.state_dict(), os.path.join(args.output_dir, 'generator_final.pth'))
    print(f"\nSaved: {args.output_dir}/generator_final.pth")
    return generator


def _seed_to_tensor(seed_int):
    """Convert int to [64] binary tensor."""
    bits = [(seed_int >> i) & 1 for i in range(64)]
    return torch.tensor(bits, dtype=torch.float32)


# ============================================================
# 3. CLI
# ============================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Adversarial Texture Generator")
    parser.add_argument("--obj-path", default="/root/ancient_character/1.obj")
    parser.add_argument("--tex-path", default="/root/ancient_character/textures/Arm_Base_color.png")
    parser.add_argument("--tex-size", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)       # seeds per step
    parser.add_argument("--views", type=int, default=2)            # views per seed
    parser.add_argument("--val-views", type=int, default=16)       # views for validation
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--output-dir", default="/root/generator_output")
    parser.add_argument("--log-interval", type=int, default=100)
    parser.add_argument("--val-interval", type=int, default=500)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    train_generator(args)
