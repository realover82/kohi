# ============================================================
# FixB(12시=0°, CCW) → 12시=0°, CW 의 180° 반전(FixB_180) 파이프라인
# - raw CSV 보존 → 미리보기(as_is/fixB/fixB_180, 동일 샘플) → fixed CSV 생성 → 학습
# ============================================================
import os, math, csv, glob, random, cv2, numpy as np
from PIL import Image

import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

# -----------------
# 경로
# -----------------
ROOT         = "./dea2/gage5/"
DATASET_DIR  = os.path.join(ROOT, "dataset_clean")
TRAIN_DIR    = os.path.join(DATASET_DIR, "train")
TEST_DIR     = os.path.join(DATASET_DIR, "test")

CSV_TRAIN_ORIG = os.path.join(DATASET_DIR, "labels_train.csv")      # 기존(원본)
CSV_TEST_ORIG  = os.path.join(DATASET_DIR, "labels_test.csv")       # 기존(원본)
CSV_TRAIN_RAW  = os.path.join(DATASET_DIR, "labels_train_raw.csv")  # 백업(원본 복사)
CSV_TEST_RAW   = os.path.join(DATASET_DIR, "labels_test_raw.csv")   # 백업(원본 복사)
CSV_TRAIN_FIXED= os.path.join(DATASET_DIR, "labels_train_fixed.csv")# 변환본(FixB_180)
CSV_TEST_FIXED = os.path.join(DATASET_DIR, "labels_test_fixed.csv") # 변환본(FixB_180)

CKPT_DIR       = os.path.join(DATASET_DIR, "checkpoints_r18_fixb180")
os.makedirs(CKPT_DIR, exist_ok=True)

# -----------------
# CSV 정리 + RAW 백업
# -----------------
def sanitize_and_backup(src, dst):
    if not os.path.isfile(src):
        raise FileNotFoundError(src)
    kept=[]
    with open(src,"r",encoding="utf-8") as f:
        r=csv.reader(f); header=next(r)
        for row in r:
            fp=row[0]
            if os.path.isfile(fp):
                kept.append(row)
            else:
                base=os.path.splitext(os.path.basename(fp))[0]
                cand=glob.glob(os.path.join(os.path.dirname(fp), base + ".*"))
                cand=[c for c in cand if os.path.isfile(c)]
                if cand:
                    row[0]=cand[0]; kept.append(row)
    with open(dst,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(header); w.writerows(kept)
    print(f"🧹 CSV 정리+백업: {os.path.basename(src)} → {os.path.basename(dst)} ({len(kept)} rows)")

sanitize_and_backup(CSV_TRAIN_ORIG, CSV_TRAIN_RAW)
if os.path.isfile(CSV_TEST_ORIG):
    sanitize_and_backup(CSV_TEST_ORIG, CSV_TEST_RAW)

# -----------------
# 공통: 화살표 그리기(12시=0°, 시계+ 기준)
# -----------------
def draw_arrow(gray, theta, color=(0,255,0)):
    h,w=gray.shape[:2]; cx,cy=w//2,h//2; R=int(min(h,w)*0.45)
    x2=int(round(cx + math.cos(theta - math.pi/2)*R))
    y2=int(round(cy + math.sin(theta - math.pi/2)*R))
    vis=cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(vis,(cx,cy),(x2,y2),color,2,cv2.LINE_AA,tipLength=0.06)
    return vis

# -----------------
# 원본 CSV에서 동일한 샘플 뽑아 미리보기(as_is / fixB / fixB_180)
# -----------------
def load_rows(csv_path):
    rows=[]
    with open(csv_path,"r",encoding="utf-8") as f:
        r=csv.reader(f); header=next(r)
        for fp, th, s, c in r:
            if os.path.isfile(fp): rows.append((fp,float(th)))
    return rows

def preview_triplet(csv_raw, out_path, seed=42, n=16, grid=4, img_size=224):
    rows=load_rows(csv_raw)
    assert len(rows)>0, "RAW CSV 비어있음"
    random.Random(seed).shuffle(rows)
    rows=rows[:n]

    tiles=[]
    deltas=[]
    for fp, th in rows:
        g=cv2.imread(fp, cv2.IMREAD_GRAYSCALE)
        g=cv2.resize(g,(img_size,img_size))
        # as_is: 원본 라벨을 그대로(= 12시=0°, 반시계+ 라고 가정하지 않음, 그냥 저장된 값)
        as_is = th
        # fixB: 12시=0°, CCW → 12시=0°, CW
        fixB  = (-th) % (2*math.pi)
        # fixB_180: 위에서 180° 반전
        # fixB_180 = (fixB + math.pi) % (2*math.pi)
        #cw를 반전했기에 원위치함
        fixB_180 = (fixB ) % (2*math.pi)

        tiles.append(draw_arrow(g, as_is, (128,128,128)))  # 회색
        tiles.append(draw_arrow(g, fixB, (0,255,0)))       # 녹색
        tiles.append(draw_arrow(g, fixB_180, (0,255,255))) # 노랑-초록

        # 각도차가 정확히 180°인지 확인(수치)
        d = abs(((fixB_180 - fixB + math.pi)%(2*math.pi)) - math.pi) # 최소각
        deltas.append(d*180/math.pi)

    # 3열 × (n 행) 캔버스
    H,W,_=tiles[0].shape
    canvas=np.zeros((H*n, W*3, 3), np.uint8)
    for i in range(n):
        canvas[i*H:(i+1)*H, 0*W:1*W]=tiles[3*i+0] # as_is
        canvas[i*H:(i+1)*H, 1*W:2*W]=tiles[3*i+1] # fixB
        canvas[i*H:(i+1)*H, 2*W:3*W]=tiles[3*i+2] # fixB_180
    cv2.imwrite(out_path, canvas)
    print("🖼  미리보기 저장:", out_path)
    print(f"   · 평균 |fixB_180 - fixB| = {np.mean(deltas):.2f}° (이상적: 180°)")

out_triplet = os.path.join(DATASET_DIR, "_label_preview_triplet.jpg")
preview_triplet(CSV_TRAIN_RAW, out_triplet, seed=51, n=16)

# -----------------
# RAW → FIXED(FixB_180) 변환본 생성
# -----------------
def write_fixed_from_raw(csv_raw, csv_fixed):
    rows=load_rows(csv_raw)
    out=[]
    for fp, th in rows:
        th_fixed = (-th + math.pi) % (2*math.pi)  # FixB_180
        out.append([fp, f"{th_fixed:.9f}", f"{math.sin(th_fixed):.9f}", f"{math.cos(th_fixed):.9f}"])
    with open(csv_fixed,"w",newline="",encoding="utf-8") as f:
        w=csv.writer(f); w.writerow(["filepath","theta_rad","sin","cos"]); w.writerows(out)
    print("✅ FIXED CSV 작성:", os.path.basename(csv_fixed), f"({len(out)} rows)")

write_fixed_from_raw(CSV_TRAIN_RAW, CSV_TRAIN_FIXED)
if os.path.isfile(CSV_TEST_RAW):
    write_fixed_from_raw(CSV_TEST_RAW,  CSV_TEST_FIXED)

# -----------------
# Dataset / Model
# -----------------
class DialDataset(Dataset):
    def __init__(self, csv_path, img_size=224, augment=True):
        self.items=[]
        with open(csv_path,"r",encoding="utf-8") as f:
            r=csv.reader(f); next(r)
            for fp, th, s, c in r:
                if os.path.isfile(fp):
                    self.items.append((fp,float(th),float(s),float(c)))
        if not self.items:
            raise RuntimeError(f"유효 항목 0: {csv_path}")
        augs=[]
        if augment: augs += [T.ColorJitter(brightness=0.2, contrast=0.2)]
        self.tfm=T.Compose(augs+[
            T.Resize((img_size,img_size)),
            T.ToTensor(),
            T.Lambda(lambda x: x.expand(3,-1,-1)),
            T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ])
    def __len__(self): return len(self.items)
    def __getitem__(self,i):
        fp, th, s, c=self.items[i]
        x=self.tfm(Image.open(fp).convert("L"))
        y=torch.tensor([s,c],dtype=torch.float32) # [sin,cos]
        return x,y

class AngleHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)
    def forward(self,x):
        y=self.backbone(x)
        return y/(y.norm(dim=1,keepdim=True)+1e-8)

def angle_mae_deg(p,t):
    dot=(p*t).sum(dim=1).clamp(-1,1)
    ang=torch.acos(dot)*180.0/math.pi
    return ang.mean().item()

# -----------------
# 학습 (FIXED CSV 사용)
# -----------------
def train_r18(csv_tr=CSV_TRAIN_FIXED, csv_va=CSV_TEST_FIXED,
              img_size=224, epochs=30, batch=32, lr=3e-4):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ds_tr=DialDataset(csv_tr, img_size, augment=True)
    try:
        ds_va=DialDataset(csv_va, img_size, augment=False)
    except:
        print("⚠️ val CSV 없음 → train 15%를 val로 사용")
        n=len(ds_tr); k=max(1,int(n*0.15))
        ds_va=torch.utils.data.Subset(ds_tr, list(range(k)))
        ds_tr=torch.utils.data.Subset(ds_tr, list(range(k,n)))

    dl_tr=DataLoader(ds_tr, batch_size=batch, shuffle=True,  num_workers=0, pin_memory=True)
    dl_va=DataLoader(ds_va, batch_size=batch, shuffle=False, num_workers=0, pin_memory=True)

    model=AngleHead().to(device)
    opt=torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    cos=nn.CosineEmbeddingLoss(); mse=nn.MSELoss()
    sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

    best=1e9
    for ep in range(1,epochs+1):
        model.train(); tr=0.0
        for x,y in dl_tr:
            x,y=x.to(device), y.to(device)
            p=model(x); tgt=torch.ones(p.size(0), device=device)
            loss=cos(p,y,tgt)+0.1*mse(p,y)
            opt.zero_grad(); loss.backward(); opt.step()
            tr+=loss.item()*x.size(0)
        tr/=len(dl_tr.dataset); sch.step()

        model.eval(); va=0.0; mae=0.0; n=0
        with torch.no_grad():
            for x,y in dl_va:
                x,y=x.to(device), y.to(device)
                p=model(x); tgt=torch.ones(p.size(0), device=device)
                loss=cos(p,y,tgt)+0.1*mse(p,y)
                va+=loss.item()*x.size(0); mae+=angle_mae_deg(p,y)*x.size(0); n+=x.size(0)
        va/=max(1,n); mae/=max(1,n)
        print(f"[{ep:02d}] train {tr:.4f} | val {va:.4f} | val-MAE {mae:.2f}°")

        torch.save(model.state_dict(), os.path.join(CKPT_DIR, f"ep{ep:02d}.pth"))
        if mae<best:
            best=mae; torch.save(model.state_dict(), os.path.join(CKPT_DIR,"best.pth"))
    print(f"✅ best val-MAE = {best:.2f}°  → {os.path.join(CKPT_DIR,'best.pth')}")
    return os.path.join(CKPT_DIR,"best.pth")

best_path = train_r18(epochs=30, batch=32, lr=3e-4)
print("BEST:", best_path)

# -----------------
# 추론 예시(오버레이) – FIXED 기준
# -----------------
def load_for_infer(ckpt):
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=AngleHead().to(device)
    sd=torch.load(ckpt, map_location=device)
    model.load_state_dict(sd, strict=True); model.eval()
    tfm=T.Compose([
        T.Resize((224,224)),
        T.ToTensor(),
        T.Lambda(lambda x: x.expand(3,-1,-1)),
        T.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
    ])
    return model, tfm, device

def infer_one(model, tfm, device, img_path, theta_zero_rad=0.0, ticks_per_rev=100, mm_per_rev=1.0):
    x=tfm(Image.open(img_path).convert("L")).unsqueeze(0).to(device)
    with torch.no_grad():
        y=model(x)[0].cpu().numpy()
    sinp,cosp=float(y[0]), float(y[1])
    theta=(math.atan2(sinp,cosp))%(2*math.pi)     # 12시=0°, 시계(+)
    delta=(theta-theta_zero_rad)%(2*math.pi)
    ticks=delta/(2*math.pi)*ticks_per_rev
    mm   =delta/(2*math.pi)*mm_per_rev
    return theta, ticks, mm

def save_overlay(img_path, theta_deg, value_mm, out_path):
    g=cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    h,w=g.shape[:2]; cx,cy=w//2,h//2; R=int(min(h,w)*0.45)
    th=math.radians(theta_deg)
    x2=int(round(cx+math.cos(th-math.pi/2)*R))
    y2=int(round(cy+math.sin(th-math.pi/2)*R))
    vis=cv2.cvtColor(g,cv2.COLOR_GRAY2BGR)
    cv2.arrowedLine(vis,(cx,cy),(x2,y2),(0,255,0),2,cv2.LINE_AA,tipLength=0.06)
    cv2.putText(vis,f"{value_mm:.3f} mm",(10,30),
                cv2.FONT_HERSHEY_SIMPLEX,0.9,(0,255,0),2,cv2.LINE_AA)
    cv2.imwrite(out_path,vis)

# 샘플 1장
pngs=glob.glob(os.path.join(TEST_DIR,"*.png")) or glob.glob(os.path.join(TRAIN_DIR,"*.png"))
if pngs:
    model,tfm,device=load_for_infer(best_path)
    sample=pngs[0]
    theta_zero_deg=0.0  # 0점 보정 필요시 수정
    theta,ticks,mm=infer_one(model,tfm,device,sample,
                              theta_zero_rad=math.radians(theta_zero_deg),
                              ticks_per_rev=100, mm_per_rev=1.0)
    out_prev=os.path.join(DATASET_DIR,"_infer_preview.jpg")
    save_overlay(sample, theta*180/math.pi, mm, out_prev)
    print(f"🔎 예측: θ={theta*180/math.pi:.2f}°, value={mm:.3f} mm")
    print("🖼  저장:", out_prev)
else:
    print("⚠️ 샘플 이미지 없음")

#결과 없다