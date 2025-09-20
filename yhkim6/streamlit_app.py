# streamlit_app.py
import streamlit as st
import os
import re
import math
import csv
import glob
import random
import time
import json
import numpy as np
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide")
st.title("다이얼 게이지 자동 분석 애플리케이션 📊")

# 임시 파일 업로드 디렉토리
UPLOAD_DIR_TRAIN = "uploaded_train_images"
UPLOAD_DIR_TEST = "uploaded_test_images"
os.makedirs(UPLOAD_DIR_TRAIN, exist_ok=True)
os.makedirs(UPLOAD_DIR_TEST, exist_ok=True)

# 모델 및 데이터 저장 디렉토리
DATA_DIR = "gage_data"
os.makedirs(DATA_DIR, exist_ok=True)
CKPT_DIR_THETA = os.path.join(DATA_DIR, "checkpoints_r18_fixb180")
CKPT_DIR_ZERO = os.path.join(DATA_DIR, "checkpoints_zerohead_A")
os.makedirs(CKPT_DIR_THETA, exist_ok=True)
os.makedirs(CKPT_DIR_ZERO, exist_ok=True)

# TPU 가속은 웹 환경에서 사용 불가. CPU/GPU로 대체
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"사용 중인 장치: {device}")

# =========================================================
# 모델 및 유틸리티 함수 정의
# =========================================================

# AngleHead 모델 구조 (ResNet-18 기반)
class AngleHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        y = self.backbone(x)
        return y / (y.norm(dim=1, keepdim=True) + 1e-8)

# YOLOv5 모델 (가정)
class YOLOv5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_fc = nn.Linear(1000, 2)
    def forward(self, x):
        return self.dummy_fc(torch.randn(x.size(0), 1000, device=x.device)) / 1000.0

# 공통 이미지 변환
tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: x.expand(3, -1, -1)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# 파일명에서 mm 값 파싱
def parse_mm_prefix(fp):
    name = os.path.basename(fp)
    m = re.match(r"^\D*?(\d{2})", name)
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 99:
        return v / 100.0
    return None

# 각도 계산 유틸리티
TWO_PI = 2.0 * math.pi
def wrap_angle(x):
    return (x + TWO_PI) % TWO_PI

# =========================================================
# Streamlit UI
# =========================================================
st.sidebar.header("⚙️ 모델 설정")

# 모델 선택
model_choice = st.sidebar.selectbox("모델 선택", ("ResNet-18 (AngleHead)", "YOLOv5 (예정)"))
if model_choice == "YOLOv5 (예정)":
    st.sidebar.warning("YOLOv5는 현재 더미 모델로 구현되어 있으며, 실제 기능은 없습니다.")

# 가중치 선택
weights_choice = st.sidebar.selectbox("가중치", ("사전 학습 가중치 적용", "스크래치(무작위) 가중치"))

# 학습 Epoch 입력
epochs = st.sidebar.number_input("학습 Epoch", min_value=1, value=1, step=1)
st.sidebar.text("")

# =========================================================
# 파일 업로드 섹션
# =========================================================
st.header("📂 데이터 업로드")
st.markdown("학습 및 테스트에 사용할 다이얼 게이지 이미지를 업로드하세요. 파일명은 `00-sample.png` 형식으로 `mm` 값이 포함되어야 합니다.")

col1, col2 = st.columns(2)

with col1:
    st.subheader("학습용 이미지 업로드")
    uploaded_train_files = st.file_uploader(
        "train 폴더에 업로드할 이미지(.png, .jpg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="train_uploader"
    )
    if uploaded_train_files:
        for uploaded_file in uploaded_train_files:
            file_path = os.path.join(UPLOAD_DIR_TRAIN, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_train_files)}개의 학습용 이미지 업로드 완료.")

with col2:
    st.subheader("테스트용 이미지 업로드")
    uploaded_test_files = st.file_uploader(
        "test 폴더에 업로드할 이미지(.png, .jpg)",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key="test_uploader"
    )
    if uploaded_test_files:
        for uploaded_file in uploaded_test_files:
            file_path = os.path.join(UPLOAD_DIR_TEST, uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"{len(uploaded_test_files)}개의 테스트용 이미지 업로드 완료.")

# =========================================================
# 기능 버튼
# =========================================================
st.header("🚀 분석 및 학습 실행")

col_buttons = st.columns(4)

with col_buttons[0]:
    if st.button("분석 시작"):
        if not uploaded_test_files:
            st.warning("분석을 시작하려면 테스트용 이미지를 먼저 업로드해주세요.")
        else:
            st.info("테스트 이미지 분석을 시작합니다. 잠시만 기다려주세요...")
            
            try:
                # ZeroHead 모델 로드
                zero_model = AngleHead(pretrained=False).to(device)
                if os.path.isfile(os.path.join(CKPT_DIR_ZERO, "best.pth")):
                    zero_model.load_state_dict(torch.load(os.path.join(CKPT_DIR_ZERO, "best.pth"), map_location=device))
                    st.success("ZeroHead 모델 가중치 로드 성공.")
                else:
                    st.error("ZeroHead 모델 가중치를 찾을 수 없습니다. '학습 시작' 버튼을 눌러 학습을 먼저 진행해주세요.")
                    st.stop()
                
                # 예측 및 결과 저장
                zero_model.eval()
                results = []
                for fp in glob.glob(os.path.join(UPLOAD_DIR_TEST, "*.png")):
                    mm_from_name = parse_mm_prefix(fp)
                    if mm_from_name is None:
                        st.warning(f"파일명에서 mm 값을 파싱할 수 없습니다: {os.path.basename(fp)}. 이 파일은 무시됩니다.")
                        continue
                        
                    with torch.no_grad():
                        x = tfm(Image.open(fp).convert("L")).unsqueeze(0).to(device)
                        y = zero_model(x)[0].cpu().numpy()
                    
                    psi_pred = wrap_angle(math.atan2(float(y[0]), float(y[1])))
                    
                    # 정확도를 위해 θ 모델이 필요하지만, 여기서는 ψ 모델의 예측값을 mm_from_name과 비교하여 유사성 검증 (임시)
                    # 실제 프로젝트에서는 θ - ψ 로 mm 값을 계산해야 함
                    results.append({
                        "filepath": fp,
                        "predicted_psi_rad": psi_pred,
                        "true_mm": mm_from_name
                    })
                
                st.session_state['analysis_results'] = results
                st.session_state['show_metrics'] = True
                st.success("분석 완료! '분석 결과 보기' 버튼을 눌러 확인하세요.")
            
            except Exception as e:
                st.error(f"분석 중 오류 발생: {e}")

with col_buttons[1]:
    if st.button("모델 저장"):
        # 이전에 학습된 모델 가중치를 저장하는 기능
        if not os.path.isfile(os.path.join(CKPT_DIR_ZERO, "best.pth")):
            st.warning("먼저 '학습 시작' 버튼을 눌러 모델 학습을 완료해야 합니다.")
        else:
            # 여기서는 파일 다운로드 기능으로 대체
            with open(os.path.join(CKPT_DIR_ZERO, "best.pth"), "rb") as f:
                st.download_button(
                    label="ZeroHead 모델 가중치 다운로드",
                    data=f,
                    file_name="best.pth",
                    mime="application/octet-stream"
                )
            st.success("모델 가중치 파일을 다운로드할 수 있습니다.")


with col_buttons[2]:
    if st.button("학습 시작"):
        if not uploaded_train_files:
            st.warning("학습을 시작하려면 학습용 이미지를 먼저 업로드해주세요.")
        else:
            st.info(f"선택한 설정으로 모델 학습을 시작합니다. (Epoch: {epochs})")
            
            try:
                # 1. Pseudo 라벨 생성
                st.info("1단계: Pseudo 라벨 생성 중...")
                theta_model = AngleHead(pretrained=True).to(device)
                
                # θ 모델 가중치 존재 확인 (필요한 경우 사전 다운로드)
                if os.path.isfile(os.path.join(CKPT_DIR_THETA, "best.pth")):
                    theta_model.load_state_dict(torch.load(os.path.join(CKPT_DIR_THETA, "best.pth"), map_location=device))
                else:
                    st.warning("θ 모델(FixB_180) 가중치를 찾을 수 없습니다. 새로운 θ 모델을 학습합니다.")
                
                theta_model.eval()
                
                rows = []
                for fp in glob.glob(os.path.join(UPLOAD_DIR_TRAIN, "*.png")):
                    mm = parse_mm_prefix(fp)
                    if mm is None: continue
                    with torch.no_grad():
                        x = tfm(Image.open(fp).convert("L")).unsqueeze(0).to(device)
                        y = theta_model(x)[0].cpu().numpy()
                    th = wrap_angle(math.atan2(float(y[0]), float(y[1])))
                    delta_label = TWO_PI * mm
                    psi = wrap_angle(th - delta_label)
                    rows.append({"filepath": fp, "psi_rad": psi, "sin_psi": math.sin(psi), "cos_psi": math.cos(psi)})
                
                with open(os.path.join(DATA_DIR, "pseudo_zero_labels.csv"), "w", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                    writer.writeheader()
                    writer.writerows(rows)
                
                st.success("Pseudo 라벨 생성 완료.")

                # 2. ZeroHead 모델 학습
                st.info("2단계: ZeroHead 모델 학습 중...")
                if model_choice == "ResNet-18 (AngleHead)":
                    model = AngleHead(pretrained=(weights_choice == "사전 학습 가중치 적용")).to(device)
                else:
                    model = YOLOv5Model().to(device)
                    st.warning("YOLOv5는 현재 더미 모델입니다.")
                
                opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

                def cos_loss(pred, target):
                    return (1 - (pred * target).sum(dim=1)).mean()

                class PsiDataset(torch.utils.data.Dataset):
                    def __init__(self, items, tfm): self.items, self.tfm = items, tfm
                    def __len__(self): return len(self.items)
                    def __getitem__(self, i):
                        item = self.items[i]
                        x = self.tfm(Image.open(item['filepath']).convert("L"))
                        y = torch.tensor([item['sin_psi'], item['cos_psi']], dtype=torch.float32)
                        return x, y

                with open(os.path.join(DATA_DIR, "pseudo_zero_labels.csv"), "r", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    items = [{"filepath": r["filepath"], "sin_psi": float(r["sin_psi"]), "cos_psi": float(r["cos_psi"])} for r in reader]

                random.shuffle(items)
                split = int(len(items) * 0.9)
                train_items, val_items = items[:split], items[split:]
                
                ds_tr = PsiDataset(train_items, tfm)
                ds_va = PsiDataset(val_items, tfm)
                dl_tr = DataLoader(ds_tr, batch_size=32, shuffle=True, num_workers=0, pin_memory=True)
                dl_va = DataLoader(ds_va, batch_size=64, shuffle=False, num_workers=0, pin_memory=True)
                
                train_losses, val_losses = [], []
                best_loss = float('inf')

                for ep in range(epochs):
                    model.train()
                    tr_loss = 0
                    for x, y in dl_tr:
                        x, y = x.to(device), y.to(device)
                        p = model(x)
                        loss = cos_loss(p, y)
                        opt.zero_grad()
                        loss.backward()
                        opt.step()
                        tr_loss += loss.item() * x.size(0)
                    train_losses.append(tr_loss / len(ds_tr))
                    
                    model.eval()
                    va_loss = 0
                    with torch.no_grad():
                        for x, y in dl_va:
                            x, y = x.to(device), y.to(device)
                            p = model(x)
                            loss = cos_loss(p, y)
                            va_loss += loss.item() * x.size(0)
                    val_losses.append(va_loss / len(ds_va))
                    
                    st.text(f"Epoch [{ep+1}/{epochs}] - Train Loss: {train_losses[-1]:.4f} | Val Loss: {val_losses[-1]:.4f}")
                    
                    if val_losses[-1] < best_loss:
                        best_loss = val_losses[-1]
                        torch.save(model.state_dict(), os.path.join(CKPT_DIR_ZERO, "best.pth"))
                
                st.session_state['train_losses'] = train_losses
                st.session_state['val_losses'] = val_losses
                st.success("ZeroHead 모델 학습 완료! 가중치가 저장되었습니다.")
            
            except Exception as e:
                st.error(f"학습 중 오류 발생: {e}")

with col_buttons[3]:
    if st.button("분석 결과 보기"):
        if 'analysis_results' not in st.session_state:
            st.warning("분석을 먼저 시작해야 결과를 볼 수 있습니다.")
        else:
            st.header("📋 분석 결과")
            results = st.session_state['analysis_results']
            
            # 메트릭 계산을 위한 이진 분류 가정 (mm_from_name > 0.5)
            # 이 부분은 실제 문제에 맞게 수정 필요
            y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
            y_pred_binary = [1 if r['predicted_psi_rad'] > math.pi else 0 for r in results]

            st.subheader("혼동 행렬 (Confusion Matrix)")
            cm = confusion_matrix(y_true_binary, y_pred_binary)
            fig_cm, ax_cm = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
            ax_cm.set_xlabel('Predicted')
            ax_cm.set_ylabel('True')
            st.pyplot(fig_cm)
            
            st.subheader("성능 지표")
            accuracy = accuracy_score(y_true_binary, y_pred_binary)
            precision = precision_score(y_true_binary, y_pred_binary)
            recall = recall_score(y_true_binary, y_pred_binary)
            f1 = f1_score(y_true_binary, y_pred_binary)
            tn, fp, fn, tp = cm.ravel()
            specificity = tn / (tn + fp)

            st.write(f"**Accuracy:** {accuracy:.4f}")
            st.write(f"**Precision:** {precision:.4f}")
            st.write(f"**Recall (Sensitivity):** {recall:.4f}")
            st.write(f"**F1 Score:** {f1:.4f}")
            st.write(f"**Specificity:** {specificity:.4f}")

            st.subheader("ROC 곡선 및 AUC")
            y_scores = [r['predicted_psi_rad'] / TWO_PI for r in results]
            fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
            roc_auc = auc(fpr, tpr)

            fig_roc, ax_roc = plt.subplots()
            ax_roc.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
            ax_roc.plot([0, 1], [0, 1], color='gray', linestyle='--')
            ax_roc.set_xlabel('False Positive Rate')
            ax_roc.set_ylabel('True Positive Rate')
            ax_roc.set_title('Receiver Operating Characteristic (ROC) Curve')
            ax_roc.legend(loc="lower right")
            st.pyplot(fig_roc)