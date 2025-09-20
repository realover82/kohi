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
import zipfile
import shutil
from torchsummary import summary
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as ReportLabImage
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

# =========================================================
# 기본 설정
# =========================================================
st.set_page_config(layout="wide")
st.title("다이얼 게이지 자동 분석 애플리케이션 📊")

# 임시 파일 업로드 디렉토리
UPLOAD_DIR = "uploaded_files"
os.makedirs(UPLOAD_DIR, exist_ok=True)
UPLOAD_DIR_TRAIN = os.path.join(UPLOAD_DIR, "train")
UPLOAD_DIR_TEST = os.path.join(UPLOAD_DIR, "test")
os.makedirs(UPLOAD_DIR_TRAIN, exist_ok=True)
os.makedirs(UPLOAD_DIR_TEST, exist_ok=True)

# 모델 및 데이터 저장 디렉토리
DATA_DIR = "gage_data"
os.makedirs(DATA_DIR, exist_ok=True)
CKPT_DIR_ZERO = os.path.join(DATA_DIR, "checkpoints_zerohead_A")
os.makedirs(CKPT_DIR_ZERO, exist_ok=True)

# TPU 가속은 웹 환경에서 사용 불가. CPU/GPU로 대체
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
st.sidebar.info(f"사용 중인 장치: {device}")

# =========================================================
# 모델 및 유틸리티 함수 정의
# =========================================================

class AngleHead(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        weights = tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = tvm.resnet18(weights=weights)
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 2)

    def forward(self, x):
        y = self.backbone(x)
        return y / (y.norm(dim=1, keepdim=True) + 1e-8)

class YOLOv5Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.dummy_fc = nn.Linear(1000, 2)
    def forward(self, x):
        return self.dummy_fc(torch.randn(x.size(0), 1000, device=x.device)) / 1000.0

tfm = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Lambda(lambda x: x.expand(3, -1, -1)),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

def parse_mm_prefix(fp):
    name = os.path.basename(fp)
    m = re.match(r"^\D*?(\d{1,2})", name)
    if not m:
        return None
    v = int(m.group(1))
    if 0 <= v <= 99:
        return v / 100.0
    return None

TWO_PI = 2.0 * math.pi
def wrap_angle(x):
    return (x + TWO_PI) % TWO_PI

# PsiDataset 클래스를 전역으로 분리
class PsiDataset(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm
    def __len__(self): return len(self.items)
    def __getitem__(self, i):
        item = self.items[i]
        
        filename = os.path.basename(item['filepath'])
        file_path = os.path.join(UPLOAD_DIR_TRAIN, filename) 
        
        if not os.path.isfile(file_path):
            raise FileNotFoundError(f"파일을 찾을 수 없습니다: {file_path}. 업로드된 파일의 유효성을 확인해주세요.")

        x = self.tfm(Image.open(file_path).convert("L"))
        y = torch.tensor([item['sin_psi'], item['cos_psi']], dtype=torch.float32)
        return x, y

# PDF 리포트 생성 함수
def create_pdf_report(filename, results, cm_fig, roc_fig_path):
    doc = SimpleDocTemplate(filename, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("<b>다이얼 게이지 분석 보고서</b>", styles['Heading1']))
    story.append(Spacer(1, 0.2 * inch))

    # 성능 지표 추가
    story.append(Paragraph("<b>1. 성능 지표</b>", styles['Heading2']))
    story.append(Spacer(1, 0.1 * inch))
    
    y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
    y_pred_binary = [1 if r['predicted_psi_rad'] > math.pi else 0 for r in results]
    accuracy = accuracy_score(y_true_binary, y_pred_binary)
    precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
    recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
    f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    
    cm = confusion_matrix(y_true_binary, y_pred_binary)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        specificity = 0

    story.append(Paragraph(f"Accuracy: {accuracy:.4f}", styles['Normal']))
    story.append(Paragraph(f"Precision: {precision:.4f}", styles['Normal']))
    story.append(Paragraph(f"Recall (Sensitivity): {recall:.4f}", styles['Normal']))
    story.append(Paragraph(f"F1 Score: {f1:.4f}", styles['Normal']))
    story.append(Paragraph(f"Specificity: {specificity:.4f}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))
    
    # 혼동 행렬 이미지 추가
    story.append(Paragraph("<b>2. 혼동 행렬 (Confusion Matrix)</b>", styles['Heading2']))
    cm_img_path = "cm_temp.png"
    cm_fig.savefig(cm_img_path)
    story.append(ReportLabImage(cm_img_path, width=4*inch, height=4*inch))
    story.append(Spacer(1, 0.2 * inch))

    # ROC 곡선 이미지 추가
    story.append(Paragraph("<b>3. ROC 곡선 (ROC Curve)</b>", styles['Heading2']))
    story.append(ReportLabImage(roc_fig_path, width=4*inch, height=4*inch))

    doc.build(story)
    
# =========================================================
# Streamlit UI
# =========================================================
st.sidebar.header("⚙️ 모델 설정")

model_choice = st.sidebar.selectbox("모델 선택", ("ResNet-18 (AngleHead)", "YOLOv5 (예정)"))
if model_choice == "YOLOv5 (예정)":
    st.sidebar.warning("YOLOv5는 현재 더미 모델로 구현되어 있으며, 실제 기능은 없습니다.")

load_mode = st.sidebar.radio("모델 가중치 로드", ("파인튜닝", "무작위 초기화"))
finetune_layer = st.sidebar.selectbox("파인튜닝 레이어 선택", ("마지막 레이어만", "모든 레이어"))

st.sidebar.text("")

# =========================================================
# 파일 업로드 섹션
# =========================================================
st.header("📂 데이터 업로드")
st.markdown("성능 테스트에 사용할 다이얼 게이지 이미지를 업로드하세요. 파일명은 `0-sample.png` 또는 `00-sample.png` 형식으로 `mm` 값이 포함되어야 합니다.")

uploaded_test_files = st.file_uploader("테스트용 이미지 업로드", type=["png", "jpg", "jpeg"], accept_multiple_files=True, key="test_uploader")
if uploaded_test_files:
    for file in glob.glob(os.path.join(UPLOAD_DIR_TEST, "*")):
        os.remove(file)
    for uploaded_file in uploaded_test_files:
        file_path = os.path.join(UPLOAD_DIR_TEST, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
    st.success(f"{len(uploaded_test_files)}개의 테스트용 이미지 업로드 완료.")

if load_mode == "파인튜닝":
    st.subheader("파인튜닝 모델 업로드")
    uploaded_best_pth = st.file_uploader("best.pth 파일 업로드", type=["pth"], key="pth_uploader")
    if uploaded_best_pth:
        best_pth_path = os.path.join(CKPT_DIR_ZERO, "best.pth")
        with open(best_pth_path, "wb") as f:
            f.write(uploaded_best_pth.getbuffer())
        st.success("best.pth 파일 업로드 완료.")

# =========================================================
# 기능 버튼 (수직 나열)
# =========================================================
st.header("🚀 성능 테스트 실행")

# 모델 구조 보기 버튼
if st.button("모델 구조 보기"):
    st.info("모델 구조를 분석 중입니다...")
    try:
        model = AngleHead(pretrained=False).to(device)
        if load_mode == "파인튜닝" and os.path.isfile(os.path.join(CKPT_DIR_ZERO, "best.pth")):
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR_ZERO, "best.pth"), map_location=device))
            st.success("파인튜닝 모드: 기존 best.pth 가중치를 로드했습니다.")
        
        # --- 수정된 부분: 콘솔 출력을 캡처하여 Streamlit에 표시 ---
        import io
        from contextlib import redirect_stdout
        
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            summary(model, (3, 224, 224), device=str(device))
        
        st.subheader("모델 구조 상세")
        st.code(buffer.getvalue())
        # -------------------------------------------------------------
        
    except Exception as e:
        st.error(f"모델 구조 분석 중 오류 발생: {e}")
        
# 재학습 버튼 (기능 미구현 상태임을 명시)
if st.button("재학습 시작"):
    st.info("재학습 기능을 준비 중입니다...")
    st.warning("재학습 기능은 현재 미구현 상태입니다. 로컬 환경에서 학습 스크립트를 사용해주세요.")

# 변형된 레이어나 가중치를 미리보기 버튼
if st.button("변형된 레이어 미리보기"):
    st.info("파인튜닝 레이어 설정을 미리보기 합니다...")
   
    try:
        model = AngleHead(pretrained=False).to(device)
        if load_mode == "파인튜닝" and os.path.isfile(os.path.join(CKPT_DIR_ZERO, "best.pth")):
            model.load_state_dict(torch.load(os.path.join(CKPT_DIR_ZERO, "best.pth"), map_location=device))
            st.success("파인튜닝 모드: 기존 best.pth 가중치를 로드했습니다.")
        
        if finetune_layer == "마지막 레이어만":
            for name, param in model.named_parameters():
                if 'fc' in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
        elif finetune_layer == "모든 레이어":
            for param in model.parameters():
                param.requires_grad = True
        
        st.session_state['finetune_settings'] = {
            'layer': finetune_layer,
            'trainable_layers': [name for name, param in model.named_parameters() if param.requires_grad]
        }
        
        st.subheader("파인튜닝 설정 적용 결과")
        st.write("다음 레이어가 학습 가능하도록 설정되었습니다:")
        for name in st.session_state['finetune_settings']['trainable_layers']:
            st.write(f"- {name}")
            
    except Exception as e:
        st.error(f"파인튜닝 설정 적용 중 오류 발생: {e}")
    
if st.button("분석 시작"):
    image_extensions = ['*.png', '*.jpg', '*.jpeg']
    test_files = []
    for ext in image_extensions:
        test_files.extend(glob.glob(os.path.join(UPLOAD_DIR_TEST, ext)))

    if not test_files:
        st.warning("분석을 시작하려면 테스트용 이미지를 먼저 업로드해주세요.")
    else:
        st.info("테스트 이미지 분석을 시작합니다. 잠시만 기다려주세요...")
        
        try:
            if model_choice == "ResNet-18 (AngleHead)":
                zero_model = AngleHead(pretrained=False).to(device)
            else:
                zero_model = YOLOv5Model().to(device)

            if load_mode == "파인튜닝" and os.path.isfile(os.path.join(CKPT_DIR_ZERO, "best.pth")):
                zero_model.load_state_dict(torch.load(os.path.join(CKPT_DIR_ZERO, "best.pth"), map_location=device))
                st.info("파인튜닝 모드: 기존 best.pth 가중치를 로드했습니다.")
            elif load_mode == "파인튜닝":
                st.error("파인튜닝 모드입니다. best.pth 파일을 먼저 업로드해야 합니다.")
                st.stop()
            else:
                st.info("무작위 초기화 모드: 새로운 모델을 초기화했습니다.")

            zero_model.eval()
            results = []
            for fp in test_files:
                mm_from_name = parse_mm_prefix(fp)
                if mm_from_name is None: 
                    st.warning(f"파일명에서 mm 값을 파싱할 수 없습니다: {os.path.basename(fp)}. 이 파일은 분석에서 제외됩니다.")
                    continue
                
                with torch.no_grad():
                    x = tfm(Image.open(fp).convert("L")).unsqueeze(0).to(device)
                    y = zero_model(x)[0].cpu().numpy()
                
                psi_pred = wrap_angle(math.atan2(float(y[0]), float(y[1])))
                
                results.append({
                    "filepath": fp,
                    "predicted_psi_rad": psi_pred,
                    "true_mm": mm_from_name
                })
            
            if not results:
                st.error("분석할 유효한 파일이 없습니다. 파일명 형식을 확인해주세요.")
                st.stop()

            st.session_state['analysis_results'] = results
            st.session_state['load_mode'] = load_mode
            st.session_state['model_choice'] = model_choice
            st.success("분석 완료! '분석 결과 보기' 버튼을 눌러 확인하세요.")
        
        except Exception as e:
            st.error(f"분석 중 오류 발생: {e}")

if st.button("분석 결과 보기"):
    if 'analysis_results' not in st.session_state:
        st.warning("분석을 먼저 시작해야 결과를 볼 수 있습니다.")
    else:
        st.header(f"📋 분석 결과 ({st.session_state.get('model_choice', '모델')} - {st.session_state.get('load_mode', '모드')})")
        results = st.session_state['analysis_results']
        
        y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in results]
        y_pred_binary = [1 if r['predicted_psi_rad'] > math.pi else 0 for r in results]

        st.subheader("혼동 행렬 (Confusion Matrix)")
        cm = confusion_matrix(y_true_binary, y_pred_binary)
        
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
        elif cm.shape == (1, 1):
            if y_true_binary[0] == 1:
                tn, fp, fn, tp = 0, 0, 0, cm[0][0]
            else:
                tn, fp, fn, tp = cm[0][0], 0, 0, 0
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
            st.warning("혼동 행렬의 크기가 예상과 다릅니다. 성능 지표를 계산할 수 없습니다.")
        
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
        ax_cm.set_xlabel('Predicted')
        ax_cm.set_ylabel('True')
        st.pyplot(fig_cm)
        
        st.subheader("성능 지표")
        accuracy = accuracy_score(y_true_binary, y_pred_binary)
        precision = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        recall = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

        st.write(f"**Accuracy:** {accuracy:.4f}")
        st.write(f"**Precision:** {precision:.4f}")
        st.write(f"**Recall (Sensitivity):** {recall:.4f}")
        st.write(f"**F1 Score:** {f1:.4f}")
        st.write(f"**Specificity:** {specificity:.4f}")

        st.subheader("ROC 곡선 및 AUC")
        y_scores = [r['predicted_psi_rad'] / TWO_PI for r in results]
        fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
        roc_auc = auc(fpr, tpr)

        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
        fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
        fig_roc.update_layout(
            title='Receiver Operating Characteristic (ROC) Curve',
            xaxis_title='False Positive Rate',
            yaxis_title='True Positive Rate',
            showlegend=True
        )
        st.plotly_chart(fig_roc)

if st.button("취소"):
    if 'analysis_results' in st.session_state:
        del st.session_state['analysis_results']
    if 'load_mode' in st.session_state:
        del st.session_state['load_mode']
    if 'model_choice' in st.session_state:
        del st.session_state['model_choice']
    
    for file in glob.glob(os.path.join(UPLOAD_DIR_TEST, "*")):
        os.remove(file)
    if os.path.exists(os.path.join(CKPT_DIR_ZERO, "best.pth")):
        os.remove(os.path.join(CKPT_DIR_ZERO, "best.pth"))
    
    st.rerun()
    st.success("앱 상태가 초기화되었습니다.")

st.markdown("---")
st.subheader("분석 결과 PDF 저장")
if st.button("분석결과 PDF파일로 저장하기"):
    if 'analysis_results' not in st.session_state:
        st.warning("분석 결과를 먼저 생성해야 합니다.")
    else:
        st.info("PDF 보고서를 생성합니다.")
        
                # 추가된 코드: PDF 파일이 저장되는 서버 경로를 보여줌
        st.info(f"PDF 파일은 서버의 다음 경로에 임시 저장됩니다: {os.getcwd()}")
        st.warning("이 경로는 Streamlit Cloud의 임시 공간이므로, 다운로드 버튼을 눌러야 사용자 PC에 저장됩니다.")
        
        filename = st.text_input("저장할 파일 이름을 입력하세요 (예: report.pdf)", "report.pdf")
        
        if st.button("PDF 저장 확인"):
            if not filename.endswith(".pdf"):
                st.error("파일 이름은 '.pdf'로 끝나야 합니다.")
            else:
                cm_fig, ax_cm = plt.subplots()
                cm = confusion_matrix(
                    [1 if r['true_mm'] > 0.5 else 0 for r in st.session_state['analysis_results']],
                    [1 if r['predicted_psi_rad'] > math.pi else 0 for r in st.session_state['analysis_results']]
                )
                sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax_cm)
                cm_fig.savefig("cm_temp.png")

                # Plotly ROC 곡선을 이미지로 저장
                y_true_binary = [1 if r['true_mm'] > 0.5 else 0 for r in st.session_state['analysis_results']]
                y_scores = [r['predicted_psi_rad'] / TWO_PI for r in st.session_state['analysis_results']]
                fpr, tpr, _ = roc_curve(y_true_binary, y_scores)
                roc_auc = auc(fpr, tpr)
                fig_roc = go.Figure()
                fig_roc.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f'ROC curve (AUC = {roc_auc:.2f})'))
                fig_roc.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', name='Random Guess', line=dict(dash='dash')))
                fig_roc.update_layout(title='ROC Curve')
                fig_roc.write_image("roc_temp.png")

                create_pdf_report(filename, st.session_state['analysis_results'], cm_fig, "roc_temp.png")
                
                with open(filename, "rb") as f:
                    st.download_button(
                        label=f"'{filename}' 다운로드",
                        data=f,
                        file_name=filename,
                        mime="application/pdf"
                    )
                st.success(f"PDF 보고서 '{filename}'가 생성되었습니다.")