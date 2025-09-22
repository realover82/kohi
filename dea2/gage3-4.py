import os, re, math, datetime, time
import numpy as np
import cv2
from glob import glob
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

# -------------------------
# 설정
# -------------------------
DEBUG_MODE = True
IMG_SIZE = 160
HOUGH_SHORT_SIDE = 320
MAX_WORKERS = 8  # 병렬 처리 수

# 경로 설정
DATA_DIR = "./dea2/gage3/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

# 디버깅 결과 저장 경로
DEBUG_DIR = "./dea2/gage3/debug_results"
DEBUG_TRAIN = os.path.join(DEBUG_DIR, "train")
DEBUG_TEST = os.path.join(DEBUG_DIR, "test")
os.makedirs(DEBUG_TRAIN, exist_ok=True)
os.makedirs(DEBUG_TEST, exist_ok=True)

# 전처리된 이미지 저장 경로 (학습용)
PREPROCESSED_DIR = "./dea2/gage3/preprocessed"
PREPROCESSED_TRAIN = os.path.join(PREPROCESSED_DIR, "train")
PREPROCESSED_TEST = os.path.join(PREPROCESSED_DIR, "test")
os.makedirs(PREPROCESSED_TRAIN, exist_ok=True)
os.makedirs(PREPROCESSED_TEST, exist_ok=True)

# Hough 파라미터
HOUGH_PRESETS = {
    "debug": {"circle_param2": 20, "minLineLen": 0.20, "maxLineGap": 0.12},
}
USE_PRESET = "debug"
ROI_SCALE = 0.35

# -------------------------
# TPU 초기화 (OpenCV 가속)
# -------------------------
print("🚀 TPU 가속 전처리 시작...")
print("📊 CPU 코어 수:", mp.cpu_count())

# -------------------------
# 유틸리티 함수
# -------------------------
def safe_imread(path):
    """이미지 안전하게 읽기"""
    try:
        with open(path, "rb") as f:
            arr = np.frombuffer(f.read(), np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    except Exception as e:
        print(f"⚠️ 이미지 읽기 실패 {path}: {e}")
        return None

def save_debug_image(img, filename, stage, subfolder=""):
    """단계별 디버깅 이미지 저장"""
    if subfolder:
        save_dir = os.path.join(DEBUG_DIR, subfolder, stage)
    else:
        save_dir = os.path.join(DEBUG_DIR, stage)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)

    try:
        if isinstance(img, np.ndarray):
            if len(img.shape) == 3 and img.shape[2] == 3:
                img_save = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            else:
                img_save = img
            cv2.imwrite(save_path, img_save)
        return save_path
    except Exception as e:
        print(f"⚠️ 이미지 저장 실패 {save_path}: {e}")
        return None

# -------------------------
# 전처리 함수들 (TPU 가속)
# -------------------------
def _resize_for_hough(bgr, short_side=HOUGH_SHORT_SIDE):
    """Hough용 이미지 리사이즈"""
    h, w = bgr.shape[:2]
    if min(h, w) <= short_side:
        return bgr, 1.0
    scale = short_side / float(min(h, w))
    nh, nw = int(round(h*scale)), int(round(w*scale))
    small = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    return small, scale

# ▼▼▼ [수정] 이 함수 전체를 교체하세요 ▼▼▼
def detect_gauge_circle_debug(bgr, filename, circle_param2=60, subfolder=""):
    """
    다이얼 게이지 원 검출(디버그용).
    - 강건한 전처리(CLAHE+샤픈+Canny)
    - 중앙부 원형 ROI 마스킹
    - Hough 후보를 '에지 일치율 + 방사 그라디언트 + 중심 근접도'로 스코어링
    - 최종 1개 원만 반환 (cx, cy, r) - 원본 크기 스케일로
    """
    import cv2, numpy as np

    # 0) 리사이즈(기존 도구 사용)
    small, s = _resize_for_hough(bgr)
    save_debug_image(small, filename, "1_hough_resized", subfolder=subfolder)

    # 1) 그레이 + 콘트라스트 향상 + 살짝 샤픈
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)   # (오타 수정: BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    gray = clahe.apply(gray)

    # Unsharp mask
    blur = cv2.GaussianBlur(gray, (0,0), 1.2)
    sharp = cv2.addWeighted(gray, 1.6, blur, -0.6, 0)
    save_debug_image(sharp, filename, "2_gray_sharpened", subfolder=subfolder)

    h, w = sharp.shape[:2]
    img_center = np.array([w/2.0, h/2.0], dtype=np.float32)

    # 2) 중앙부 원형 ROI(테두리 잡음 억제)
    #     게이지는 대체로 중앙에 있음: 반지름은 화면의 0.48배 정도로 설정
    roi_mask = np.zeros_like(sharp, dtype=np.uint8)
    roi_r = int(min(h, w) * 0.48)
    cv2.circle(roi_mask, (int(img_center[0]), int(img_center[1])), roi_r, 255, -1)

    # 3) Canny 에지 (히스테리시스 임계 자동 설정)
    med = np.median(sharp[roi_mask==255])
    low = int(max(0, 0.66*med))
    high = int(min(255, 1.33*med))
    edges = cv2.Canny(sharp, low, high)
    edges = cv2.bitwise_and(edges, edges, mask=roi_mask)
    save_debug_image(edges, filename, "3_edges_roi", subfolder=subfolder)

    # 4) HoughCircles 파라미터 (눈금 링/베젤 범위로 고정)
    min_r = int(min(h, w) * 0.30)
    max_r = int(min(h, w) * 0.46)

    circles = cv2.HoughCircles(
        sharp, cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=int(min(h, w) * 0.22),
        param1=max(80, high),
        param2=int(circle_param2),
        minRadius=min_r,
        maxRadius=max_r
    )

    debug_img = cv2.cvtColor(sharp, cv2.COLOR_GRAY2BGR)
    best = None
    best_score = -1.0

    def circle_score(cx, cy, r):
        """후보 원 품질 점수: 에지일치 + 방사그라디언트 + 중심근접(가중치 낮음)"""
        cx, cy, r = int(cx), int(cy), int(r)
        if not (0 <= cx < w and 0 <= cy < h): return -1.0
        if r < min_r or r > max_r: return -1.0

        # ① 둘레 샘플링
        pts = cv2.ellipse2Poly((cx, cy), (r, r), 0, 0, 360, 2)  # 약 180점
        if len(pts) < 60: return -1.0

        # ② 에지 일치율
        edge_hits = 0
        for (px, py) in pts:
            if 0 <= px < w and 0 <= py < h and edges[py, px] != 0:
                edge_hits += 1
        edge_ratio = edge_hits / float(len(pts))  # 0~1

        # ③ 방사 그라디언트(원 경계 안/밖 평균 밝기 차이)
        t = max(1, r // 20)  # 얇은 띠
        inner_r = max(0, r - t)
        outer_r = min(max_r + t, r + t)

        # 단순 원형 링 평균: 안쪽/바깥쪽
        yy, xx = np.ogrid[:h, :w]
        dist2 = (xx - cx)**2 + (yy - cy)**2
        inner_mask = (dist2 <= inner_r*inner_r).astype(np.uint8)
        ring_out_mask = ((dist2 >= outer_r*outer_r - 0) & (dist2 <= (outer_r+2*t)**2)).astype(np.uint8)

        if inner_mask.sum() == 0 or ring_out_mask.sum() == 0:
            grad_score = 0.0
        else:
            m_in = float(sharp[inner_mask==1].mean())
            m_out = float(sharp[ring_out_mask==1].mean())
            grad_score = abs(m_out - m_in) / 255.0  # 0~1 정규화

        # ④ 중심 근접도(약한 가중). 중앙에서 멀수록 패널티
        d_center = np.linalg.norm(np.array([cx, cy]) - img_center) / (min(h, w) / 2.0)
        center_score = max(0.0, 1.0 - d_center)  # 0(멀다)~1(가깝다)

        # 가중합 (에지 0.55, 그라디언트 0.35, 중심 0.10)
        return 0.55*edge_ratio + 0.35*grad_score + 0.10*center_score

    if circles is not None and len(circles[0]) > 0:
        # 모든 후보(초록) 표시 + 점수화
        for (x, y, r) in circles[0]:
            sc = circle_score(x, y, r)
            color = (0, 255, 0)
            cv2.circle(debug_img, (int(x), int(y)), int(r), color, 1)
            if sc > best_score:
                best_score = sc
                best = (int(round(x)), int(round(y)), int(round(r)))

    # 최종 선택(파랑) + 중심표시(빨강)
    if best is not None:
        cx, cy, r = best
        cv2.circle(debug_img, (cx, cy), r, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(debug_img, (cx, cy), 3, (0, 0, 255), -1)
    save_debug_image(debug_img, filename, "4_circle_detection_selected", subfolder=subfolder)

    if best is None:
        return None

    # 스케일 복원
    inv = 1.0 / s
    cx, cy, r = best
    return int(round(cx*inv)), int(round(cy*inv)), int(round(r*inv))


#####
def detect_needle_line_debug(
    bgr, cx, cy, r, filename,
    # 과거 호환 (무시됨)
    minLineLength=0.42, maxLineGap=0.06,
    # 레이 스캔 파라미터
    inner_ratio=0.12, outer_ratio=0.96,
    deg_step=0.5, ray_clip_pct=0.25,
    subfolder=""
):
    """
    방사선 스캔 기반 바늘 검출 (강건한 마스크 + 대비 기반 점수 + 폴백)
    디버그: 4_needle_roi, 5_needle_mask, 6_needle_edges, 7_line_detection, 8_final_needle
    """
    import cv2, numpy as np, math

    # 1) ROI
    x1 = max(0, cx - int(r*1.10)); y1 = max(0, cy - int(r*1.10))
    x2 = min(bgr.shape[1], cx + int(r*1.10)); y2 = min(bgr.shape[0], cy + int(r*1.10))
    roi = bgr[y1:y2, x1:x2]
    if roi.size == 0: return None
    save_debug_image(roi, filename, "4_needle_roi", subfolder=subfolder)

    # 2) 축소 + 전처리
    small, s = _resize_for_hough(roi)
    gray  = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8,8))
    g     = clahe.apply(gray)
    g     = cv2.addWeighted(g, 1.5, cv2.GaussianBlur(g, (0,0), 1.0), -0.5, 0)
    save_debug_image(g, filename, "6_needle_edges", subfolder=subfolder)

    h, w = g.shape[:2]
    scx = int(round((cx - x1) * s)); scy = int(round((cy - y1) * s))
    if not (0 <= scx < w and 0 <= scy < h):
        scx, scy = w//2, h//2

    # --- 반경 산정: sr과 ROI기반 둘 다 사용하여 강건화 ---
    sr = int(round(r * s))                      # 원 검출 반경(스케일된)
    R_half = 0.5 * min(h, w)                    # ROI 절반
    # sr이 비정상적으로 작/크면 ROI 기반으로 보정
    if not (0.55*R_half <= sr <= 1.25*R_half):
        sr = int(0.90*R_half)
    # 최종 바깥 반경
    R_out = int(min(0.98*sr, 0.98*R_half))
    R_out = max(R_out, int(0.65*R_half))         # 너무 작지 않게 하한
    R_in_min, R_in_max = int(0.05*R_out), int(0.18*R_out)

    # 3) 허브 반경 자동 추정 (+클램프)
    def estimate_hub_radius():
        rs = np.arange(2, max(6, int(0.30*R_out)), 1, dtype=np.int32)
        if len(rs) == 0: return 0
        theta = np.linspace(0, 2*np.pi, 72, endpoint=False)
        ct, st = np.cos(theta), np.sin(theta)
        means = []
        for rr in rs:
            xs = np.clip((scx + rr*ct).round().astype(np.int32), 0, w-1)
            ys = np.clip((scy + rr*st).round().astype(np.int32), 0, h-1)
            means.append(g[ys, xs].mean())
        means = np.array(means, np.float32)
        diffs = np.diff(means, prepend=means[0])
        j = np.argmax(diffs > 10.0)
        if j == 0 and not (diffs[0] > 10.0):
            j = int(np.argmax(diffs))
        r_edge = int(rs[min(j, len(rs)-1)])
        return int(np.clip(0.85*r_edge, R_in_min, R_in_max))

    r_in = estimate_hub_radius()

    # 4) 마스크 생성 (+폴백)
    mask = np.zeros_like(g, np.uint8)
    cv2.circle(mask, (scx, scy), R_out, 255, -1)
    if 0 < r_in < R_out:
        cv2.circle(mask, (scx, scy), r_in, 0, -1)
    # 커버리지 체크: 너무 작거나 이상하면 디스크로 폴백
    if mask.mean() < 15.0:  # 평균 픽셀값(0~255)
        mask[:] = 0
        cv2.circle(mask, (scx, scy), R_out, 255, -1)
    save_debug_image(mask, filename, "5_needle_mask", subfolder=subfolder)

    # 5) 방사선 스캔 (어둡기+대비 점수)
    thetas = np.deg2rad(np.arange(0, 360, max(0.1, float(deg_step))))
    R0 = max(1, (r_in if r_in > 0 else int(0.12*R_out)) + 1)
    R1 = max(R0+16, R_out-1)                          # 최소 길이 보장
    base = np.arange(R0, R1)
    scores = np.full(len(thetas), 255.0, np.float32)
    offsets = [0.0, +1.0, -1.0]
    for i, th in enumerate(thetas):
        nx, ny = -np.sin(th), np.cos(th)               # 평행 오프셋(대비 안정화)
        xs0 = scx + np.cos(th)*base
        ys0 = scy + np.sin(th)*base
        vals_dark, vals_bg = [], []
        for off in offsets:
            xs = np.clip(np.round(xs0 + nx*off).astype(np.int32), 0, w-1)
            ys = np.clip(np.round(ys0 + ny*off).astype(np.int32), 0, h-1)
            m  = mask[ys, xs] > 0
            if not np.any(m):
                continue
            line_vals = g[ys[m], xs[m]].astype(np.float32)
            k = int(len(line_vals)*float(ray_clip_pct))
            if k > 0 and len(line_vals) > 2*k:
                line_vals = line_vals[k:-k]

            # 양옆 배경 밝기(법선 ±2px 평균)
            xs_l = np.clip(xs + (nx*2).round().astype(np.int32), 0, w-1)
            ys_l = np.clip(ys + (ny*2).round().astype(np.int32), 0, h-1)
            xs_r = np.clip(xs - (nx*2).round().astype(np.int32), 0, w-1)
            ys_r = np.clip(ys - (ny*2).round().astype(np.int32), 0, h-1)
            bg = 0.5*(g[ys_l[m], xs_l[m]].astype(np.float32) +
                      g[ys_r[m], xs_r[m]].astype(np.float32))

            if len(line_vals) > 0:
                vals_dark.append(line_vals)
                vals_bg.append(bg)

        if vals_dark:
            vd = np.concatenate(vals_dark)
            vb = np.concatenate(vals_bg) if vals_bg else vd
            # 점수: 낮을수록 바늘일 가능성 ↑ (어둡기 - 대비 가중)
            q25 = np.quantile(vd, 0.25)
            contrast = float(np.clip(vb.mean() - vd.mean(), 0, 255))
            scores[i] = q25 - 0.5*contrast  # 0.5 가중치는 경험치

    if not np.isfinite(scores).any():
        return None

    best_theta = thetas[int(np.nanargmin(scores))]

    # 6) 디버그(최적 레이)
    line_dbg = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
    px2 = int(round(scx + math.cos(best_theta)*R1))
    py2 = int(round(scy + math.sin(best_theta)*R1))
    cv2.line(line_dbg, (scx, scy), (px2, py2), (0,255,0), 2)
    save_debug_image(line_dbg, filename, "7_line_detection", subfolder=subfolder)

    # 7) 원본 좌표 복원 + 게이지 경계까지 연장
    ang = math.atan2((py2/s + y1) - cy, (px2/s + x1) - cx)
    X1, Y1 = cx, cy
    X2 = int(round(cx + math.cos(ang) * (r*0.98)))
    Y2 = int(round(cy + math.sin(ang) * (r*0.98)))

    final_img = bgr.copy()
    cv2.circle(final_img, (cx, cy), r, (0,255,0), 2)
    cv2.line(final_img, (X1, Y1), (X2, Y2), (0,0,255), 3)
    save_debug_image(final_img, filename, "8_final_needle", subfolder=subfolder)

    return (X1, Y1, X2, Y2)

def _safe_crop(img, x1, y1, x2, y2):
    """안전한 이미지 크롭"""
    h, w = img.shape[:2]
    x1 = max(0, min(w-1, int(x1))); y1 = max(0, min(h-1, int(y1)))
    x2 = max(0, min(w, int(x2))); y2 = max(0, min(h, int(y2)))
    if x2 <= x1 or y2 <= y1: return None
    return img[y1:y2, x1:x2]

def _center_square(img, cx=None, cy=None, side=None, pad=0.9):
    """중앙 정사각형 크롭"""
    h, w = img.shape[:2]
    if side is None: side = int(min(h, w) * pad)
    side = max(16, min(side, min(h, w)-2))
    if cx is None: cx = w // 2
    if cy is None: cy = h // 2
    crop = _safe_crop(img, cx-side//2, cy-side//2, cx+side//2, cy+side//2)
    if crop is None:
        crop = _safe_crop(img, (w-side)//2, (h-side)//2, (w+side)//2, (h+side)//2)
    return crop

# -------------------------
# 단일 이미지 전처리 함수
# -------------------------
def process_single_image(args):
    """단일 이미지 전처리 (병렬 처리용)"""
    src_path, dst_path, debug_filename, is_train = args

    bgr = safe_imread(src_path)
    if bgr is None:
        print(f"⚠️ 이미지 로드 실패: {src_path}")
        return False

    subfolder = "train" if is_train else "test"

    # 원본 이미지 저장
    save_debug_image(bgr, debug_filename, "0_original", subfolder=subfolder)

    params = HOUGH_PRESETS[USE_PRESET]

    try:
        # 1. 원 검출
        circle = detect_gauge_circle_debug(bgr, debug_filename, params["circle_param2"], subfolder=subfolder)

        if not circle:
            print(f"⚠️ 원 검출 실패: {debug_filename}")
            crop = _center_square(bgr, side=int(min(bgr.shape[:2])*0.8))
            save_debug_image(crop, debug_filename, "9_fallback_crop", subfolder=subfolder)
        else:
            cx, cy, r = circle
            print(f"✅ 원 검출 성공: {debug_filename}")

            # 2. 바늘 검출
            line = detect_needle_line_debug(bgr, cx, cy, r, debug_filename,
                                            params["minLineLen"], params["maxLineGap"], subfolder=subfolder)

            if line is not None:
                # 바늘 각도 계산 및 회전
                x1,y1,x2,y2 = map(int, line)
                d1 = np.hypot(x1-cx, y1-cy); d2 = np.hypot(x2-cx, y2-cy)
                tip = (x1,y1) if d1 > d2 else (x2,y2)
                dx, dy = tip[0]-cx, tip[1]-cy
                angle = math.degrees(math.atan2(-dy, dx))

                # 회전
                M = cv2.getRotationMatrix2D((float(cx), float(cy)), -angle, 1.0)
                rot = cv2.warpAffine(bgr, M, (bgr.shape[1], bgr.shape[0]))
                save_debug_image(rot, debug_filename, "9_rotated", subfolder=subfolder)

                # ROI 크롭
                tx, ty = int(cx+dx), int(cy+dy)
                side = int(max(20, r*ROI_SCALE))
                crop = _safe_crop(rot, tx-side//2, ty-side//2, tx+side//2, ty+side//2)

                if crop is None:
                    crop = _center_square(rot, cx, cy, side=int(min(rot.shape[:2])*0.8))
                    save_debug_image(crop, debug_filename, "10_center_crop", subfolder=subfolder)
                else:
                    save_debug_image(crop, debug_filename, "10_needle_tip_crop", subfolder=subfolder)
            else:
                crop = _center_square(bgr, cx, cy, side=int(r*0.8))
                save_debug_image(crop, debug_filename, "10_center_circle_crop", subfolder=subfolder)

        if crop is None:
            crop = _center_square(bgr, side=int(min(bgr.shape[:2])*0.8))
            save_debug_image(crop, debug_filename, "11_final_fallback_crop", subfolder=subfolder)

        # 3. 최종 전처리
        roi = cv2.resize(crop, (IMG_SIZE, IMG_SIZE))
        rgb_result = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)

        # 최종 결과 저장
        save_debug_image(rgb_result, debug_filename, "12_final_result", subfolder=subfolder)

        # 학습용 이미지 저장
        cv2.imwrite(dst_path, cv2.cvtColor(rgb_result, cv2.COLOR_RGB2BGR))

        return True

    except Exception as e:
        print(f"❌ 전처리 실패 {debug_filename}: {e}")
        try:
            fallback = cv2.resize(bgr, (IMG_SIZE, IMG_SIZE))
            save_debug_image(fallback, debug_filename, "13_error_fallback", subfolder=subfolder)
            cv2.imwrite(dst_path, fallback)
            return True
        except:
            return False

# -------------------------
# 메인 실행 함수
# -------------------------
def run_preprocessing():
    """메인 전처리 실행 함수"""
    print("🚀 TPU 가속 전처리 시작!")
    start_time = time.time()

    # 처리할 이미지 목록 준비
    tasks = []

    # 학습 데이터
    train_files = [f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in train_files:
        src_path = os.path.join(TRAIN_DIR, fname)
        dst_path = os.path.join(PREPROCESSED_TRAIN, fname)
        tasks.append((src_path, dst_path, fname, True))

    # 테스트 데이터
    test_files = [f for f in os.listdir(TEST_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for fname in test_files:
        src_path = os.path.join(TEST_DIR, fname)
        dst_path = os.path.join(PREPROCESSED_TEST, fname)
        tasks.append((src_path, dst_path, fname, False))

    print(f"📊 총 처리할 이미지: {len(tasks)}개")
    print(f"🔧 병렬 처리 workers: {MAX_WORKERS}")

    # 병렬 처리 실행
    success_count = 0
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        results = list(executor.map(process_single_image, tasks))

    success_count = sum(results)

    total_time = time.time() - start_time
    avg_time_per_image = total_time / len(tasks) if tasks else 0

    print(f"\n🎉 전처리 완료!")
    print(f"⏰ 소요 시간: {total_time:.1f}초")
    print(f"📊 처리 결과: {success_count}/{len(tasks)} 성공")
    print(f"📈 이미지당 평균 시간: {avg_time_per_image:.2f}초")
    print(f"💾 디버깅 결과: {DEBUG_DIR}")
    print(f"💾 학습용 이미지: {PREPROCESSED_DIR}")

    # 폴더 구조 설명
    print("\n📋 디버깅 폴더 구조:")
    stages = [
        "0_original", "1_hough_resized", "2_gray_blurred", "3_circle_detection",
        "4_needle_roi", "5_needle_mask", "6_needle_edges", "7_line_detection",
        "8_final_needle", "9_rotated", "10_center_crop", "10_needle_tip_crop",
        "10_center_circle_crop", "11_final_fallback_crop", "12_final_result", "13_error_fallback"
    ]
    for stage in stages:
        print(f"  {stage}/")

# -------------------------
# 즉시 실행
# -------------------------
if __name__ == "__main__":
    run_preprocessing()
    
# ============================================
# 1) Clean Dataset & Labels 만들기 (수정판)
# ============================================
import os, json, math, csv
from glob import glob
import numpy as np
import cv2
from PIL import Image

DATA_DIR = "./dea2/gage3/"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR  = os.path.join(DATA_DIR, "test")

DATASET_DIR   = os.path.join(DATA_DIR, "dataset_clean")
DATASET_TRAIN = os.path.join(DATASET_DIR, "train")
DATASET_TEST  = os.path.join(DATASET_DIR, "test")
os.makedirs(DATASET_TRAIN, exist_ok=True); os.makedirs(DATASET_TEST, exist_ok=True)

CSV_TRAIN = os.path.join(DATASET_DIR, "labels_train.csv")
CSV_TEST  = os.path.join(DATASET_DIR, "labels_test.csv")

def compute_theta_cw(cx, cy, x2, y2):
    theta = math.atan2(y2 - cy, x2 - cx)
    return (math.pi/2 - theta) % (2*math.pi)   # 12시=0°, 시계+

def safe_center_crop_square(bgr, cx, cy, r, pad_ratio=1.15, out_size=256):
    """
    회전/워프 없이: 중심(cx,cy) 기준 한 변 S=2*r*pad_ratio의 정사각형을
    getRectSubPix로 잘라 out_size로 리사이즈. (가장 안전)
    """
    H, W = bgr.shape[:2]
    S = int(max(32, min(max(2*r*pad_ratio, 32), 1.5*min(H, W))))   # 과/과소 스케일 방지
    # getRectSubPix는 범위를 넘어가면 검정 패딩이 생길 수 있음 → 패딩이 과도하면 실패로 간주
    crop = cv2.getRectSubPix(bgr, (S, S), (float(cx), float(cy)))
    if crop is None or crop.size == 0:
        return None
    clean = cv2.resize(crop, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return cv2.cvtColor(clean, cv2.COLOR_BGR2GRAY)

def quality_is_bad(img_gray):
    """완전히 검정에 가까운 잘못된 샘플 걸러내기"""
    m, s = float(img_gray.mean()), float(img_gray.std())
    # 평균/표준편차 둘 다 너무 낮으면 사실상 '검정'
    return (m < 5.0) or (s < 3.0)

def build_clean_dataset(split_dir, out_dir, csv_path, out_size=256):
    items = sorted([p for p in glob(os.path.join(split_dir, "*"))
                     if p.lower().endswith(('.png','.jpg','.jpeg'))])
    rows = [["filepath","theta_rad","sin","cos"]]
    ok = fail = 0

    for src_path in items:
        fname = os.path.basename(src_path)
        bgr = safe_imread(src_path)
        if bgr is None:
            fail += 1; continue

        circ = detect_gauge_circle_debug(bgr, fname, circle_param2=60)
        if not circ:
            fail += 1; continue
        cx, cy, r = circ

        line = detect_needle_line_debug(bgr, cx, cy, r, fname)
        if line is None:
            fail += 1; continue

        x1,y1,x2,y2 = map(int, line)
        # 바늘 팁(더 먼 점)
        d1 = math.hypot(x1-cx, y1-cy); d2 = math.hypot(x2-cx, y2-cy)
        tipx, tipy = (x1,y1) if d1>d2 else (x2,y2)

        theta = compute_theta_cw(cx, cy, tipx, tipy)
        s, c  = math.sin(theta), math.cos(theta)

        # *** 안전한 정규화: 중심 사각형 크롭 ***
        clean_gray = safe_center_crop_square(bgr, cx, cy, r, pad_ratio=1.15, out_size=out_size)
        if clean_gray is None or quality_is_bad(clean_gray):
            # 폴백: 전체 중앙 크롭
            H,W = bgr.shape[:2]
            side = int(min(H,W)*0.9)
            fallback = bgr[(H-side)//2:(H+side)//2, (W-side)//2:(W+side)//2]
            if fallback is None or fallback.size==0:
                fail += 1; continue
            clean_gray = cv2.resize(cv2.cvtColor(fallback, cv2.COLOR_BGR2GRAY),
                                     (out_size,out_size), interpolation=cv2.INTER_AREA)
            if quality_is_bad(clean_gray):
                fail += 1; continue

        out_path = os.path.join(out_dir, os.path.splitext(fname)[0] + ".png")
        cv2.imwrite(out_path, clean_gray)
        rows.append([out_path, f"{theta:.8f}", f"{s:.8f}", f"{c:.8f}"])
        ok += 1

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)
    print(f"[{os.path.basename(out_dir)}] OK={ok}  FAIL={fail}   → {csv_path}")

# 실행
build_clean_dataset(TRAIN_DIR, DATASET_TRAIN, CSV_TRAIN, out_size=224)
build_clean_dataset(TEST_DIR,  DATASET_TEST,  CSV_TEST,  out_size=224)