# -*- coding: utf-8 -*-
"""
아웃포커스(Defocus) 분류 스크립트 - 폴더 자동 생성/저장 포함
작성자: you & gpt
필요 패키지: opencv-python, numpy, pandas
(선택) HEIC 지원: pillow, pillow-heif
    pip install opencv-python numpy pandas
    # HEIC가 있다면:
    pip install pillow pillow-heif
"""

import os, glob, math
import numpy as np
import pandas as pd
import cv2

# ====== 설정 ======
# 사진이 들어있는 폴더(절대경로 권장)
INPUT_DIR = r"C:/Users/SSAFY/Desktop/Athens/photos"   # ← 여기만 바꾸면 됨
RECURSIVE = False          # 하위 폴더까지 처리하려면 True
THRESHOLD = 0.55           # 점수↑ = defocus↑ (임계값 조정: 0.05 간격 추천)
LONG_SIDE = 1024           # 스코어 안정화를 위한 리사이즈 기준(긴변)
USE_HEIC_READER = False    # HEIC(아이폰) 처리하려면 True 로 바꾸고 pillow, pillow-heif 설치

# ====== (선택) HEIC 리더 준비 ======
if USE_HEIC_READER:
    try:
        from PIL import Image
        import pillow_heif
    except Exception:
        print("[WARN] HEIC 지원을 켰지만 라이브러리 로드 실패. 'pip install pillow pillow-heif' 후 다시 시도하세요.")
        USE_HEIC_READER = False


# ====== 유틸: 이미지 로더(한글/공백 경로 안전 + HEIC 옵션) ======
def read_image_any(path):
    if USE_HEIC_READER and path.lower().endswith((".heic", ".heif")):
        heif_file = pillow_heif.read_heif(path)
        img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw").convert("RGB")
        arr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return arr
    # 한글/공백 경로 안전 로딩
    data = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    return data


# ====== 특징량 ======
def variance_of_laplacian(gray):
    return cv2.Laplacian(gray, cv2.CV_64F).var()

def tenengrad(gray):
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy)
    return float(np.mean(mag))

def edge_spread_width(gray, sample_edges=200):
    # 엣지 폭(10~90%)의 중앙값 → 아웃포커스일수록 커짐
    edges = cv2.Canny(gray, 80, 160)
    ys, xs = np.where(edges > 0)
    if len(xs) == 0:
        return 0.0
    idx = np.random.choice(len(xs), size=min(sample_edges, len(xs)), replace=False)
    widths = []
    for i in idx:
        y, x = int(ys[i]), int(xs[i])
        r = 9
        y0, y1 = max(0, y-r), min(gray.shape[0], y+r+1)
        x0, x1 = max(0, x-r), min(gray.shape[1], x+r+1)
        patch = gray[y0:y1, x0:x1]
        if patch.size < 5:
            continue
        hline = np.mean(patch, axis=0)
        vline = np.mean(patch, axis=1)
        for arr in (hline, vline):
            a = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
            p10 = np.argmax(a >= 0.1)
            p90 = np.argmax(a >= 0.9)
            if p90 > p10:
                widths.append(p90 - p10)
    return float(np.median(widths) if widths else 0.0)

def radial_spectrum_slope(gray, cutoff=0.6):
    # 방사 스펙트럼의 고주파 구간 기울기(로그 스케일)
    img = gray.astype(np.float32)
    wy = np.hanning(img.shape[0])[:, None]
    wx = np.hanning(img.shape[1])[None, :]
    win = wy * wx
    f = np.fft.fftshift(np.fft.fft2(img * win))
    mag = np.abs(f)
    h, w = gray.shape
    cy, cx = h//2, w//2
    Y, X = np.ogrid[:h, :w]
    R = np.sqrt((X-cx)**2 + (Y-cy)**2)
    r = R.astype(np.int32)
    rmax = int(min(h, w) * 0.5 * cutoff)
    bins = []
    for rad in range(1, rmax):
        mask = (r == rad)
        if np.any(mask):
            bins.append(np.mean(mag[mask]))
    bins = np.array(bins, dtype=np.float32)
    if len(bins) < 8:
        return 0.0
    x = np.arange(len(bins), dtype=np.float32)
    y = np.log(bins + 1e-8)
    start = len(x) // 2
    Xmat = np.vstack([x[start:], np.ones_like(x[start:])]).T
    slope, _ = np.linalg.lstsq(Xmat, y[start:], rcond=None)[0]
    return float(slope)  # 더 음수일수록 defocus ↑

def anisotropy_index(gray):
    # 방향성 지표(등방성: 낮음 / 모션블러: 높음)
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx*gx + gy*gy) + 1e-8
    ang = (np.arctan2(gy, gx) + np.pi)  # 0~2π
    nbins = 18
    hist, _ = np.histogram(ang, bins=nbins, range=(0, 2*np.pi), weights=mag)
    hist = hist / (hist.sum() + 1e-8)
    return float(np.std(hist))  # 작을수록 등방성(=defocus 쪽)

def defocus_score_from_image(gray, long_side=LONG_SIDE):
    # 해상도 정규화
    H, W = gray.shape
    if max(H, W) > long_side:
        s = long_side / max(H, W)
        gray = cv2.resize(gray, (int(W*s), int(H*s)), interpolation=cv2.INTER_AREA)

    vol   = variance_of_laplacian(gray)      # ↓ defocus
    esw   = edge_spread_width(gray)          # ↑ defocus
    slope = radial_spectrum_slope(gray)      # 더 음수 → ↑ defocus
    aniso = anisotropy_index(gray)           # ↓ defocus (등방성)

    # 경험적 박스 정규화(데이터 보면 조정 가능)
    def norm_box(x, lo, hi, invert=False):
        x = float(x)
        val = (x - lo) / (hi - lo + 1e-8)
        val = min(max(val, 0.0), 1.0)
        return 1.0 - val if invert else val

    vol_n   = norm_box(vol,   lo=50,   hi=600,  invert=True)
    esw_n   = norm_box(esw,   lo=1.0,  hi=6.0,  invert=False)
    slope_n = norm_box(slope, lo=-6.0, hi=-0.5, invert=True)
    aniso_n = norm_box(aniso, lo=0.0,  hi=0.12, invert=True)

    # 가중합(초기값)
    score = 0.4*esw_n + 0.25*vol_n + 0.25*slope_n + 0.10*aniso_n
    feats = {
        "VoL": vol, "ESW": esw, "Slope": slope, "Aniso": aniso,
        "vol_n": vol_n, "esw_n": esw_n, "slope_n": slope_n, "aniso_n": aniso_n
    }
    return float(score), feats


# ====== 분류 실행 ======
def classify_folder_defocus(
    input_dir,
    threshold=THRESHOLD,
    recursive=RECURSIVE,
    out_csv=None,
    out_soft=None,
    out_sharp=None
):
    if not os.path.isdir(input_dir):
        raise FileNotFoundError(f"[ERROR] 입력 폴더가 없습니다: {input_dir}")

    # 출력 경로: 사진 폴더 내부에 생성
    base = input_dir
    if out_csv is None:
        out_csv = os.path.join(base, "defocus_results.csv")
    if out_soft is None:
        out_soft = os.path.join(base, "defocus")
    if out_sharp is None:
        out_sharp = os.path.join(base, "in_focus")

    os.makedirs(out_soft, exist_ok=True)
    os.makedirs(out_sharp, exist_ok=True)

    # 대상 파일 수집 (대/소문자 + 선택적 재귀)
    patterns = ["*.jpg","*.jpeg","*.png","*.bmp","*.tif","*.tiff","*.webp",
                "*.JPG","*.JPEG","*.PNG","*.BMP","*.TIF","*.TIFF","*.WEBP"]
    if USE_HEIC_READER:
        patterns += ["*.heic", "*.heif", "*.HEIC", "*.HEIF"]

    paths = []
    if recursive:
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(input_dir, "**", pat), recursive=True))
    else:
        for pat in patterns:
            paths.extend(glob.glob(os.path.join(input_dir, pat), recursive=False))

    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"[ERROR] 이미지가 없습니다: {input_dir} (확장자/재귀 옵션 확인)")

    rows = []
    skipped = 0

    print(f"[INFO] 입력: {input_dir}")
    print(f"[INFO] 총 {len(paths)}장 처리 시작... (threshold={threshold})")

    for idx, p in enumerate(paths, 1):
        img = read_image_any(p)
        if img is None:
            skipped += 1
            print(f"[WARN] 읽기 실패 → 스킵: {p}")
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        score, feats = defocus_score_from_image(gray)

        label = "defocus" if score >= threshold else "in_focus"
        dst_dir = out_soft if label == "defocus" else out_sharp
        dst = os.path.join(dst_dir, os.path.basename(p))

        ok, buf = cv2.imencode('.jpg', img)
        if not ok:
            skipped += 1
            print(f"[WARN] 저장 실패 → 스킵: {p}")
            continue
        buf.tofile(dst)

        feats.update({"path": p, "score": round(score, 4), "label": label})
        rows.append(feats)

        if idx % 50 == 0:
            print(f"  ..{idx}/{len(paths)} 처리중")

    if not rows:
        raise RuntimeError("[ERROR] 처리된 이미지가 0장입니다. (전부 읽기 실패/스킵)")

    df = pd.DataFrame(rows).sort_values("score", ascending=False)
    df.to_csv(out_csv, index=False, encoding="utf-8-sig")

    print(f"[완료] 총 {len(paths)}장 중 {len(df)}장 처리, 스킵 {skipped}장")
    print(f"       결과 폴더: '{out_soft}/', '{out_sharp}/'")
    print(f"       결과 CSV : {out_csv}")
    return df


# ====== 메인 ======
if __name__ == "__main__":
    df = classify_folder_defocus(
        input_dir=INPUT_DIR,
        threshold=THRESHOLD,
        recursive=RECURSIVE,
        # None 이면 자동으로 INPUT_DIR 아래에 생성됨
        out_csv=None,
        out_soft=None,
        out_sharp=None
    )
    # 경계컷 몇 장 확인용(원하면 주석 해제)
    # print(df.head(5)[["path","score","label"]])
    # print(df.tail(5)[["path","score","label"]])
