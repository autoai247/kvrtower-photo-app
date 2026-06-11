"""실시간 카메라 + 인쇄 버튼 = 영수증 사진 즉시 출력

- 라이브 미리보기에 영수증 폭(80mm) 비율 가이드를 표시
- 인쇄 버튼을 누르면 현재 프레임을 캡처하여 ESC/POS 흑백 디더링으로 출력
- 카메라/프린터/제목/타임스탬프/연사 잠금시간 등 옵션 패널 제공
"""

import json
import logging
import os
import sys
import threading
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import ttk, messagebox

import cv2
from PIL import Image, ImageDraw, ImageFont, ImageTk

# src 폴더에서 직접 실행하든 EXE로 묶이든 동일하게 import 되도록
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from printer import PrinterManager, list_windows_printers, PRINT_DOTS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger(__name__)

APP_TITLE = "카메라 영수증 프린터"
SETTINGS_PATH = Path(os.path.expanduser("~")) / ".camera_printer.json"

# 영수증 출력 폭 (printer.py 와 동일: 80mm = 576 dots)
# 미리보기에서 잘라낼 가로:세로 비율의 가로 기준값. 세로는 사용자 조절.
DEFAULT_ASPECT = 1.0  # 1.0 = 정사각형 고정


def _load_settings() -> dict:
    try:
        return json.loads(SETTINGS_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {}


def _save_settings(data: dict) -> None:
    try:
        SETTINGS_PATH.write_text(
            json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    except Exception as e:
        log.warning("설정 저장 실패: %s", e)


def _find_korean_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    bold_first = [
        r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothicExtraBold.ttf",
        r"C:\Windows\Fonts\NanumGothicBold.ttf",
        r"C:\Windows\Fonts\GmarketSansTTFBold.ttf",
    ]
    regular = [
        r"C:\Windows\Fonts\malgun.ttf",
        r"C:\Windows\Fonts\GmarketSansTTFMedium.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf",
        r"C:\Windows\Fonts\gulim.ttc",
    ]
    candidates = (bold_first + regular) if bold else (regular + bold_first)
    for p in candidates:
        if os.path.exists(p):
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                pass
    return ImageFont.load_default()


def enhance_for_thermal(pil_img: Image.Image) -> Image.Image:
    """영수증 사진용 최소 처리 파이프라인 — 선명도와 대비만, 밝기 조정 X.

    밝기 조정(Shadow Lift, 감마, S-curve)을 누적하면 사진이 평평·밝아짐.
    CLAHE로 대비만 살리고, Unsharp Mask로 윤곽만 또렷하게.
    디더링은 GDI 드라이버가 처리.
    """
    import numpy as np

    arr = np.array(pil_img.convert("RGB"))
    # 인물용 그레이 변환 — 빨강 비중 키워서 피부톤 자연스럽게
    gray = (arr[:, :, 0].astype(np.float32) * 0.42 +
            arr[:, :, 1].astype(np.float32) * 0.45 +
            arr[:, :, 2].astype(np.float32) * 0.13)
    gray = np.clip(gray, 0, 255).astype(np.uint8)

    # 1) CLAHE — 강한 로컬 대비 (입·코 주변 디테일 분리)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    sharp = clahe.apply(gray)

    # 2) 매우 강한 Unsharp Mask — 윤곽 또렷이
    blur = cv2.GaussianBlur(sharp.astype(np.float32), (0, 0), 1.0)
    sharp = cv2.addWeighted(sharp.astype(np.float32), 2.5, blur, -1.5, 0)
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)

    # 3) AI 얼굴 처리 — Mediapipe Face Mesh 우선, 실패 시 Haar fallback
    rgb_for_mesh = np.array(pil_img.convert("RGB"))
    face_parts = _detect_face_parts(rgb_for_mesh)

    if face_parts is not None:
        # ── 부위별 차별화 처리 ──
        # 얼굴 외곽 영역에서 눈/입/눈썹 빼면 = 피부 마스크
        skin_mask = face_parts["oval"].copy() if face_parts.get("oval") is not None \
                    else np.zeros_like(sharp, dtype=np.float32)
        for k in ("lips", "left_eye", "right_eye", "left_brow", "right_brow"):
            m = face_parts.get(k)
            if m is not None:
                skin_mask = np.maximum(skin_mask - m, 0)

        # 피부 영역 — bilateral filter로 부드럽게 (피부톤 매끈, 잡티 줄임)
        skin_smooth = cv2.bilateralFilter(sharp, d=9, sigmaColor=45, sigmaSpace=45)
        sharp_f = sharp.astype(np.float32)
        sharp_f = (sharp_f * (1 - skin_mask * 0.55)
                   + skin_smooth.astype(np.float32) * (skin_mask * 0.55))

        # 눈/입/눈썹 영역 — 추가 강한 sharpening (디테일 살림)
        detail_mask = np.zeros_like(sharp_f, dtype=np.float32)
        for k in ("lips", "left_eye", "right_eye", "left_brow", "right_brow"):
            m = face_parts.get(k)
            if m is not None:
                detail_mask = np.maximum(detail_mask, m)

        blur_detail = cv2.GaussianBlur(sharp_f, (0, 0), 0.5)
        extra_sharp = sharp_f + (sharp_f - blur_detail) * 1.6
        sharp_f = sharp_f * (1 - detail_mask) + extra_sharp * detail_mask
        sharp = np.clip(sharp_f, 0, 255).astype(np.uint8)
        log.info("Face Mesh 적용 — 피부 부드럽게 + 눈·입·눈썹 강화")
    else:
        # Mediapipe 실패 시 Haar Cascade로 폴백 (얼굴 영역만 추가 sharpening)
        face_mask = _detect_face_mask(gray)
        if face_mask is not None:
            face_blur = cv2.GaussianBlur(sharp.astype(np.float32), (0, 0), 0.6)
            face_extra_sharp = sharp.astype(np.float32) + (sharp.astype(np.float32) - face_blur) * 1.0
            sharp = (sharp.astype(np.float32) * (1 - face_mask)
                     + face_extra_sharp * face_mask)
            sharp = np.clip(sharp, 0, 255).astype(np.uint8)
            log.info("Haar Cascade 폴백 — 얼굴 영역 추가 처리")

    # 4) 약한 Dark Floor — 매우 어두운(0~50) 픽셀만 살짝 들어올림
    sharp_f = sharp.astype(np.float32)
    knee = 50.0
    floor = 30.0
    mapped = floor + (sharp_f / knee) * (knee - floor + 10)
    final = np.where(sharp_f < knee, mapped, sharp_f)
    final = np.clip(final, 0, 255).astype(np.uint8)

    return Image.fromarray(final)


# ─────────── 음성 안내 (윈도우 SAPI 비동기) ───────────

_TTS_ENGINE = None
_TTS_LOCK = threading.Lock()


# 음성 — win32com SAPI 직접 호출 (subprocess 없음, 즉시 발화)
_TTS_QUEUE = []
_TTS_QUEUE_LOCK = threading.Lock()
_TTS_WORKER = None


def _tts_loop():
    """워커 — SSML로 어리고 경쾌한 톤(피치 ↑, 속도 ↑) 발화.
    SVSFlagsAsync (1) + SVSFPurgeBeforeSpeak (2) + SVSFIsXML (8)
    """
    SVS_ASYNC = 1
    SVS_PURGE = 2
    SVS_XML = 8
    try:
        import pythoncom
        import win32com.client
        pythoncom.CoInitialize()
        sapi = win32com.client.Dispatch("SAPI.SpVoice")
        # 기본 볼륨/속도 (SSML 미적용 안내용)
        try:
            sapi.Volume = 100
            sapi.Rate = 1   # 살짝 빠르게 (경쾌)
        except Exception:
            pass
        voices = sapi.GetVoices()
        en_voice = None
        ko_voice = None
        for i in range(voices.Count):
            v = voices.Item(i)
            desc = v.GetDescription().lower()
            # 여성 음성 우선 (어리고 밝음): Zira > David
            if en_voice is None and ("zira" in desc):
                en_voice = v
            if ko_voice is None and ("heami" in desc or "korean" in desc
                                       or "한국어" in desc):
                ko_voice = v
        # 폴백
        if en_voice is None:
            for i in range(voices.Count):
                v = voices.Item(i)
                if "en" in v.GetDescription().lower():
                    en_voice = v; break
        log.info("SAPI 로드 — en=%s, ko=%s",
                 en_voice.GetDescription() if en_voice else "default",
                 ko_voice.GetDescription() if ko_voice else "default")
    except Exception as e:
        log.warning("SAPI 초기화 실패: %s", e)
        return

    def to_ssml(text):
        """피치 +30%, 속도 +15% — 어리고 경쾌한 톤"""
        # XML 안전 escape
        safe = (text.replace("&", "&amp;")
                    .replace("<", "&lt;")
                    .replace(">", "&gt;")
                    .replace('"', "&quot;"))
        return ('<speak version="1.0" xml:lang="en-US">'
                '<prosody pitch="+30%" rate="+15%" volume="+10%">'
                f'{safe}</prosody></speak>')

    while True:
        item = None
        with _TTS_QUEUE_LOCK:
            if _TTS_QUEUE:
                item = _TTS_QUEUE[-1]
                _TTS_QUEUE.clear()
        if item is None:
            time.sleep(0.03)
            continue
        en, ko = item
        try:
            if en:
                if en_voice is not None:
                    sapi.Voice = en_voice
                sapi.Speak(to_ssml(en), SVS_ASYNC | SVS_PURGE | SVS_XML)
            if ko:
                if ko_voice is not None:
                    sapi.Voice = ko_voice
                sapi.Speak(to_ssml(ko), SVS_ASYNC | SVS_XML)
        except Exception as e:
            log.debug("SAPI 발화 실패: %s", e)


def _ensure_tts_proc():
    """워커 스레드 시작 (호환성을 위해 함수명 유지)"""
    global _TTS_WORKER
    if _TTS_WORKER is None or not _TTS_WORKER.is_alive():
        _TTS_WORKER = threading.Thread(target=_tts_loop, daemon=True)
        _TTS_WORKER.start()


def speak(en_text: str = "", ko_text: str = "", urgent: bool = False):
    """음성 발화 — 매 호출이 직전 발화를 즉시 중단하고 새 발화 시작.
    화면에 보이는 텍스트와 음성이 절대 어긋나지 않게 보장.
    """
    if not en_text and not ko_text:
        return
    with _TTS_QUEUE_LOCK:
        _TTS_QUEUE.clear()  # 직전 대기 항목 모두 폐기 (가장 최신만)
        _TTS_QUEUE.append((en_text or "", ko_text or ""))
    _ensure_tts_proc()


# ─────────── Mediapipe Face Detection (거리·각도에 강함) ───────────

_FACE_DETECTOR = None


def _get_face_detector():
    global _FACE_DETECTOR
    if _FACE_DETECTOR is None:
        try:
            import mediapipe as mp
            # model_selection=1 = 5m 이내 얼굴 검출 (TV 카메라 매장 거리 대응)
            # 임계 0.3 — 약한 검출(멀리/측면) 까지 받아 호객 단계 트리거.
            # 인쇄 단계는 코드에서 score >= 0.6 + 위치/크기 필터로 별도 컷.
            _FACE_DETECTOR = mp.solutions.face_detection.FaceDetection(
                model_selection=1,
                min_detection_confidence=0.3,
            )
            log.info("Face Detection 로드 완료")
        except Exception as e:
            log.warning("Face Detection 로드 실패: %s", e)
            _FACE_DETECTOR = False
    return _FACE_DETECTOR if _FACE_DETECTOR is not False else None


def _detect_face_box(rgb_image):
    """얼굴 bounding box 빠르게 검출 (Face Mesh보다 빠르고 거리 강함).
    returns: (cx, cy, w, h, img_w, img_h) 또는 None
    """
    det = _get_face_detector()
    if det is None:
        return None
    try:
        results = det.process(rgb_image)
        if not results.detections:
            return None
        # 가장 큰 (가장 가까운) 얼굴 선택
        h, w = rgb_image.shape[:2]
        best = None
        best_area = 0
        for d in results.detections:
            bb = d.location_data.relative_bounding_box
            bw, bh = bb.width * w, bb.height * h
            area = bw * bh
            if area > best_area:
                best_area = area
                best = (bb.xmin * w + bw / 2, bb.ymin * h + bh / 2,
                        bw, bh, w, h)
        return best
    except Exception as e:
        log.debug("Face Detection 처리 실패: %s", e)
        return None


def _detect_faces_all(rgb_image):
    """모든 얼굴 (cx, cy, w, h, img_w, img_h, score) 리스트 — 면적 큰 순.
    자동 인쇄 락온 + 호객/인쇄 단계 분기용. score 는 0~1.
    """
    det = _get_face_detector()
    if det is None:
        return []
    try:
        results = det.process(rgb_image)
        if not results.detections:
            return []
        h, w = rgb_image.shape[:2]
        faces = []
        for d in results.detections:
            bb = d.location_data.relative_bounding_box
            bw, bh = bb.width * w, bb.height * h
            score = float(d.score[0]) if d.score else 0.5
            faces.append((bb.xmin * w + bw / 2, bb.ymin * h + bh / 2,
                          bw, bh, w, h, score))
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces
    except Exception as e:
        log.debug("Face Detection (multi) 실패: %s", e)
        return []


# ─────────── Mediapipe Face Mesh (468 랜드마크 부위별 처리) ───────────

_FACE_MESH = None  # 싱글톤 (한 번만 로드)


def _get_face_mesh():
    global _FACE_MESH
    if _FACE_MESH is None:
        try:
            import mediapipe as mp
            _FACE_MESH = mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
            )
            log.info("Mediapipe Face Mesh 로드 완료")
        except Exception as e:
            log.warning("Mediapipe Face Mesh 로드 실패 (Haar fallback): %s", e)
            _FACE_MESH = False
    return _FACE_MESH if _FACE_MESH is not False else None


def _detect_face_parts(rgb_image):
    """Mediapipe로 얼굴 부위별(피부/눈/입/눈썹) 마스크 추출.
    rgb_image: numpy RGB (H, W, 3) uint8
    returns: dict {part: mask (0~1 float32)} 또는 None
    """
    import numpy as np
    mesh = _get_face_mesh()
    if mesh is None:
        return None
    try:
        results = mesh.process(rgb_image)
        if not results.multi_face_landmarks:
            return None
        landmarks = results.multi_face_landmarks[0]
        h, w = rgb_image.shape[:2]
        import mediapipe as mp
        FM = mp.solutions.face_mesh

        def to_points(indices):
            return np.array(
                [[int(landmarks.landmark[i].x * w),
                  int(landmarks.landmark[i].y * h)] for i in indices],
                dtype=np.int32,
            )

        def make_mask(indices, blur_sigma=2.0):
            pts = to_points(indices)
            if len(pts) < 3:
                return None
            m = np.zeros((h, w), dtype=np.float32)
            hull = cv2.convexHull(pts)
            cv2.fillConvexPoly(m, hull, 1.0)
            m = cv2.GaussianBlur(m, (0, 0), blur_sigma)
            return np.clip(m, 0, 1)

        # 부위별 인덱스 (connection tuple → unique vertex set)
        LIPS = sorted({v for c in FM.FACEMESH_LIPS for v in c})
        L_EYE = sorted({v for c in FM.FACEMESH_LEFT_EYE for v in c})
        R_EYE = sorted({v for c in FM.FACEMESH_RIGHT_EYE for v in c})
        L_BROW = sorted({v for c in FM.FACEMESH_LEFT_EYEBROW for v in c})
        R_BROW = sorted({v for c in FM.FACEMESH_RIGHT_EYEBROW for v in c})
        OVAL = sorted({v for c in FM.FACEMESH_FACE_OVAL for v in c})

        return {
            "lips": make_mask(LIPS, 2.0),
            "left_eye": make_mask(L_EYE, 1.5),
            "right_eye": make_mask(R_EYE, 1.5),
            "left_brow": make_mask(L_BROW, 1.5),
            "right_brow": make_mask(R_BROW, 1.5),
            "oval": make_mask(OVAL, 8.0),
        }
    except Exception as e:
        log.debug("Face Mesh 처리 실패: %s", e)
        return None


def _detect_face_mask(gray):
    """OpenCV Haar Cascade로 얼굴 detect → 부드러운 마스크 반환.
    얼굴 없으면 None.
    """
    try:
        cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        if face_cascade.empty():
            return None
        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(100, 100)
        )
        if len(faces) == 0:
            return None
        # 가장 큰 얼굴 사용
        import numpy as np
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        mask = np.zeros_like(gray, dtype=np.float32)
        # 타원 마스크 — 얼굴 형태에 맞춤
        cv2.ellipse(mask, (x + w // 2, y + h // 2),
                    (int(w * 0.55), int(h * 0.65)),
                    0, 0, 360, 1.0, -1)
        # 부드러운 가장자리
        mask = cv2.GaussianBlur(mask, (0, 0), max(w // 8, 5))
        return np.clip(mask, 0, 1)
    except Exception as e:
        log.debug("얼굴 detect 실패: %s", e)
        return None


# Atkinson 디더링 (Mac 클래식) — 오차의 6/8 만 6픽셀에 분산.
# 1/4는 의도적으로 버림 → 자연스럽게 밝고 도트가 적음. 부드러운 결과.
_ATKINSON_WEIGHTS = (
    (0, 1, 1.0 / 8.0), (0, 2, 1.0 / 8.0),
    (1, -1, 1.0 / 8.0), (1, 0, 1.0 / 8.0), (1, 1, 1.0 / 8.0),
    (2, 0, 1.0 / 8.0),
)


def _stucki_dither(arr_uint8):
    """Atkinson error-diffusion — Stucki보다 도트 적고 부드러움.

    원리: 오차의 6/8(=75%)만 6픽셀에 분산하고 1/4는 버림.
    어두운 영역의 도트 밀도가 낮고, 그라데이션이 매끈하게 표현됨.
    Mac 초기 1bit 사진 인쇄에서 사용된 클래식한 알고리즘.
    (함수명은 호환성 위해 유지, 내부 알고리즘만 교체)
    """
    import numpy as np
    arr = arr_uint8.astype(np.float32)
    h, w = arr.shape
    for y in range(h):
        for x in range(w):
            old = arr[y, x]
            new = 255.0 if old > 127.0 else 0.0
            arr[y, x] = new
            err = old - new
            for dy, dx, wgt in _ATKINSON_WEIGHTS:
                ny = y + dy
                nx = x + dx
                if ny < h and 0 <= nx < w:
                    arr[ny, nx] += err * wgt
    return (arr > 127).astype(np.uint8) * 255


def _safe_textlength(draw, text, font, fallback_char_w=20):
    try:
        w = draw.textlength(text, font=font)
        return max(1, int(w))
    except Exception:
        return len(text) * fallback_char_w


def _draw_center(draw, text, font, y, canvas_w):
    """캔버스 폭 기준 정중앙에 텍스트. 폭 정확 측정 후 정수 좌표로 그림."""
    tw = _safe_textlength(draw, text, font)
    x = (canvas_w - tw) // 2
    draw.text((x, y), text, fill="black", font=font)
    return tw


def _fit_font(draw, text, max_w, start_size, min_size=14, bold=True, step=4):
    f = _find_korean_font(start_size, bold=bold)
    tw = _safe_textlength(draw, text, f)
    while tw > max_w and f.size > min_size:
        f = _find_korean_font(f.size - step, bold=bold)
        tw = _safe_textlength(draw, text, f)
    return f, tw


def _textlength_spaced(draw, text, font, extra=4):
    """글자별 자간 추가했을 때의 전체 폭"""
    total = 0
    for ch in text:
        total += _safe_textlength(draw, ch, font) + extra
    return max(1, total - extra)


def _draw_text_spaced(draw, text, font, y, canvas_w, extra=4, fill="black"):
    """자간 강조 텍스트를 정중앙 정렬로 그림 (명품 브랜드 무드)"""
    tw = _textlength_spaced(draw, text, font, extra)
    x = (canvas_w - tw) // 2
    for ch in text:
        draw.text((x, y), ch, fill=fill, font=font)
        x += _safe_textlength(draw, ch, font) + extra
    return tw


def _fit_font_spaced(draw, text, max_w, start_size, min_size=14,
                     bold=False, step=2, extra=4):
    f = _find_korean_font(start_size, bold=bold)
    tw = _textlength_spaced(draw, text, f, extra)
    while tw > max_w and f.size > min_size:
        f = _find_korean_font(f.size - step, bold=bold)
        tw = _textlength_spaced(draw, text, f, extra)
    return f, tw


def build_kpop_header(width: int, title: str) -> Image.Image:
    """미니멀 명품 브랜드 무드 헤더.

    - 굵은 검정 선 모두 제거. 1dot 가는 선 한 줄만 사용
    - 메인 카피는 자간 강조(letter-spacing)로 우아하게
    - 브랜드 라벨 / 메인 / 메타 3섹션 동적 높이 (빈 공간 없음)
    """
    margin = 60

    # 사전 측정
    _tmp = Image.new("RGB", (width, 10), "white")
    _td = ImageDraw.Draw(_tmp)

    # 브랜드 라벨 (자간 강조)
    brand = "K · CULTURE · VR · TOWER"
    fbrand, _ = _fit_font_spaced(_td, brand, width - margin * 2,
                                  start_size=17, min_size=12, step=1, extra=2)
    brand_extra = 2

    # 메인 카피 (자간 강조)
    main = (title or "I'M A K-POP STAR").upper()
    fmain, _ = _fit_font_spaced(_td, main, width - margin * 2,
                                 start_size=46, min_size=22, step=2,
                                 bold=True, extra=4)
    main_extra = 4

    # 메타 (한 줄: ON AIR · N SEOUL TOWER · 날짜)
    fmeta = _find_korean_font(14, bold=False)

    # 섹션 위치 — pad_top 0 (영수증 첫 픽셀부터 콘텐츠 시작)
    pad_top = 0
    y_brand = pad_top
    gap1 = 12
    y_line1 = y_brand + fbrand.size + gap1
    gap2 = 16
    y_main = y_line1 + gap2
    gap3 = 16
    y_line2 = y_main + fmain.size + gap3
    gap4 = 10
    y_meta = y_line2 + gap4
    pad_bot = 12
    h = y_meta + fmeta.size + pad_bot

    img = Image.new("RGB", (width, h), "white")
    draw = ImageDraw.Draw(img)

    # 1) 브랜드 라벨 (자간 강조, 정중앙)
    _draw_text_spaced(draw, brand, fbrand, y_brand, width, extra=brand_extra)

    # 2) 가는 가로선 (1dot)
    draw.line([(margin, y_line1), (width - margin, y_line1)], fill="black", width=1)

    # 3) 메인 빅 카피 (자간 강조 + 굵게)
    _draw_text_spaced(draw, main, fmain, y_main, width, extra=main_extra)

    # 4) 가는 가로선 (1dot)
    draw.line([(margin, y_line2), (width - margin, y_line2)], fill="black", width=1)

    # 5) 메타 — 짧게 한 줄 (자간 없음, 흐림 방지 위해 굵게)
    today = datetime.now().strftime("%Y.%m.%d")
    meta = f"ON AIR  ·  N SEOUL TOWER  ·  {today}"
    fmeta_fit, _ = _fit_font(draw, meta, width - margin * 2,
                              start_size=15, min_size=11, step=1, bold=True)
    _draw_center(draw, meta, fmeta_fit, y_meta, width)
    return img


def build_kpop_footer(width: int, add_timestamp: bool) -> Image.Image:
    """미니멀 명품 무드 푸터 — 1dot 가로선 + 자간 강조 텍스트."""
    margin = 60

    _tmp = Image.new("RGB", (width, 10), "white")
    _td = ImageDraw.Draw(_tmp)

    # 시간 — 자간 X, 굵게 (영수증에서 또렷이)
    ts = datetime.now().strftime("%Y. %m. %d   %H:%M:%S")
    fts, _ = _fit_font(_td, ts, width - margin * 2,
                        start_size=22, min_size=14, step=1, bold=True)

    # 태그라인 한 줄 — 자간 X, 굵게
    tagline = "THE WORLD'S FIRST  ·  YEAR-ROUND K-POP VR THEATER"
    ftag, _ = _fit_font(_td, tagline, width - margin * 2,
                         start_size=16, min_size=11, step=1, bold=True)

    # 도메인 — 자간 X, 굵게 강조
    dom = "www.kvrtower.com  ·  @kvrtower"
    fdom, _ = _fit_font(_td, dom, width - margin * 2,
                         start_size=24, min_size=16, step=1, bold=True)

    pad_top = 20
    gap1 = 16
    sec1_h = (fts.size + 14) if add_timestamp else 0
    sec2_h = ftag.size + 18
    sec3_h = fdom.size + 0
    pad_bot = 20
    h = pad_top + 1 + gap1 + sec1_h + sec2_h + sec3_h + gap1 + 1 + pad_bot

    img = Image.new("RGB", (width, h), "white")
    draw = ImageDraw.Draw(img)

    y = pad_top
    draw.line([(margin, y), (width - margin, y)], fill="black", width=1)
    y += gap1

    if add_timestamp:
        _draw_center(draw, ts, fts, y, width)
        y += fts.size + 14

    _draw_center(draw, tagline, ftag, y, width)
    y += ftag.size + 18

    _draw_center(draw, dom, fdom, y, width)
    y += fdom.size + gap1

    draw.line([(margin, y), (width - margin, y)], fill="black", width=1)
    return img


def generate_event_code() -> str:
    """매장 카운터 교환용 고유 코드.
    KVR + YYMMDDHHMMSS + 3자리 랜덤 = 18자 (CODE128 인식 양호)
    """
    import random
    return f"KVR{datetime.now().strftime('%y%m%d%H%M%S')}{random.randint(100, 999)}"


def get_asset_path(name: str) -> Path | None:
    """assets 폴더의 파일 경로 — EXE/개발 실행 모두 안전하게.
    프로젝트 root/assets, EXE 옆 assets 등 후보 경로 순차 탐색.
    """
    candidates = []
    if getattr(sys, "frozen", False):
        exe_dir = Path(sys.executable).resolve().parent
        candidates += [exe_dir / "assets" / name,
                       exe_dir.parent / "assets" / name]
        mei = getattr(sys, "_MEIPASS", None)
        if mei:
            candidates.append(Path(mei) / "assets" / name)
    else:
        base = Path(__file__).resolve().parent.parent
        candidates.append(base / "assets" / name)
    for p in candidates:
        if p.exists():
            return p
    return None


def get_local_ip() -> str:
    """로컬 네트워크 IP — 같은 와이파이의 폰에서 접속할 주소"""
    import socket
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return "127.0.0.1"


REMOTE_PORT = 5555


def make_qr_for_url(url: str, size: int = 280) -> Image.Image:
    """폰으로 스캔할 QR — 단순 흑백, 영수증/화면 둘 다 또렷"""
    try:
        import qrcode
        qr = qrcode.QRCode(border=2, box_size=10,
                           error_correction=qrcode.constants.ERROR_CORRECT_M)
        qr.add_data(url)
        qr.make(fit=True)
        img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
        return img.resize((size, size), Image.NEAREST)
    except Exception:
        img = Image.new("RGB", (size, size), "white")
        d = ImageDraw.Draw(img)
        f = _find_korean_font(14, bold=True)
        _draw_center(d, "QR 생성 실패", f, size // 2 - 10, size)
        return img


def start_remote_server(trigger_print_callable):
    """Flask 백그라운드 서버 — 폰에서 PRINT 버튼 누르면 실행

    trigger_print_callable: 메인 스레드에서 안전하게 인쇄를 트리거하는 함수
    (예: lambda: app.root.after(0, app._on_print))
    """
    try:
        from flask import Flask
    except Exception as e:
        log.warning("Flask 임포트 실패 — 원격 서버 비활성화: %s", e)
        return

    flask_app = Flask(__name__)
    flask_app.logger.disabled = True
    import logging as _lg
    _lg.getLogger("werkzeug").disabled = True

    @flask_app.route("/")
    def index():
        return """<!DOCTYPE html>
<html lang="ko"><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, user-scalable=no">
<title>K-Culture VR Tower</title>
<style>
  *{margin:0;padding:0;box-sizing:border-box}
  body{background:#111;color:#fff;font-family:-apple-system,sans-serif;
       display:flex;flex-direction:column;align-items:center;
       justify-content:center;min-height:100vh;padding:24px;text-align:center}
  h1{font-size:18px;font-weight:600;letter-spacing:4px;color:#7BA6DD;
     margin-bottom:8px}
  .sub{font-size:13px;color:#888;margin-bottom:48px;letter-spacing:1px}
  button{font-size:28px;font-weight:700;padding:48px 0;width:100%;max-width:340px;
         background:#7BA6DD;color:#fff;border:none;border-radius:24px;
         box-shadow:0 8px 24px rgba(123,166,221,0.3);
         cursor:pointer;letter-spacing:6px;transition:all .1s}
  button:active{background:#5a8fcd;transform:scale(0.97)}
  button:disabled{background:#444;color:#888}
  .status{margin-top:32px;font-size:14px;color:#aaa;min-height:24px}
</style></head><body>
  <h1>K · CULTURE · VR · TOWER</h1>
  <div class="sub">REMOTE PHOTO PRINT</div>
  <button id="b" onclick="go()">PRINT</button>
  <div class="status" id="s">사진을 보고 준비되면 누르세요</div>
<script>
async function go(){
  const b=document.getElementById('b'),s=document.getElementById('s');
  b.disabled=true;s.textContent='인쇄 중…';
  try{const r=await fetch('/print');const t=await r.text();s.textContent=t;}
  catch(e){s.textContent='연결 실패';}
  setTimeout(()=>{b.disabled=false;s.textContent='다시 누를 수 있습니다';},2500);
}
</script></body></html>"""

    @flask_app.route("/print")
    def remote_print():
        try:
            trigger_print_callable()
            return "OK — 인쇄 시작!"
        except Exception as e:
            return f"실패: {e}", 500

    def run():
        try:
            flask_app.run(host="0.0.0.0", port=REMOTE_PORT,
                          debug=False, use_reloader=False, threaded=True)
        except Exception as e:
            log.warning("원격 서버 실행 실패: %s", e)

    t = threading.Thread(target=run, daemon=True)
    t.start()
    log.info("원격 서버 시작: http://%s:%d", get_local_ip(), REMOTE_PORT)


def get_photo_save_dir() -> Path:
    """인쇄된 사진을 모아두는 폴더. EXE 빌드/개발 실행 양쪽 모두 안정.
    OneDrive 동기화 폴더 안이면 다른 PC에서도 자동 접근 가능.
    """
    if getattr(sys, "frozen", False):
        base = Path(sys.executable).resolve().parent
    else:
        base = Path(__file__).resolve().parent.parent
    folder = base / "photos"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_photo_for_instagram(square_pil: Image.Image, code: str) -> Path:
    """카메라 raw 정사각 컬러 사진을 인스타 업로드용으로 저장.
    파일명: YYYYMMDD_HHMMSS_KVR...코드.jpg
    """
    folder = get_photo_save_dir()
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = folder / f"{ts}_{code}.jpg"
    img = square_pil.convert("RGB")
    # 인스타 권장 최소 폭 1080 — 카메라가 더 작으면 업스케일
    if img.size[0] < 1080:
        scale = 1080 / img.size[0]
        img = img.resize((1080, int(img.size[1] * scale)), Image.LANCZOS)
    img.save(path, "JPEG", quality=92)
    return path


def make_barcode_image(code_text: str, target_width: int) -> Image.Image:
    """CODE128 바코드 이미지.

    python-barcode 가 내부적으로 자체 폰트를 truetype 로드하다 OSError 가
    날 수 있으므로(EXE 빌드 환경에서 흔함) write_text=False 로 받고,
    바코드 아래 숫자 텍스트는 우리가 직접 안전한 윈도우 폰트로 그려 합성한다.
    """
    try:
        from io import BytesIO
        from barcode import Code128
        from barcode.writer import ImageWriter
    except Exception:
        # python-barcode 자체가 없으면 텍스트만 그려서 폴백
        fallback = Image.new("RGB", (target_width, 60), "white")
        d = ImageDraw.Draw(fallback)
        f = _find_korean_font(22, bold=True)
        tw = _safe_textlength(d, code_text, f)
        d.text(((target_width - tw) // 2, 18), code_text, fill="black", font=f)
        return fallback

    options = {
        "module_height": 16.0,
        "module_width": 0.34,
        "write_text": False,    # ← 핵심: 라이브러리가 폰트 로드 안 하게
        "quiet_zone": 4.0,
        "background": "white",
        "foreground": "black",
        "dpi": 203,
    }
    writer = ImageWriter()
    code = Code128(code_text, writer=writer)
    buf = BytesIO()
    try:
        code.write(buf, options=options)
    except Exception:
        # 그래도 실패하면 텍스트만 폴백
        fallback = Image.new("RGB", (target_width, 60), "white")
        d = ImageDraw.Draw(fallback)
        f = _find_korean_font(22, bold=True)
        tw = _safe_textlength(d, code_text, f)
        d.text(((target_width - tw) // 2, 18), code_text, fill="black", font=f)
        return fallback

    buf.seek(0)
    bar_img = Image.open(buf).convert("RGB")
    # 바코드 폭 맞춤
    w, h = bar_img.size
    if w != target_width:
        scale = target_width / w
        bar_img = bar_img.resize((target_width, max(1, int(h * scale))),
                                  Image.LANCZOS)
        w, h = bar_img.size

    # 숫자 텍스트를 바코드 아래에 안전한 폰트로 직접 합성
    text_pad = 10        # 바코드와 숫자 사이 여백
    text_below_pad = 6   # 숫자 아래 여백
    f_txt = _find_korean_font(18, bold=True)
    text_h = f_txt.size + text_pad + text_below_pad

    combined = Image.new("RGB", (w, h + text_h), "white")
    combined.paste(bar_img, (0, 0))
    d = ImageDraw.Draw(combined)
    tw = _safe_textlength(d, code_text, f_txt)
    d.text(((w - tw) // 2, h + text_pad), code_text, fill="black", font=f_txt)
    return combined


def wrap_photo_as_magazine(photo: Image.Image, width: int) -> Image.Image:
    """미니멀 갤러리 액자 wrap — 가는 검정 1dot 프레임 + 자간 강조 캡션.

    사진 폭을 줄여(좌우 여백 확대) 전체 영수증 높이를 절약 — 분할 없이
    한 번에 인쇄되도록 영수증 총 길이 ≤ 2303dot 유지.
    """
    margin = 20          # 좌우 흰 여백 최소화 — 사진 가로폭 가능한 크게
    border = 1
    label_band = 28
    caption_band = 30
    inner_pad = 6

    inner_w = width - margin * 2 - (border + inner_pad) * 2
    pw, ph = photo.size
    scale = inner_w / pw
    photo = photo.resize((inner_w, max(1, int(ph * scale))), Image.LANCZOS)
    pw, ph = photo.size

    total_h = label_band + border + inner_pad + ph + inner_pad + border + caption_band
    canvas = Image.new("RGB", (width, total_h), "white")
    draw = ImageDraw.Draw(canvas)

    # 상단 라벨 (자간 강조 영문) + 양옆 가는 라인
    label = "K - POP MAGAZINE"
    flbl, _ = _fit_font_spaced(draw, label, width - margin * 2 - 80,
                                 start_size=15, min_size=10, step=1, extra=2)
    tw_lbl = _textlength_spaced(draw, label, flbl, 2)
    cx = width // 2
    ly = label_band // 2 + 4
    draw.line([(margin, ly), (cx - tw_lbl // 2 - 14, ly)], fill="black", width=1)
    draw.line([(cx + tw_lbl // 2 + 14, ly), (width - margin, ly)], fill="black", width=1)
    _draw_text_spaced(draw, label, flbl, 10, width, extra=2)

    # 가는 검정 액자 (1dot)
    fx = margin
    fy = label_band
    fw = pw + (border + inner_pad) * 2
    fh = ph + (border + inner_pad) * 2
    draw.rectangle([(fx, fy), (fx + fw - 1, fy + fh - 1)],
                   outline="black", width=border)
    canvas.paste(photo, (fx + border + inner_pad, fy + border + inner_pad))

    # 하단 캡션 — 매장 브랜드명 (풀네임)
    cap = "PHOTOGRAPHED AT K-CULTURE VR TOWER"
    fcap, _ = _fit_font(draw, cap, width - margin * 2,
                         start_size=16, min_size=11, step=1, bold=True)
    _draw_center(draw, cap, fcap, fy + fh + 14, width)
    return canvas


def build_event_box(width: int, code_text: str) -> Image.Image:
    """2단계 응모 이벤트 박스 — 영역을 굵은 검정 외곽으로 명확히 구분.

    STEP 01 : 100% 즉시 당첨 (피드 공개 게시 → 매장 카운터 바코드 제시)
    STEP 02 : 추첨 — 2026.9.16 부산 TMA 시상식 VIP 초대권 2장
    """
    margin_outer = 60   # 헤더/푸터/사진 wrap과 동일 흰 여백
    border = 2          # 가는 검정 액자 (미니멀)
    pad = 22            # 박스 내부 여백
    inner_w = width - margin_outer * 2

    _tmp = Image.new("RGB", (width, 10), "white")
    _td = ImageDraw.Draw(_tmp)

    # 바코드 미리 생성
    barcode = make_barcode_image(code_text, inner_w - pad * 2 - 20)

    # 영문 메인 (외국인 손님 기본) + 한글 부제 작게 — 폰트 계층
    # 안전 마진 — 박스 외곽선과 텍스트 사이 충분한 여백 확보
    inner_text_w = inner_w - pad * 2 - border * 2 - 24
    f_step = _find_korean_font(15, bold=True)        # STEP 01 라벨
    f_eng_main = _find_korean_font(36, bold=True)    # 영문 메인 카피 (강조)
    f_eng_sub = _find_korean_font(17, bold=True)     # 영문 부제
    f_kor_sub = _find_korean_font(13, bold=True)     # 한글 부제 (작게)
    f_eng_body = _find_korean_font(16, bold=True)    # 영문 안내
    f_kor_small = _find_korean_font(12, bold=True)   # 한글 보조
    f_tag = _find_korean_font(28, bold=True)         # 해시태그
    f_box_title = _find_korean_font(24, bold=True)   # 박스 헤더 EVENT

    # 박스 헤더 (검정 배경 - 영역 구분 강조)
    box_header_h = 48

    # 충분히 큰 캔버스에 그린 뒤 마지막 y 위치로 trim — 폰트 변경에 강함
    img = Image.new("RGB", (width, 2400), "white")
    draw = ImageDraw.Draw(img)

    bx0 = margin_outer
    bx1 = width - margin_outer

    # 박스 헤더 검정 배경 (영역 강조)
    draw.rectangle([(bx0 + border, border),
                    (bx1 - border, border + box_header_h)], fill="black")
    title = "EVENT"
    _draw_text_spaced(draw, title, f_box_title,
                      border + (box_header_h - f_box_title.size) // 2 - 2,
                      width, extra=6, fill="white")

    y = border + box_header_h + pad

    def _fit_and_draw_spaced(text, start_size, min_size=12, extra=2, bold=True):
        f, _ = _fit_font_spaced(draw, text, inner_text_w,
                                 start_size=start_size, min_size=min_size,
                                 step=1, bold=bold, extra=extra)
        _draw_text_spaced(draw, text, f, y, width, extra=extra)
        return f.size

    def _fit_and_draw(text, start_size, min_size=10, bold=True):
        f, _ = _fit_font(draw, text, inner_text_w,
                          start_size=start_size, min_size=min_size,
                          step=1, bold=bold)
        _draw_center(draw, text, f, y, width)
        return f.size

    def _draw_left(text, font, x_left):
        draw.text((x_left, y), text, fill="black", font=font)

    # ── STEP 01 ────────────────────────────────────
    sz = _fit_and_draw_spaced("STEP 01", 15, extra=3); y += sz + 6
    sz = _fit_and_draw_spaced("INSTANT GIFT", 32, min_size=20, extra=2)
    y += sz + 4
    sz = _fit_and_draw("Guaranteed for everyone", 15); y += sz + 12

    # 단계별 번호 안내 (왼쪽 정렬, 영문)
    f_step_num = _find_korean_font(16, bold=True)
    f_step_txt = _find_korean_font(16, bold=False)
    x_num = bx0 + pad + 8
    x_txt = x_num + 26

    steps = [
        "1.  Post your photo on Instagram.",
        "2.  Add these hashtags in this order:",
        "         #tma    #kvrtower",
        "3.  Set your account to PUBLIC.",
        "4.  Show the barcode below at",
        "         the counter to claim your gift.",
    ]
    for line in steps:
        if line.strip().startswith("#") or line.strip().startswith("the counter"):
            # 들여쓰기 줄은 일반 폰트, 본 줄은 굵게
            draw.text((x_num, y), line, fill="black", font=f_step_txt)
        else:
            draw.text((x_num, y), line, fill="black", font=f_step_num)
        y += f_step_num.size + 2

    y += 8
    sz = _fit_and_draw("Your photo may be featured on @kvrtower", 14)
    y += sz + 12

    # 한글 요약 (한 줄 컴팩트)
    draw.line([(bx0 + pad + 50, y), (bx1 - pad - 50, y)], fill="black", width=1)
    y += 6
    sz = _fit_and_draw("두 해시태그 순서대로 + \"전체 공개\" 계정", 13); y += sz + 2
    sz = _fit_and_draw("→ 카운터에 바코드 제시 → 경품 증정", 13); y += sz + 10

    # 바코드 (가운데 정렬)
    bx = (width - barcode.size[0]) // 2
    img.paste(barcode, (bx, y))
    y += barcode.size[1] + 14

    # ── 구분 가로선 (STEP 01 끝) ──
    y += 4
    draw.line([(bx0 + pad, y), (bx1 - pad, y)], fill="black", width=1)
    y += 12

    # ── STEP 02 — FANNSTAR ────────────────────────
    sz = _fit_and_draw_spaced("STEP 02", 15, extra=3); y += sz + 6
    sz = _fit_and_draw_spaced("JOIN FANNSTAR", 32, min_size=20, extra=2)
    y += sz + 4
    sz = _fit_and_draw("Free TMA Highlight VR admission", 15); y += sz + 12

    # FANNSTAR 단계 안내
    steps2 = [
        "1.  Install the FANNSTAR app",
        "         (TMA's official voting platform)",
        "2.  Sign up with your account.",
        "3.  Show your sign-up screen at",
        "         the counter to watch the",
        "         TMA Highlight VR for free.",
    ]
    for line in steps2:
        sub = (line.strip().startswith("(") or line.strip().startswith("the counter")
               or line.strip().startswith("TMA Highlight"))
        draw.text((x_num, y), line, fill="black",
                  font=(f_step_txt if sub else f_step_num))
        y += f_step_num.size + 2

    y += 14

    # STEP 02 QR 코드 합성 — FANNSTAR 가입 / TMA 초대권 페이지
    qr_path = get_asset_path("qr_step2_tma.png")
    if qr_path is not None:
        try:
            qr_img = Image.open(str(qr_path)).convert("RGB")
            qr_size = 200   # 영수증 한 번 전송 한계 위해 축소 (인식 OK)
            qr_img = qr_img.resize((qr_size, qr_size), Image.LANCZOS)  # noqa: F841
            qr_x = (width - qr_size) // 2
            img.paste(qr_img, (qr_x, y))
            y += qr_size + 6
            sz = _fit_and_draw("Scan with your phone camera", 13); y += sz + 14
        except Exception:
            pass

    # STEP 02 한글 요약 (한 줄 컴팩트)
    draw.line([(bx0 + pad + 50, y), (bx1 - pad - 50, y)], fill="black", width=1)
    y += 6
    sz = _fit_and_draw("팬앤스타 앱 설치 + 회원가입 → TMA 하이라이트 VR 무료", 13)
    y += sz + 10

    # ── 구분 가로선 (STEP 02 끝) ──
    y += 4
    draw.line([(bx0 + pad, y), (bx1 - pad, y)], fill="black", width=1)
    y += 12

    # ── STEP 03 — VIP INVITATION ──────────────────
    sz = _fit_and_draw_spaced("STEP 03", 15, extra=3); y += sz + 6
    sz = _fit_and_draw_spaced("VIP INVITATION", 32, min_size=20, extra=2)
    y += sz + 4
    sz = _fit_and_draw("Lucky draw open to all participants", 14); y += sz + 10

    sz = _fit_and_draw("2 winners will receive VIP tickets to", 15); y += sz + 8

    sz = _fit_and_draw_spaced("TMA AWARDS 2026", 24, min_size=18, extra=2)
    y += sz + 4
    sz = _fit_and_draw("The Fact Music Awards", 14); y += sz + 6

    sz = _fit_and_draw("Sep 16, 2026  ·  Busan, Korea", 16); y += sz + 8

    # STEP 03 한글 요약 (한 줄 컴팩트)
    draw.line([(bx0 + pad + 50, y), (bx1 - pad - 50, y)], fill="black", width=1)
    y += 6
    sz = _fit_and_draw("TMA 시상식 VIP 초대권 2장 추첨 · 2026.9.16 부산", 13)
    y += sz + 10

    # 박스 외곽 액자 + trim (안전 마진 줄임)
    box_bottom = y
    draw.rectangle([(bx0, 0), (bx1 - 1, box_bottom - 1)],
                   outline="black", width=border)
    img = img.crop((0, 0, width, box_bottom + 12))
    return img


def _camera_backends():
    """윈도우는 DSHOW + MSMF 둘 다 시도 (캡처카드/분배기 호환)"""
    if os.name == "nt":
        return [cv2.CAP_DSHOW, cv2.CAP_MSMF]
    return [0]


def _list_camera_indexes(max_index: int = 10) -> list[int]:
    """0..max_index-1 에서 열리는 카메라 인덱스 탐지.
    캡처카드/분배기는 인덱스가 높거나 MSMF 백엔드가 필요할 수 있어
    범위를 10까지 넓히고 두 백엔드 모두 시도.
    """
    found = []
    for i in range(max_index):
        for backend in _camera_backends():
            cap = cv2.VideoCapture(i, backend)
            opened = cap is not None and cap.isOpened()
            ok = False
            if opened:
                ok, _ = cap.read()
            if cap is not None:
                cap.release()
            if ok:
                found.append(i)
                break  # 이 인덱스는 잡힘, 다음 인덱스로
    return sorted(set(found)) or [0]


class CameraSource:
    """카메라 캡처 스레드 — 항상 최신 프레임만 보관"""

    def __init__(self, index: int = 0):
        self.index = index
        self._cap = None
        self._frame = None
        self._lock = threading.Lock()
        self._stop = False
        self._thread = None
        self._open()

    def _open(self):
        # 여러 백엔드 시도 — 캡처카드/분배기는 MSMF 가 필요한 경우가 있음
        cap = None
        for backend in _camera_backends():
            c = cv2.VideoCapture(self.index, backend)
            if c is not None and c.isOpened():
                ok, _f = c.read()
                if ok and _f is not None:
                    cap = c
                    log.info("카메라 %d 백엔드 %s 로 열림", self.index, backend)
                    break
            if c is not None:
                c.release()
        if cap is None:
            # 최후 폴백 — 백엔드 미지정
            cap = cv2.VideoCapture(self.index)

        # 최고 해상도 요청 + MJPG 코덱
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        cap.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)

        # 첫 프레임을 실제로 받을 때까지 polling (최대 3초)
        # 카메라가 OS 레벨에서 준비될 시간 확보 — 한 번 더 선택해야 잡히던 문제 해결
        warmup_ok = False
        for _ in range(30):
            if cap is not None and cap.isOpened():
                ok, _f = cap.read()
                if ok and _f is not None:
                    warmup_ok = True
                    break
            time.sleep(0.1)

        aw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        ah = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        log.info("카메라 %d 해상도: %dx%d  (warmup=%s)",
                 self.index, aw, ah, "OK" if warmup_ok else "FAIL")
        self._cap = cap

    def grab_avg(self, count: int = 5, interval: float = 0.04):
        """여러 프레임 평균 — 노이즈 줄이고 디테일 살림.
        센서 노이즈는 프레임마다 랜덤, 평균하면 √N 만큼 감소.
        5프레임이면 약 55% 노이즈 감소 → 미세 디테일 디더링 시 선명.
        """
        import numpy as np
        frames = []
        for _ in range(count):
            f = self.grab()
            if f is not None:
                frames.append(f.astype(np.float32))
            time.sleep(interval)
        if not frames:
            return self.grab()
        avg = np.mean(frames, axis=0)
        return np.clip(avg, 0, 255).astype(np.uint8)

    def start(self):
        if self._thread and self._thread.is_alive():
            return
        if self._cap is None or not self._cap.isOpened():
            self._open()
        self._stop = False
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    @property
    def is_on(self) -> bool:
        return (self._thread is not None and self._thread.is_alive()
                and self._cap is not None and self._cap.isOpened())

    def _loop(self):
        while not self._stop:
            if self._cap is None or not self._cap.isOpened():
                time.sleep(0.1)
                continue
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.03)
                continue
            with self._lock:
                self._frame = frame
            time.sleep(0.01)

    def grab(self):
        with self._lock:
            return None if self._frame is None else self._frame.copy()

    def change(self, index: int):
        self.stop()
        self.index = index
        self._open()
        self.start()

    def stop(self):
        self._stop = True
        if self._thread:
            self._thread.join(timeout=1.0)
            self._thread = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None


def cv_to_pil(frame) -> Image.Image:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def crop_to_print_aspect(pil: Image.Image, aspect_h_over_w: float) -> Image.Image:
    """프린터 폭 비율(세로/가로)에 맞춰 중앙 크롭.

    aspect_h_over_w <= 0 이면 원본 비율 그대로 (자르기 없음).
    카메라 16:9 영상을 강제 크롭하면 좌우가 잘리므로, 기본은 0(원본).
    """
    if aspect_h_over_w <= 0:
        return pil
    w, h = pil.size
    target_h = int(w * aspect_h_over_w)
    if target_h <= h:
        top = (h - target_h) // 2
        return pil.crop((0, top, w, top + target_h))
    target_w = int(h / aspect_h_over_w)
    left = (w - target_w) // 2
    return pil.crop((left, 0, left + target_w, h))


def build_print_image(
    pil: Image.Image,
    title: str = "",
    add_timestamp: bool = True,
    aspect_h_over_w: float = DEFAULT_ASPECT,
    event_code: str = "",
    with_decoration: bool = True,
    with_event: bool = True,
) -> Image.Image:
    """최종 인쇄 이미지 — 헤더 / 사진(화보 wrap) / 푸터 (+ 이벤트 박스).

    처리 흐름 (가장 정교한 순서):
    1) 카메라 원본 → 정사각 크롭 (큰 해상도 유지)
    2) 큰 사이즈에서 enhance_for_thermal — 디테일 정확하게 추출
    3) LANCZOS3 다운샘플 → 인쇄 폭으로 (안티앨리어싱이 자연스러움)
    4) 작은 사이즈에서 Stucki 디더링 (부드러운 점 패턴)
    """
    cropped = crop_to_print_aspect(pil, aspect_h_over_w)

    # 1) 큰 사이즈에서 enhance — 더 많은 디테일 정보로 처리 (그레이 반환)
    enhanced_large = enhance_for_thermal(cropped)

    # 2) LANCZOS 다운샘플 → 인쇄 폭
    w0, h0 = enhanced_large.size
    scale = PRINT_DOTS / w0
    new_w = PRINT_DOTS
    new_h = max(1, int(h0 * scale))
    img_small = enhanced_large.resize((new_w, new_h), Image.LANCZOS)

    # 3) 디더링은 인쇄 백엔드에서 모드에 맞춰 처리
    img = img_small.convert("RGB")

    if not with_decoration:
        return img

    header = build_kpop_header(PRINT_DOTS, title)
    magazine = wrap_photo_as_magazine(img.convert("RGB"), PRINT_DOTS)
    footer = build_kpop_footer(PRINT_DOTS, add_timestamp)

    if not with_event:
        # 사진만 모드 — 헤더 + 화보 사진 + 푸터
        tail_safe = 24
        total_h = (header.size[1] + magazine.size[1] + footer.size[1]
                   + tail_safe)
        canvas = Image.new("RGB", (PRINT_DOTS, total_h), "white")
        y = 0
        for part in (header, magazine, footer):
            canvas.paste(part, (0, y))
            y += part.size[1]
        return canvas

    code = event_code or generate_event_code()
    event = build_event_box(PRINT_DOTS, code)

    gap = 8
    tail_safe = 24
    total_h = (header.size[1] + magazine.size[1] + footer.size[1]
               + gap + event.size[1] + tail_safe)
    canvas = Image.new("RGB", (PRINT_DOTS, total_h), "white")
    y = 0
    for part in (header, magazine, footer):
        canvas.paste(part, (0, y))
        y += part.size[1]
    canvas.paste(event, (0, y + gap))
    return canvas


class App:
    PREVIEW_W = 640  # 미리보기 캔버스 폭 (UI 표시용)

    def __init__(self, root: tk.Tk):
        self.root = root
        self.settings = _load_settings()

        # 백엔드
        self.printer = PrinterManager(
            printer_name=self.settings.get("printer_name", ""),
            interval=float(self.settings.get("interval", 1.0)),
        )
        # 카메라 인덱스 자동 검증 — 저장된 인덱스가 사용 불가면 첫 가능 인덱스로 폴백
        saved_idx = int(self.settings.get("camera_index", 0))
        available = _list_camera_indexes()
        if saved_idx not in available and available:
            log.info("저장된 카메라 인덱스 %d 사용 불가 → %d 로 자동 폴백",
                     saved_idx, available[0])
            saved_idx = available[0]
            self.settings["camera_index"] = saved_idx
            _save_settings(self.settings)
        self.camera = CameraSource(index=saved_idx)
        self.camera.start()

        self._kiosk = False
        self._camera_only = False
        self._auto_face_frames = 0
        self._auto_countdown = 0
        self._auto_countdown_start = 0.0
        self._auto_last_print = 0.0
        self._auto_last_countdown_speak = -1
        self._auto_no_face_count = 0   # 카운트다운 중 얼굴 사라짐 카운터
        self._locked_face = None       # 카운트다운 락온: (cx, cy, w, h) 또는 None
        self._last_greet_time = 0.0    # 마지막 인사 시각 (반복 인사 방지)
        self._last_lead_in_time = 0.0  # 마지막 호객 음성 시각
        self._lead_in_active = False   # 호객 단계 표시 플래그
        self._prev_motion_gray = None  # 움직임 감지용 이전 프레임 (다운샘플 grayscale)
        # TTS 백그라운드 워밍업 — 첫 음성 호출 시 지연 0
        threading.Thread(target=_ensure_tts_proc, daemon=True).start()
        self._build_ui()
        # 시작 시 자동으로 전체화면(키오스크) 진입
        self.root.after(150, self._toggle_kiosk)
        self._refresh_loop()

    def _build_ui(self):
        self.root.title(APP_TITLE)
        self.root.configure(bg="#111")
        # 시작 시 화면 사이즈에 맞춰 즉시 최대화
        try:
            sw = self.root.winfo_screenwidth()
            sh = self.root.winfo_screenheight()
            self.root.geometry(f"{sw}x{sh}+0+0")
            self.root.state("zoomed")
        except Exception:
            self.root.geometry("1280x960")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass

        top = tk.Frame(self.root, bg="#111")
        top.pack(fill="x", padx=16, pady=(14, 6))
        self._top_bar = top

        tk.Label(top, text=APP_TITLE, fg="white", bg="#111",
                 font=("Malgun Gothic", 18, "bold")).pack(side="left")
        self.status = tk.Label(top, text="", fg="#9bd1ff", bg="#111",
                               font=("Malgun Gothic", 11))
        self.status.pack(side="right")

        # 본문: 좌측 미리보기 / 우측 옵션
        body = tk.Frame(self.root, bg="#111")
        body.pack(fill="both", expand=True, padx=16, pady=8)

        # 좌측 미리보기
        left = tk.Frame(body, bg="#000", bd=1, relief="solid")
        left.pack(side="left", fill="both", expand=True)
        self.preview = tk.Label(left, bg="#000")
        self.preview.pack(fill="both", expand=True)

        # 우측 옵션
        right = tk.Frame(body, bg="#1a1a1a", width=300)
        right.pack(side="right", fill="y", padx=(12, 0))
        right.pack_propagate(False)
        self._right_panel = right

        self._opt_label(right, "카메라")
        cam_row = tk.Frame(right, bg="#1a1a1a")
        cam_row.pack(fill="x", padx=12, pady=(0, 4))
        cams = _list_camera_indexes()
        self.var_cam = tk.StringVar(value=str(self.camera.index))
        self.cam_combo = ttk.Combobox(cam_row, textvariable=self.var_cam,
                                      state="readonly",
                                      values=[str(i) for i in cams])
        self.cam_combo.pack(side="left", fill="x", expand=True)
        self.cam_combo.bind("<<ComboboxSelected>>", self._on_change_camera)
        tk.Button(cam_row, text="↻", command=self._refresh_camera_list,
                  bg="#333", fg="white", activebackground="#444",
                  font=("Malgun Gothic", 10, "bold"), relief="flat", bd=0,
                  cursor="hand2", width=3).pack(side="right", padx=(4, 0))
        # 현재 카메라 해상도 표시 — 폰 캠인지 노트북 캠인지 구분에 도움
        self.cam_res_label = tk.Label(right, text="해상도: …",
                                       fg="#7BA6DD", bg="#1a1a1a",
                                       font=("Malgun Gothic", 9, "bold"))
        self.cam_res_label.pack(anchor="w", padx=12, pady=(0, 2))
        self.root.after(800, self._poll_cam_resolution)

        # 카메라 ON / OFF 토글 버튼
        self.btn_cam = tk.Button(
            right, text="카메라 끄기", command=self._toggle_camera,
            bg="#333", fg="white", activebackground="#444",
            font=("Malgun Gothic", 10), relief="flat", bd=0,
            cursor="hand2", height=1,
        )
        self.btn_cam.pack(fill="x", padx=12, pady=(0, 8))

        self._opt_label(right, "프린터")
        self.var_printer = tk.StringVar(value=self.printer.printer_name or "")
        printers = [""] + list_windows_printers()
        p_combo = ttk.Combobox(right, textvariable=self.var_printer, state="readonly",
                               values=printers)
        p_combo.pack(fill="x", padx=12, pady=(0, 8))
        p_combo.bind("<<ComboboxSelected>>", self._on_change_printer)
        tk.Label(right, text="(빈칸 = 윈도우 기본 프린터)",
                 fg="#888", bg="#1a1a1a", font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        self._opt_label(right, "상단 제목 (선택)", pady=(14, 0))
        self.var_title = tk.StringVar(value=self.settings.get("title", ""))
        tk.Entry(right, textvariable=self.var_title, font=("Malgun Gothic", 11)).pack(
            fill="x", padx=12, pady=(0, 8))

        self.var_ts = tk.BooleanVar(value=bool(self.settings.get("timestamp", True)))
        tk.Checkbutton(right, text="하단 날짜·시간 인쇄",
                       variable=self.var_ts, fg="white", bg="#1a1a1a",
                       activebackground="#1a1a1a", activeforeground="white",
                       selectcolor="#1a1a1a",
                       font=("Malgun Gothic", 10)).pack(anchor="w", padx=12)

        # 프린터 거꾸로 설치 시 180도 회전
        self.var_rotate = tk.BooleanVar(value=bool(self.settings.get("rotate_180", True)))
        tk.Checkbutton(right, text="180° 회전 인쇄 (프린터 거꾸로 설치)",
                       variable=self.var_rotate, fg="white", bg="#1a1a1a",
                       activebackground="#1a1a1a", activeforeground="white",
                       selectcolor="#1a1a1a",
                       font=("Malgun Gothic", 10),
                       command=lambda: self._save_rotate()).pack(anchor="w", padx=12)

        # 사진 비율 = 정사각형 고정 (UI 옵션 없음)
        self.var_aspect = tk.DoubleVar(value=1.0)

        # 자동 무인 인쇄 모드 (촬영모드에서 얼굴 감지 → 자동 카운트다운 → 인쇄)
        self.var_auto = tk.BooleanVar(value=bool(self.settings.get("auto_print", False)))
        tk.Checkbutton(right, text="자동 무인 인쇄 (얼굴 감지 → 카운트다운)",
                       variable=self.var_auto, fg="white", bg="#1a1a1a",
                       activebackground="#1a1a1a", activeforeground="white",
                       selectcolor="#1a1a1a",
                       font=("Malgun Gothic", 10, "bold"),
                       command=lambda: self._save_auto_mode()).pack(anchor="w", padx=12, pady=(12, 0))
        tk.Label(right, text="F12 촬영모드에서만 작동 · 직원 대신 자동 인쇄",
                 fg="#888", bg="#1a1a1a",
                 font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        # 인쇄 모드 선택 — 풀버전(이벤트 포함) / 사진만
        self._opt_label(right, "인쇄 모드", pady=(16, 2))
        self.var_print_mode = tk.StringVar(
            value=self.settings.get("print_mode", "full"))
        mode_frame = tk.Frame(right, bg="#1a1a1a")
        mode_frame.pack(fill="x", padx=12)
        for label, val in [("이벤트 포함 (풀버전)", "full"), ("사진만", "photo")]:
            tk.Radiobutton(mode_frame, text=label, value=val,
                           variable=self.var_print_mode,
                           fg="white", bg="#1a1a1a", activebackground="#1a1a1a",
                           activeforeground="white", selectcolor="#333",
                           font=("Malgun Gothic", 10),
                           command=self._save_print_mode).pack(anchor="w")

        # 큰 인쇄 버튼 — 선택된 모드로 인쇄
        self.btn = tk.Button(
            right, text="사진 찍어서 인쇄", command=self._on_print_selected,
            bg="#7BA6DD", fg="white", activebackground="#5a8fcd",
            font=("Malgun Gothic", 16, "bold"),
            relief="flat", bd=0, height=2, cursor="hand2",
        )
        self.btn.pack(fill="x", padx=12, pady=(14, 6))

        tk.Label(right, text="Enter / Space  =  위에서 선택한 모드로 인쇄",
                 fg="#888", bg="#1a1a1a",
                 font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        # 키오스크(전체화면) 토글
        tk.Button(
            right, text="전체화면 ON/OFF  (F11)",
            command=self._toggle_kiosk,
            bg="#444", fg="white", activebackground="#555",
            font=("Malgun Gothic", 10), relief="flat", bd=0,
            cursor="hand2", height=1,
        ).pack(fill="x", padx=12, pady=(16, 2))

        # 촬영모드 — 카메라 영상만 풀스크린
        tk.Button(
            right, text="촬영모드 ON  (F12)",
            command=self._toggle_camera_only,
            bg="#7BA6DD", fg="white", activebackground="#5a8fcd",
            font=("Malgun Gothic", 12, "bold"), relief="flat", bd=0,
            cursor="hand2", height=2,
        ).pack(fill="x", padx=12, pady=(8, 2))
        tk.Label(right, text="카메라 영상만 풀스크린 · ESC로 종료",
                 fg="#888", bg="#1a1a1a",
                 font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        # 사진 파일 불러와서 인쇄 (외부 고화질 사진 인쇄용)
        tk.Button(
            right, text="사진 파일 불러와서 인쇄",
            command=self._on_print_from_file,
            bg="#7BA6DD", fg="white", activebackground="#5a8fcd",
            font=("Malgun Gothic", 11, "bold"), relief="flat", bd=0,
            cursor="hand2", height=2,
        ).pack(fill="x", padx=12, pady=(16, 2))
        tk.Label(right, text="폰으로 찍은 사진을 PC로 옮긴 후 직접 인쇄",
                 fg="#888", bg="#1a1a1a",
                 font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        # 저장된 인스타용 사진 폴더 열기
        tk.Button(
            right, text="저장된 사진 폴더 열기",
            command=self._open_photo_folder,
            bg="#222", fg="white", activebackground="#333",
            font=("Malgun Gothic", 10), relief="flat", bd=0,
            cursor="hand2", height=1,
        ).pack(fill="x", padx=12, pady=(10, 4))
        tk.Label(right, text="(인쇄한 사진이 자동 저장됩니다)",
                 fg="#888", bg="#1a1a1a",
                 font=("Malgun Gothic", 9)).pack(anchor="w", padx=12)

        # 키보드 단축키 — Enter/Space = 선택된 인쇄 모드로 인쇄
        self.root.bind("<Return>", lambda e: self._on_print_selected())
        self.root.bind("<space>", lambda e: self._on_print_selected())
        self.root.bind("<F11>", lambda e: self._toggle_kiosk())
        self.root.bind("<F12>", lambda e: self._toggle_camera_only())
        self.root.bind("<Escape>", lambda e: self._exit_kiosk())

    def _opt_label(self, parent, text, pady=(8, 4)):
        tk.Label(parent, text=text, fg="#bbb", bg="#1a1a1a",
                 font=("Malgun Gothic", 10)).pack(anchor="w", padx=12, pady=pady)

    def _on_change_camera(self, _evt=None):
        try:
            idx = int(self.var_cam.get())
        except ValueError:
            return
        self.settings["camera_index"] = idx
        _save_settings(self.settings)
        # 카메라가 켜져 있을 때만 즉시 전환. OFF면 인덱스만 저장
        if self.camera.is_on:
            threading.Thread(target=lambda: self.camera.change(idx), daemon=True).start()
            self._set_status(f"카메라 {idx} 로 전환…")
        else:
            self.camera.index = idx
            self._set_status(f"카메라 {idx} 선택 (꺼진 상태)")

    def _refresh_camera_list(self):
        """카메라 목록 다시 검색 — 폰 가상 캠(Iriun/DroidCam) 새로 연결 시 사용"""
        cams = _list_camera_indexes()
        self.cam_combo["values"] = [str(i) for i in cams]
        self._set_status(f"카메라 목록 새로고침 — {len(cams)}대 감지: {cams}")

    def _poll_cam_resolution(self):
        """카메라 실제 해상도를 라벨에 표시 (어떤 카메라인지 구분에 도움)"""
        try:
            cap = self.camera._cap
            if cap is not None and cap.isOpened():
                w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.cam_res_label.config(text=f"해상도: {w}×{h}")
            else:
                self.cam_res_label.config(text="해상도: 카메라 OFF")
        except Exception:
            pass
        self.root.after(1500, self._poll_cam_resolution)

    def _toggle_camera(self):
        if self.camera.is_on:
            self.camera.stop()
            self.btn_cam.config(text="카메라 켜기", bg="#7BA6DD")
            self._set_status("카메라 OFF — 다른 앱에서 사용 가능")
        else:
            try:
                idx = int(self.var_cam.get())
            except ValueError:
                idx = self.camera.index
            self.camera.index = idx
            self.camera.start()
            self.btn_cam.config(text="카메라 끄기", bg="#333")
            self._set_status(f"카메라 ON ({idx})")

    def _on_change_printer(self, _evt=None):
        name = self.var_printer.get()
        self.printer.set_options(printer_name=name)
        self.settings["printer_name"] = name
        _save_settings(self.settings)
        self._set_status(f"프린터: {name or '(기본)'}")

    def _set_status(self, msg: str):
        self.status.config(text=msg)

    def _show_off_screen(self):
        if getattr(self, "_off_shown", False):
            return
        off = Image.new("RGB", (PRINT_DOTS, 380), "#1a1a1a")
        draw = ImageDraw.Draw(off)
        f1 = _find_korean_font(54, bold=True)
        f2 = _find_korean_font(20)
        msg1 = "CAMERA OFF"
        msg2 = "오른쪽 [카메라 켜기] 버튼을 누르세요"
        tw = draw.textlength(msg1, font=f1)
        draw.text(((PRINT_DOTS - tw) // 2, 150), msg1, fill="#7BA6DD", font=f1)
        tw = draw.textlength(msg2, font=f2)
        draw.text(((PRINT_DOTS - tw) // 2, 230), msg2, fill="#aaaaaa", font=f2)
        self._photo = ImageTk.PhotoImage(off)
        self.preview.config(image=self._photo)
        self._off_shown = True

    def _ensure_event_preview_cache(self):
        """이벤트 박스는 옵션 영향 X — 한 번만 만들고 미리보기에서 재사용."""
        if getattr(self, "_event_preview", None) is None:
            placeholder = "KVR" + datetime.now().strftime("%y%m%d%H%M%S") + "000"
            self._event_preview = build_event_box(PRINT_DOTS, placeholder)

    def _refresh_loop(self):
        if not self.camera.is_on:
            self._show_off_screen()
            self.root.after(200, self._refresh_loop)
            return
        self._off_shown = False
        frame = self.camera.grab()

        # 촬영모드 — 좌측 카메라(정사각) + 우측 BIG 안내/카운트다운
        # 분할 레이아웃: 손님은 좌측 카메라에 시선, 우측에 행동 유도
        if self._camera_only and frame is not None:
            pil = cv_to_pil(frame)
            pil = crop_to_print_aspect(pil, 1.0)
            sw = max(640, self.root.winfo_width())
            sh = max(480, self.root.winfo_height())

            # 좌측 카메라 + 우측 안내 — 작은 모니터(17인치, 1920x1080)도 고려
            # 좌측 영역 = 화면 폭의 48%, 우측 = 52% (안내 더 잘 보이게 우측 더 큼)
            left_zone_w = int(sw * 0.48)
            # 카메라 정사각 — 좌측 영역 또는 화면 높이 - 100 중 작은 쪽
            cam_size = min(left_zone_w - 60, sh - 100)
            cam_size = max(280, cam_size)
            pil_disp = pil.resize((cam_size, cam_size), Image.LANCZOS)
            nw = nh = cam_size

            cx = (left_zone_w - nw) // 2
            cy = (sh - nh) // 2

            canvas = Image.new("RGB", (sw, sh), "black")
            canvas.paste(pil_disp, (cx, cy))
            d = ImageDraw.Draw(canvas)

            # 우측 영역 좌표 (안내가 시각적으로 더 강조)
            right_x = left_zone_w + 20
            right_w = sw - right_x - 30
            right_cx = right_x + right_w // 2

            # 카메라 외곽 가는 흰색 라인 + 코너 마크
            d.rectangle([(cx - 2, cy - 2), (cx + nw + 1, cy + nh + 1)],
                        outline="white", width=2)
            corner_len, corner_w, off = 42, 3, 16
            x1, y1 = cx - off, cy - off
            x2, y2 = cx + nw + off, cy + nh + off
            for (sx, sy, dx, dy) in [
                (x1, y1, 1, 1), (x2, y1, -1, 1),
                (x1, y2, 1, -1), (x2, y2, -1, -1),
            ]:
                d.line([(sx, sy), (sx + corner_len * dx, sy)],
                       fill="white", width=corner_w)
                d.line([(sx, sy), (sx, sy + corner_len * dy)],
                       fill="white", width=corner_w)

            # 좌측 카메라 위 — STAGE 라벨
            f_stage = _find_korean_font(20, bold=True)
            d.text((cx, max(20, cy - 38)), "STAGE 01  ·  LIVE",
                   fill="#7BA6DD", font=f_stage)

            # 좌측 카메라 아래 — 위치
            f_loc = _find_korean_font(16, bold=True)
            now = datetime.now().strftime("%Y.%m.%d  %H:%M")
            loc = f"N SEOUL TOWER  ·  {now}"
            tw = _safe_textlength(d, loc, f_loc)
            d.text((cx + (nw - tw) // 2, cy + nh + 14),
                   loc, fill="#888", font=f_loc)

            # ── 우측 영역: 브랜드 + 카운트다운/안내 + 이벤트 ──
            # 1) 우측 상단: 브랜드 (자간 강조)
            f_brand = _find_korean_font(22, bold=True)
            brand = "K · CULTURE · VR · TOWER"
            tw = _safe_textlength(d, brand, f_brand)
            while tw > right_w - 20 and f_brand.size > 14:
                f_brand = _find_korean_font(f_brand.size - 1, bold=True)
                tw = _safe_textlength(d, brand, f_brand)
            d.text((right_cx - tw // 2, 40), brand, fill="white", font=f_brand)

            # ON AIR (깜빡임) — 브랜드 옆
            self._blink = (getattr(self, "_blink", 0) + 1) % 30
            if self._blink < 22:
                bd_x = right_cx - tw // 2 - 70
                d.ellipse([(bd_x, 46), (bd_x + 16, 62)], fill="#ff5555")
                d.text((bd_x + 22, 42), "ON AIR",
                       fill="white", font=_find_korean_font(16, bold=True))

            # 2) 우측 가운데: BIG 카운트다운 또는 메인 안내
            self._draw_right_main(d, right_cx, right_w, sh)

            # 3) 우측 하단: 이벤트 안내 (영문)
            f_evt_title = _find_korean_font(22, bold=True)
            f_evt = _find_korean_font(18, bold=True)
            f_evt_small = _find_korean_font(15)
            evt_y = sh - 260
            d.text((right_cx - _safe_textlength(d, "EVENT", f_evt_title) // 2, evt_y),
                   "EVENT", fill="#7BA6DD", font=f_evt_title)
            d.line([(right_x + 20, evt_y + 32),
                    (right_x + right_w - 20, evt_y + 32)],
                   fill="#7BA6DD", width=1)
            for i, (line, font, color) in enumerate([
                ("Post on Instagram with", f_evt_small, "#aaa"),
                ("#tma   #kvrtower", f_evt, "white"),
                ("→ Free TMA Highlight VR", f_evt_small, "#aaa"),
                ("→ Win VIP tickets to TMA 2026", f_evt_small, "#aaa"),
            ]):
                tw = _safe_textlength(d, line, font)
                d.text((right_cx - tw // 2, evt_y + 50 + i * 30),
                       line, fill=color, font=font)

            # 자동 모드 트리거만 (안내는 이미 우측에 그림)
            self._auto_step(frame)

            self._photo = ImageTk.PhotoImage(canvas)
            self.preview.config(image=self._photo)
            self.root.after(50, self._refresh_loop)
            return

        if frame is not None:
            pil = cv_to_pil(frame)
            cam = crop_to_print_aspect(pil, 1.0)  # 정사각 고정
            w, h = cam.size
            s = PRINT_DOTS / w
            cam = cam.resize((PRINT_DOTS, max(1, int(h * s))), Image.LANCZOS)

            # 인쇄 결과와 동일한 합성 (헤더 + 화보 사진 + 푸터 + 이벤트 박스)
            magazine = wrap_photo_as_magazine(cam, PRINT_DOTS)
            header = build_kpop_header(PRINT_DOTS, self.var_title.get().strip())
            footer = build_kpop_footer(PRINT_DOTS, bool(self.var_ts.get()))
            self._ensure_event_preview_cache()
            event = self._event_preview

            gap = 14
            total_h = (header.size[1] + magazine.size[1] + footer.size[1]
                       + gap + event.size[1])
            canvas = Image.new("RGB", (PRINT_DOTS, total_h), "white")
            y = 0
            for part in (header, magazine, footer):
                canvas.paste(part, (0, y))
                y += part.size[1]
            canvas.paste(event, (0, y + gap))

            # 좌측 영역 높이에 맞춰 다운스케일
            max_h = 900
            if canvas.size[1] > max_h:
                ds = max_h / canvas.size[1]
                canvas = canvas.resize(
                    (max(1, int(canvas.size[0] * ds)), max_h), Image.LANCZOS
                )
            self._photo = ImageTk.PhotoImage(canvas)
            self.preview.config(image=self._photo)
        self.root.after(80, self._refresh_loop)  # 합성 부담 완화

    def _save_print_mode(self):
        self.settings["print_mode"] = self.var_print_mode.get()
        _save_settings(self.settings)

    def _save_auto_mode(self):
        self.settings["auto_print"] = bool(self.var_auto.get())
        _save_settings(self.settings)
        # 자동 모드 켜졌을 때 상태 초기화
        self._auto_face_frames = 0
        self._auto_countdown = 0
        self._locked_face = None

    def _draw_right_main(self, d, right_cx, right_w, sh):
        """우측 영역 중앙: BIG 카운트다운 / 안내.

        레이아웃 원칙: 각 텍스트 블록의 (y, y+size) 영역이 절대 안 겹치게.
        sh 기준 동적 사이즈 + 검증된 gap 으로 작은 모니터에서도 안전.
        """
        big_max_w = right_w - 30
        med_max_w = right_w - 50

        def fit_big(text, max_w, start, min_sz=18, bold=True):
            f = _find_korean_font(start, bold=bold)
            tw = _safe_textlength(d, text, f)
            while tw > max_w and f.size > min_sz:
                f = _find_korean_font(f.size - 3, bold=bold)
                tw = _safe_textlength(d, text, f)
            return f, tw

        # 사용 가능한 우측 영역 세로 범위
        top_pad = max(120, int(sh * 0.13))   # 위쪽 (상단 브랜드 영역 피함)
        bot_pad = max(280, int(sh * 0.26))   # 아래쪽 (이벤트 영역 피함)
        usable_h = sh - top_pad - bot_pad

        # 1) 인쇄 직후 (2초만 PRINTING 표시)
        if time.time() - self._auto_last_print < 2.0:
            t1 = "PRINTING…"
            t2 = "인쇄 중입니다"
            f1, tw1 = fit_big(t1, big_max_w, int(usable_h * 0.5))
            f2, tw2 = fit_big(t2, med_max_w, int(usable_h * 0.18))
            total = f1.size + 40 + f2.size
            y = top_pad + (usable_h - total) // 2
            d.text((right_cx - tw1 // 2, y), t1, fill="#7BA6DD", font=f1)
            d.text((right_cx - tw2 // 2, y + f1.size + 40),
                   t2, fill="white", font=f2)
            return

        # 2) 카운트다운 — 숫자 BIG + SMILE + 한글
        if self._auto_countdown > 0:
            num = str(self._auto_countdown)
            # 숫자: usable_h의 50%
            f_n, tw_n = fit_big(num, big_max_w, int(usable_h * 0.5))
            t_smile = "S M I L E !"
            f_s, tw_s = fit_big(t_smile, big_max_w, int(usable_h * 0.16))
            t_kr = "웃어주세요!"
            f_kr, tw_kr = fit_big(t_kr, med_max_w, int(usable_h * 0.12))

            gap1 = 30
            gap2 = 20
            total = f_n.size + gap1 + f_s.size + gap2 + f_kr.size
            y = top_pad + (usable_h - total) // 2
            d.text((right_cx - tw_n // 2, y), num, fill="#7BA6DD", font=f_n)
            y += f_n.size + gap1
            d.text((right_cx - tw_s // 2, y), t_smile, fill="white", font=f_s)
            y += f_s.size + gap2
            d.text((right_cx - tw_kr // 2, y), t_kr, fill="#aaa", font=f_kr)
            return

        # 2.5) 호객 단계 — 사람은 보이는데 아직 인쇄 단계 아님
        if (self.var_auto.get() and self._lead_in_active
                and self._auto_face_frames == 0):
            t1 = "STEP IN"
            t2 = "카메라 앞으로 와주세요"
            t3 = "FREE  ·  무료"
            f1, tw1 = fit_big(t1, big_max_w, int(usable_h * 0.32))
            f2, tw2 = fit_big(t2, med_max_w, int(usable_h * 0.15))
            f3, tw3 = fit_big(t3, med_max_w, int(usable_h * 0.13))
            gap = 26
            total = f1.size + gap + f2.size + gap + f3.size
            y = top_pad + (usable_h - total) // 2
            d.text((right_cx - tw1 // 2, y), t1, fill="white", font=f1)
            y += f1.size + gap
            d.text((right_cx - tw2 // 2, y), t2, fill="#7BA6DD", font=f2)
            y += f2.size + gap
            d.text((right_cx - tw3 // 2, y), t3, fill="#4040C8", font=f3)
            return

        # 3) 얼굴 감지 중 (자동 모드) — 게이지 바 시각화
        if self.var_auto.get() and self._auto_face_frames > 0:
            t1 = "STAY STILL"
            t2 = "그대로 있어주세요"
            pct = int(min(100, self._auto_face_frames / 8 * 100))
            t3 = f"{pct}%"

            f1, tw1 = fit_big(t1, big_max_w, int(usable_h * 0.28))
            f2, tw2 = fit_big(t2, med_max_w, int(usable_h * 0.14))
            f3, tw3 = fit_big(t3, med_max_w, int(usable_h * 0.10))

            gap = 22
            bar_h = max(18, int(usable_h * 0.05))
            bar_w = right_w - 60
            total = f1.size + gap + f2.size + gap + bar_h + 14 + f3.size
            y = top_pad + (usable_h - total) // 2

            d.text((right_cx - tw1 // 2, y), t1, fill="#7BA6DD", font=f1)
            y += f1.size + gap
            d.text((right_cx - tw2 // 2, y), t2, fill="white", font=f2)
            y += f2.size + gap

            # 게이지 바 — 둥근 배경 + 채워지는 파스텔 블루
            bar_x = right_cx - bar_w // 2
            radius = bar_h // 2
            try:
                d.rounded_rectangle(
                    (bar_x, y, bar_x + bar_w, y + bar_h),
                    radius=radius, fill="#2a2a2a", outline="#444", width=2,
                )
                fill_w = int(bar_w * pct / 100)
                if fill_w > radius * 2:
                    d.rounded_rectangle(
                        (bar_x, y, bar_x + fill_w, y + bar_h),
                        radius=radius, fill="#7BA6DD",
                    )
            except Exception:
                # rounded_rectangle 없는 Pillow 구버전 폴백
                d.rectangle(
                    (bar_x, y, bar_x + bar_w, y + bar_h), fill="#2a2a2a")
                fill_w = int(bar_w * pct / 100)
                d.rectangle(
                    (bar_x, y, bar_x + fill_w, y + bar_h), fill="#7BA6DD")
            y += bar_h + 14

            d.text((right_cx - tw3 // 2, y), t3, fill="white", font=f3)
            return

        # 4) 기본 안내 (영어 크게 / 한글 작게) — 무료 강조
        if self.var_auto.get():
            big_eng = "FREE K-POP PHOTO"
            small_eng = "Step in & smile!"
            kor1 = "무료 K-POP 사진"
            kor2 = "앞으로 와서 웃어주세요!"
        else:
            big_eng = "BE A K-POP STAR"
            small_eng = "FREE INSTANT PHOTO"
            kor1 = "K-POP 스타가 되어보세요"
            kor2 = "무료 즉석 사진 · SPACE 또는 ENTER"

        f1, tw1 = fit_big(big_eng, big_max_w, int(usable_h * 0.22))
        f2, tw2 = fit_big(small_eng, big_max_w, int(usable_h * 0.22))
        f3, tw3 = fit_big(kor1, med_max_w, int(usable_h * 0.11))
        f4, tw4 = fit_big(kor2, med_max_w, int(usable_h * 0.09))

        gap_eng = 16
        gap_kr = 28
        gap_kr2 = 12
        total = f1.size + gap_eng + f2.size + gap_kr + f3.size + gap_kr2 + f4.size
        y = top_pad + (usable_h - total) // 2
        d.text((right_cx - tw1 // 2, y), big_eng, fill="white", font=f1)
        y += f1.size + gap_eng
        d.text((right_cx - tw2 // 2, y), small_eng, fill="white", font=f2)
        y += f2.size + gap_kr
        d.text((right_cx - tw3 // 2, y), kor1, fill="#7BA6DD", font=f3)
        y += f3.size + gap_kr2
        d.text((right_cx - tw4 // 2, y), kor2, fill="#888", font=f4)

    def _draw_auto_overlay(self, d, sw, sh):
        """자동 모드 안내·카운트다운을 화면 상단에 표시 (인쇄 영역과 안 겹치게)"""
        # 상단 영역 사용 — 인쇄 결과의 하단 푸터/이벤트와 분리
        top_y = 80

        if time.time() - self._auto_last_print < 15.0:
            # 인쇄 중/직후 — 대형 메시지 가운데 표시
            f1 = _find_korean_font(90, bold=True)
            f2 = _find_korean_font(32, bold=True)
            t1 = "PRINTING…"
            t2 = "잠시만 기다려 주세요"
            tw = _safe_textlength(d, t1, f1)
            d.text(((sw - tw) // 2, sh // 2 - 70), t1, fill="#7BA6DD", font=f1)
            tw = _safe_textlength(d, t2, f2)
            d.text(((sw - tw) // 2, sh // 2 + 50), t2, fill="white", font=f2)
            return

        if self._auto_countdown > 0:
            # 카운트다운 큰 숫자 — 화면 가운데 (덮어쓰기)
            num = str(self._auto_countdown)
            f_big = _find_korean_font(280, bold=True)
            tw = _safe_textlength(d, num, f_big)
            d.text(((sw - tw) // 2, sh // 2 - 200), num, fill="#7BA6DD", font=f_big)
            f_msg = _find_korean_font(56, bold=True)
            msg = "S M I L E !"
            tw = _safe_textlength(d, msg, f_msg)
            d.text(((sw - tw) // 2, sh // 2 + 130), msg, fill="white", font=f_msg)
        elif self._auto_face_frames > 0:
            # 얼굴 감지됨 — 상단 안내
            f = _find_korean_font(36, bold=True)
            msg = f"Stay still…   가만히  ({self._auto_face_frames}/25)"
            tw = _safe_textlength(d, msg, f)
            d.text(((sw - tw) // 2, top_y), msg, fill="#7BA6DD", font=f)
        else:
            # 얼굴 없음 — 상단 손님 유치 안내
            f = _find_korean_font(42, bold=True)
            msg = "STEP IN FRONT TO TAKE A PHOTO"
            tw = _safe_textlength(d, msg, f)
            d.text(((sw - tw) // 2, top_y), msg, fill="white", font=f)
            f2 = _find_korean_font(22, bold=True)
            msg2 = "카메라 앞에 서면 자동으로 사진이 인쇄됩니다"
            tw = _safe_textlength(d, msg2, f2)
            d.text(((sw - tw) // 2, top_y + 60), msg2, fill="#aaaaaa", font=f2)

    def _auto_step(self, frame):
        """촬영모드 + 자동 모드에서 매 프레임마다 호출.
        얼굴 감지 → 안정적이면 카운트다운 → 0초가 되면 자동 인쇄.

        락온(lock-on):
          카운트다운 시작 시 그 얼굴의 위치를 잠금. 카운트다운 중에는 락 위치에
          가장 가까운 얼굴만 같은 사람으로 인정. 뒤로 다른 사람이 지나가도
          그 얼굴들은 무시. 락된 사람이 6프레임 연속 사라져야만 취소.
        """
        if not self.var_auto.get() or not self._camera_only:
            self._auto_face_frames = 0
            self._auto_countdown = 0
            self._locked_face = None
            self._lead_in_active = False
            return

        # 쿨다운 (인쇄 후 10초 동안은 새 자동 트리거 안 받음)
        if time.time() - self._auto_last_print < 10.0:
            self._auto_face_frames = 0
            self._auto_countdown = 0
            self._auto_last_countdown_speak = -1
            self._auto_no_face_count = 0
            self._locked_face = None
            self._lead_in_active = False
            return

        # ─── 움직임 감지 (호객 음성 트리거) ───
        # 다운샘플(160×90) gray + 이전 프레임 비교 → 변화 픽셀 비율
        motion_detected = False
        try:
            small = cv2.resize(frame, (160, 90))
            mgray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            mgray = cv2.GaussianBlur(mgray, (5, 5), 0)
            if (self._prev_motion_gray is not None
                    and self._prev_motion_gray.shape == mgray.shape):
                delta = cv2.absdiff(self._prev_motion_gray, mgray)
                _, thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)
                pixels = int(cv2.countNonZero(thresh))
                # 화면의 1.5% 이상 변화 = 사람 움직임으로 간주
                motion_detected = pixels > int(mgray.size * 0.015)
            self._prev_motion_gray = mgray
        except Exception as e:
            log.debug("움직임 감지 실패: %s", e)

        # ─── 모든 얼굴 검출 (튜플 7번째 = score) ───
        rgb = None
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        except Exception:
            pass

        all_faces = _detect_faces_all(rgb) if rgb is not None else []

        # Fallback: Detection 실패 시 Mesh → Haar (단일 얼굴, score=0.7 추정)
        if not all_faces and rgb is not None:
            try:
                parts = _detect_face_parts(rgb)
                if parts is not None and parts.get("oval") is not None:
                    mask = parts["oval"]
                    ys, xs = np.where(mask > 0.3)
                    if len(xs) > 0:
                        img_h, img_w = mask.shape
                        all_faces = [(
                            (int(xs.min()) + int(xs.max())) / 2,
                            (int(ys.min()) + int(ys.max())) / 2,
                            int(xs.max()) - int(xs.min()),
                            int(ys.max()) - int(ys.min()),
                            img_w, img_h, 0.7,
                        )]
            except Exception:
                pass

        if not all_faces:
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                cascade = cv2.CascadeClassifier(cascade_path)
                if not cascade.empty():
                    faces = cascade.detectMultiScale(
                        gray, scaleFactor=1.07, minNeighbors=4,
                        minSize=(60, 60),
                    )
                    if len(faces) > 0:
                        ih, iw = gray.shape
                        all_faces = sorted(
                            [(x + w/2, y + h/2, w, h, iw, ih, 0.7)
                             for (x, y, w, h) in faces],
                            key=lambda f: f[2] * f[3], reverse=True,
                        )
            except Exception:
                pass

        # ─── 단계 1: 호객 후보 — 약한 검출이라도 화면 안에 있으면 사람 있음 ───
        anyone_present = len(all_faces) > 0

        # ─── 단계 2: 인쇄 가능 후보 — 강한 검출(score >= 0.6) + 위치/크기 통과 ───
        #     1) score >= 0.6 (확실한 정면 얼굴)
        #     2) 얼굴 폭 6% 이상 (충분히 가까이)
        #     3) 화면 좌우 50% 이내
        #     4) 화면 상단 5% ~ 65% (바닥조명·바닥 노이즈 무시)
        strong_faces = []
        for f in all_faces:
            fcx, fcy, fw, fh, iw, ih, score = f
            if score < 0.6:
                continue
            face_ratio = fw / iw
            fcy_ratio = fcy / ih
            if (face_ratio >= 0.06 and
                    abs(fcx - iw / 2) < iw * 0.50 and
                    0.05 < fcy_ratio < 0.65):
                strong_faces.append(f)

        # ─── 락온 모드: 카운트다운 진행 중이면 락 위치 근처 얼굴만 선택 ───
        chosen = None
        if self._locked_face is not None and strong_faces:
            lcx, lcy, lw, lh = self._locked_face
            iw_ref = strong_faces[0][4]
            best = None
            best_dist = float("inf")
            for f in strong_faces:
                fcx, fcy, fw, fh, _, _, _ = f
                dist = ((fcx - lcx) ** 2 + (fcy - lcy) ** 2) ** 0.5
                size_ratio = min(fw, lw) / max(fw, lw)
                if dist < iw_ref * 0.30 and size_ratio > 0.5 and dist < best_dist:
                    best_dist = dist
                    best = f
            if best is not None:
                chosen = best
                fcx, fcy, fw, fh, _, _, _ = best
                self._locked_face = (
                    lcx * 0.7 + fcx * 0.3,
                    lcy * 0.7 + fcy * 0.3,
                    lw * 0.7 + fw * 0.3,
                    lh * 0.7 + fh * 0.3,
                )
        elif strong_faces:
            chosen = strong_faces[0]

        face_ok = chosen is not None

        # ─── 호객 단계 ───
        # 화면 표시: 사람(얼굴) 또는 움직임 감지되면 호객 화면
        # 음성 호객: 움직임 감지되었을 때만 (정적인 사진/포스터 무시)
        any_signal = anyone_present or motion_detected
        if any_signal and not face_ok and self._auto_countdown_start == 0:
            self._lead_in_active = True
            if motion_detected and time.time() - self._last_lead_in_time > 20:
                self._last_lead_in_time = time.time()
                speak(
                    "Hey! Free K-POP photo — step in front of the camera!",
                    "이리 오세요! 무료 K팝 사진! 카메라 앞에 서주세요!",
                )
        else:
            self._lead_in_active = False

        # ─── 카운트다운 진행 중 ───
        if self._auto_countdown_start > 0:
            if not face_ok:
                self._auto_no_face_count += 1
                # 약 0.3초(6프레임) 연속 락된 얼굴 사라지면 취소
                if self._auto_no_face_count >= 6:
                    log.info("카운트다운 중 락된 얼굴 사라짐 → 취소")
                    self._auto_countdown_start = 0
                    self._auto_countdown = 0
                    self._auto_face_frames = 0
                    self._auto_last_countdown_speak = -1
                    self._auto_no_face_count = 0
                    self._locked_face = None
                    speak("Aww! Come back, it's totally free!",
                          "아쉽다! 무료니까 다시 와주세요!")
                    return
            else:
                self._auto_no_face_count = 0

            # 카운트다운 진행
            elapsed = time.time() - self._auto_countdown_start
            new_cd = max(0, 3 - int(elapsed))
            self._auto_countdown = new_cd
            last = getattr(self, "_auto_last_countdown_speak", -1)
            if new_cd != last:
                self._auto_last_countdown_speak = new_cd
                if new_cd == 3:
                    speak("Three", "셋")
                elif new_cd == 2:
                    speak("Two", "둘")
                elif new_cd == 1:
                    speak("One", "하나")
                elif new_cd == 0:
                    speak("Smile!", "치즈!")
            if elapsed >= 4.0:
                self._auto_countdown_start = 0
                self._auto_last_countdown_speak = -1
                self._auto_no_face_count = 0
                self._auto_last_print = time.time()
                self._locked_face = None
                log.info("자동 인쇄 트리거")
                self.root.after(0, self._auto_print)
            return

        # ─── 카운트다운 시작 전 — 얼굴 안정성 누적 ───
        if face_ok:
            prev = self._auto_face_frames
            self._auto_face_frames += 1
            # 첫 인식 인사 — 30초 쿨다운 (반복 인사 방지)
            if prev == 0 and time.time() - self._last_greet_time > 30:
                speak("Yay! Free K-POP photo just for you!",
                      "와! K팝 무료 사진 찍어드릴게요!")
                self._last_greet_time = time.time()
            if self._auto_face_frames >= 8 and self._auto_countdown_start == 0:
                self._auto_countdown_start = time.time()
                self._auto_countdown = 3
                self._auto_last_countdown_speak = -1
                self._auto_no_face_count = 0
                # ★ 락온! — 이 얼굴 위치 잠금
                fcx, fcy, fw, fh, _, _, _ = chosen
                self._locked_face = (fcx, fcy, fw, fh)
                log.info("얼굴 안정 → 카운트다운 시작 (락온 위치 %d,%d)",
                         int(fcx), int(fcy))
        else:
            # 얼굴 사라지면 즉시 0 (서서히 줄이지 않음 — 처음부터 다시)
            self._auto_face_frames = 0
            # 손님 없을 때 30초마다 끌어들이는 음성 (무료 강조)
            if not hasattr(self, "_auto_last_call_time"):
                self._auto_last_call_time = time.time()
            if time.time() - self._auto_last_call_time > 30.0 and \
               time.time() - self._auto_last_print > 12.0:
                self._auto_last_call_time = time.time()
                # 4종 카피 순환 (매번 다른 멘트)
                if not hasattr(self, "_auto_call_idx"):
                    self._auto_call_idx = 0
                calls = [
                    ("Hey there! Free K-POP photo for you!",
                     "거기 누구나! K팝 무료 사진 찍어가세요!"),
                    ("Step in front, become a K-POP STAR!",
                     "카메라 앞으로! K팝 스타가 되어보세요!"),
                    ("Free instant photo! Don't miss out!",
                     "무료 즉석 사진! 놓치지 마세요!"),
                    ("Smile for free! Come over here!",
                     "무료 사진! 이쪽으로 와주세요!"),
                ]
                en, ko = calls[self._auto_call_idx % len(calls)]
                self._auto_call_idx += 1
                speak(en, ko)

    def _auto_print(self):
        """자동 모드에서 호출되는 인쇄 — 연사 잠금/메시지박스 우회"""
        if not self.camera.is_on:
            log.warning("자동 인쇄 실패 — 카메라 OFF")
            return
        # 연사 잠금 강제 해제 (자동 모드는 자체 쿨다운)
        self.printer._last_print = 0.0
        speak("Wow! Your free photo is printing now!",
              "와! 무료 사진 인쇄 중이에요!")
        self._on_print_selected()

    def _on_print_selected(self):
        """우측 '인쇄 모드' 라디오 선택에 따라 인쇄 (풀버전 / 사진만)"""
        with_event = (self.var_print_mode.get() == "full")
        self._do_print(with_event=with_event)

    def _on_print(self):
        self._do_print(with_event=True)

    def _on_print_photo_only(self):
        """이벤트 영역 제외, 사진(헤더+화보+푸터)만 인쇄 — 디스플레이용"""
        self._do_print(with_event=False)

    def _do_print(self, with_event: bool):
        if not self.camera.is_on:
            messagebox.showinfo(APP_TITLE, "카메라가 꺼져 있습니다. '카메라 켜기' 후 다시 시도하세요.")
            return
        if not self.printer.can_print_now():
            self._set_status("연사 잠금 중… 잠시 후 다시")
            return
        # 인쇄 시점에 5프레임 평균 캡처 — 노이즈 감소, 디테일 살림
        frame = self.camera.grab_avg(count=5, interval=0.04)
        if frame is None:
            messagebox.showwarning(APP_TITLE, "카메라 영상이 아직 없습니다.")
            return

        # 옵션 저장
        self.settings["title"] = self.var_title.get()
        self.settings["timestamp"] = bool(self.var_ts.get())
        self.settings["aspect"] = float(self.var_aspect.get())
        _save_settings(self.settings)

        pil = cv_to_pil(frame)

        # 인스타 업로드용 정사각 컬러 원본 저장 (디더링 전, enhance 전)
        event_code = generate_event_code()
        try:
            square_raw = crop_to_print_aspect(pil, 1.0)
            saved_path = save_photo_for_instagram(square_raw, event_code)
            log.info("사진 저장: %s", saved_path)
        except Exception as e:
            log.warning("사진 저장 실패: %s", e)
            saved_path = None

        image = build_print_image(
            pil,
            title=self.var_title.get().strip(),
            add_timestamp=bool(self.var_ts.get()),
            aspect_h_over_w=float(self.var_aspect.get()),
            event_code=event_code,
            with_event=with_event,
        )
        self._last_saved_path = saved_path

        self.btn.config(state="disabled", text="인쇄 중…")
        self._set_status(
            "사진만 인쇄 전송 중…" if not with_event else "인쇄 전송 중…"
        )

        rotate = bool(self.var_rotate.get())

        def work():
            ok = self.printer.print_image(image, rotate_180=rotate)
            self.root.after(0, lambda: self._on_print_done(ok))

        threading.Thread(target=work, daemon=True).start()

    def _on_print_done(self, ok: bool):
        self.btn.config(state="normal", text="사진 찍어서 인쇄")
        if ok:
            saved = getattr(self, "_last_saved_path", None)
            if saved:
                self._set_status(f"인쇄·저장 완료 — {saved.name}")
            else:
                self._set_status(f"인쇄 완료 — {datetime.now().strftime('%H:%M:%S')}")
        else:
            self._set_status("인쇄 실패 — 프린터 연결 확인")
            messagebox.showerror(APP_TITLE,
                                 "인쇄 실패. 프린터 전원/케이블/드라이버를 확인하세요.")

    def _toggle_kiosk(self):
        """키오스크(전체화면 + 우측 패널 숨김) 모드 토글"""
        self._kiosk = not self._kiosk
        try:
            self.root.attributes("-fullscreen", self._kiosk)
        except Exception:
            self.root.state("zoomed" if self._kiosk else "normal")
        if self._kiosk:
            self._right_panel.pack_forget()
            self._set_status("키오스크 ON — F11/ESC로 종료")
        else:
            self._right_panel.pack(side="right", fill="y", padx=(12, 0))
            self._set_status("키오스크 OFF")

    def _toggle_camera_only(self):
        """촬영모드 — 카메라 영상만 풀스크린.
        확장 디스플레이면 보조 모니터(TV)로 자동 이동.
        """
        self._camera_only = not self._camera_only
        if self._camera_only:
            # 보조 모니터(TV)로 이동
            self._move_to_external_monitor()
            try:
                self.root.attributes("-fullscreen", True)
            except Exception:
                self.root.state("zoomed")
            self._kiosk = True
            self._right_panel.pack_forget()
            self._top_bar.pack_forget()
        else:
            try:
                self.root.attributes("-fullscreen", False)
            except Exception:
                self.root.state("normal")
            self._top_bar.pack(fill="x", padx=16, pady=(14, 6),
                               before=self.preview.master.master)
            self._right_panel.pack(side="right", fill="y", padx=(12, 0))
            # 일반 모드 복귀 — 메인 모니터로
            self.root.geometry("1080x1020+50+50")
            self._kiosk = False

    def _move_to_external_monitor(self):
        """확장 디스플레이 모드에서 보조 모니터(TV)로 윈도우 이동.
        모니터 1대면 그대로.
        """
        try:
            from screeninfo import get_monitors
            monitors = get_monitors()
            if len(monitors) < 2:
                log.info("모니터 1대 — 보조 이동 생략")
                return
            target = next((m for m in monitors if not m.is_primary), monitors[-1])
            # 풀스크린 전에 위치 + 크기 지정
            self.root.attributes("-fullscreen", False)
            self.root.geometry(f"{target.width}x{target.height}+{target.x}+{target.y}")
            self.root.update_idletasks()
            log.info("보조 모니터 이동: %dx%d @ (%d,%d)",
                     target.width, target.height, target.x, target.y)
        except Exception as e:
            log.warning("보조 모니터 이동 실패: %s", e)

    def _exit_kiosk(self):
        """ESC — 촬영/키오스크 모두 종료"""
        if self._camera_only:
            self._toggle_camera_only()
        elif self._kiosk:
            self._toggle_kiosk()

    def _save_rotate(self):
        self.settings["rotate_180"] = bool(self.var_rotate.get())
        _save_settings(self.settings)

    def _save_preview(self):
        self.settings["preview_before_print"] = bool(self.var_preview.get())
        _save_settings(self.settings)

    def _on_print_from_file(self):
        """디스크의 사진 파일을 불러와서 인쇄 (외부 고화질 사진 인쇄용).
        폰으로 잘 찍은 사진을 PC로 옮긴 후 직접 인쇄 = 최고 화질.
        """
        from tkinter import filedialog
        path = filedialog.askopenfilename(
            title="인쇄할 사진 선택",
            filetypes=[
                ("이미지 파일", "*.jpg *.jpeg *.png *.bmp *.tiff *.webp"),
                ("모든 파일", "*.*"),
            ],
        )
        if not path:
            return
        try:
            pil = Image.open(path).convert("RGB")
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"이미지를 열 수 없습니다: {e}")
            return

        if not self.printer.can_print_now():
            self._set_status("연사 잠금 중… 잠시 후 다시")
            return

        event_code = generate_event_code()
        try:
            square_raw = crop_to_print_aspect(pil, 1.0)
            saved_path = save_photo_for_instagram(square_raw, event_code)
        except Exception as e:
            log.warning("사진 저장 실패: %s", e)
            saved_path = None

        # 사진만 인쇄 (이벤트 영역 없이) — 외부 파일은 보통 디스플레이용
        image = build_print_image(
            pil,
            title=self.var_title.get().strip(),
            add_timestamp=bool(self.var_ts.get()),
            aspect_h_over_w=1.0,
            event_code=event_code,
            with_event=False,
        )
        self._last_saved_path = saved_path

        self.btn.config(state="disabled", text="인쇄 중…")
        self._set_status(f"파일 인쇄 중… ({os.path.basename(path)})")
        rotate = bool(self.var_rotate.get())

        def work():
            ok = self.printer.print_image(image, rotate_180=rotate)
            self.root.after(0, lambda: self._on_print_done(ok))

        threading.Thread(target=work, daemon=True).start()

    def _show_print_preview_dialog(self, pil_image, raw_pil):
        """인쇄 미리보기 모달 — 결과 확인 후 [인쇄]/[다시 찍기] 선택.

        pil_image: 최종 인쇄 합성 결과 (RGB)
        raw_pil  : 원본 카메라 사진 (밝기/대비/선명도 미세 조정용)
        returns  : ('print', adjusted_image) | ('retry', None) | ('cancel', None)
        """
        import numpy as np
        from PIL import ImageEnhance as IE

        result = {"action": "cancel", "image": None}
        win = tk.Toplevel(self.root)
        win.title("인쇄 미리보기")
        win.transient(self.root)
        win.configure(bg="#111")
        win.grab_set()

        # 좌측: 이미지 미리보기 / 우측: 슬라이더
        body = tk.Frame(win, bg="#111")
        body.pack(fill="both", expand=True, padx=14, pady=14)

        img_label = tk.Label(body, bg="#000", bd=1, relief="solid")
        img_label.pack(side="left", fill="both", expand=True)

        side = tk.Frame(body, bg="#1a1a1a", width=220)
        side.pack(side="right", fill="y", padx=(12, 0))
        side.pack_propagate(False)

        def lbl(text):
            tk.Label(side, text=text, fg="#bbb", bg="#1a1a1a",
                     font=("Malgun Gothic", 10)).pack(anchor="w", padx=10, pady=(10, 2))

        v_bright = tk.DoubleVar(value=1.0)
        v_contrast = tk.DoubleVar(value=1.0)
        v_sharp = tk.DoubleVar(value=1.0)

        def render():
            # 슬라이더 값으로 raw 사진 재처리 → 합성
            adj = raw_pil
            if abs(v_bright.get() - 1.0) > 0.01:
                adj = IE.Brightness(adj).enhance(v_bright.get())
            if abs(v_contrast.get() - 1.0) > 0.01:
                adj = IE.Contrast(adj).enhance(v_contrast.get())
            if abs(v_sharp.get() - 1.0) > 0.01:
                adj = IE.Sharpness(adj).enhance(v_sharp.get())
            new_image = build_print_image(
                adj,
                title=self.var_title.get().strip(),
                add_timestamp=bool(self.var_ts.get()),
                aspect_h_over_w=1.0,
                event_code=getattr(self, "_pending_event_code", ""),
                with_event=self._pending_with_event,
            )
            result["image"] = new_image
            # 화면 크기에 맞춰 다운스케일
            sw = max(800, self.root.winfo_width() - 350)
            sh = max(600, self.root.winfo_height() - 200)
            iw, ih = new_image.size
            scale = min(sw / iw, sh / ih, 1.0)
            disp = new_image.resize((max(1, int(iw * scale)),
                                     max(1, int(ih * scale))), Image.LANCZOS)
            photo = ImageTk.PhotoImage(disp)
            img_label.config(image=photo)
            img_label.image = photo

        render()
        result["image"] = pil_image

        def slider(parent, var, frm, to, step):
            sc = tk.Scale(parent, from_=frm, to=to, resolution=step,
                          variable=var, orient="horizontal",
                          bg="#1a1a1a", fg="white",
                          troughcolor="#333", highlightthickness=0,
                          font=("Malgun Gothic", 9))
            sc.pack(fill="x", padx=10)
            sc.bind("<ButtonRelease-1>", lambda e: render())
            return sc

        lbl("밝기")
        slider(side, v_bright, 0.6, 1.6, 0.05)
        lbl("대비")
        slider(side, v_contrast, 0.6, 1.8, 0.05)
        lbl("선명도")
        slider(side, v_sharp, 0.5, 3.0, 0.1)

        # 버튼
        btn_frame = tk.Frame(win, bg="#111")
        btn_frame.pack(fill="x", padx=14, pady=(0, 14))

        def do_print():
            result["action"] = "print"
            win.destroy()

        def do_retry():
            result["action"] = "retry"
            win.destroy()

        def do_cancel():
            result["action"] = "cancel"
            win.destroy()

        tk.Button(btn_frame, text="취소 (Esc)", command=do_cancel,
                  bg="#333", fg="white", activebackground="#444",
                  font=("Malgun Gothic", 11), relief="flat", bd=0,
                  cursor="hand2", width=14, height=2).pack(side="left", padx=4)
        tk.Button(btn_frame, text="다시 찍기 (R)", command=do_retry,
                  bg="#555", fg="white", activebackground="#666",
                  font=("Malgun Gothic", 11), relief="flat", bd=0,
                  cursor="hand2", width=14, height=2).pack(side="left", padx=4)
        tk.Button(btn_frame, text="인쇄 (Enter)", command=do_print,
                  bg="#7BA6DD", fg="white", activebackground="#5a8fcd",
                  font=("Malgun Gothic", 12, "bold"), relief="flat", bd=0,
                  cursor="hand2", height=2).pack(side="right", fill="x",
                                                  expand=True, padx=4)

        win.bind("<Return>", lambda e: do_print())
        win.bind("<r>", lambda e: do_retry())
        win.bind("<R>", lambda e: do_retry())
        win.bind("<Escape>", lambda e: do_cancel())

        # 다이얼로그 사이즈 큰 화면에 맞춤
        win.geometry(f"{self.root.winfo_width() - 80}x{self.root.winfo_height() - 80}+40+40")
        win.focus_set()
        self.root.wait_window(win)
        return result["action"], result["image"]

    def _open_photo_folder(self):
        folder = get_photo_save_dir()
        try:
            if os.name == "nt":
                os.startfile(str(folder))  # 윈도우 탐색기
            else:
                import subprocess
                subprocess.Popen(["xdg-open", str(folder)])
        except Exception as e:
            messagebox.showerror(APP_TITLE, f"폴더 열기 실패: {e}\n경로: {folder}")

    def shutdown(self):
        self.camera.stop()


def main():
    root = tk.Tk()
    app = App(root)

    def on_close():
        app.shutdown()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
