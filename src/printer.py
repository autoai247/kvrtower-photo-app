"""영수증 프린터 출력 — 윈도우 GDI 모드 (가로줄 없음)

기본 모드: 윈도우 GDI 드라이버로 인쇄.
- 워드/한글 등 일반 프로그램 인쇄와 같은 경로
- 프린터 드라이버가 헤드 동기화를 알아서 처리 → 가로줄 없음
- 사진은 PIL ImageWin.Dib로 페이지에 그려서 출력

폴백 모드(ESC/POS RAW): 호환성 위해 유지.
"""

import logging
import threading
import time

logger = logging.getLogger(__name__)

PRINT_DOTS = 544       # 200dpi 80mm 영수증 — 인쇄 영역 약 68.6mm
# 200dpi: 80mm × 200/25.4 ≈ 630dot, 인쇄 가능 영역 72mm ≈ 567dot
# 좌우 약 5mm 마진 빼고 544dot = 사진 크게 + 잘림 안전


# ─────────── 윈도우 GDI 인쇄 (기본·권장) ───────────

def _print_image_gdi(printer_name: str, pil_image, dpi: int = 203,
                     paper_width_mm: float = 80.0):
    """윈도우 GDI 드라이버로 영수증 프린터에 사진 인쇄.

    HORZRES(인쇄 가능 가로 영역)를 기준으로 가운데 정렬.
    PHYSICALWIDTH(종이 전체 폭)를 쓰면 좌측 안 인쇄 마진 때문에 우측 쏠림이 생김.
    """
    import win32print
    import win32ui
    from PIL import ImageWin

    name = printer_name or win32print.GetDefaultPrinter()
    hDC = win32ui.CreateDC()
    hDC.CreatePrinterDC(name)
    try:
        hDC.StartDoc("Camera Receipt")
        hDC.StartPage()

        # StretchBlt 모드를 HALFTONE 으로 설정 — GDI 최고 품질 리샘플링
        # (COLORONCOLOR 기본보다 훨씬 부드럽고 디테일 보존)
        import ctypes
        HALFTONE = 4
        hdc_handle = hDC.GetHandleOutput()
        try:
            ctypes.windll.gdi32.SetStretchBltMode(hdc_handle, HALFTONE)
            ctypes.windll.gdi32.SetBrushOrgEx(hdc_handle, 0, 0, None)
        except Exception:
            pass

        # HORZRES = 인쇄 가능 가로 (마진 제외 후), GDI가 PHYSICALOFFSETX 자동 보정
        horz_res = hDC.GetDeviceCaps(8)
        if horz_res <= 0:
            horz_res = int(paper_width_mm / 25.4 * dpi)

        iw, ih = pil_image.size
        ratio = min(horz_res / iw, 1.0)
        nw, nh = max(1, int(iw * ratio)), max(1, int(ih * ratio))
        x = max(0, (horz_res - nw) // 2)
        y = 0

        dib = ImageWin.Dib(pil_image)
        dib.draw(hdc_handle, (x, y, x + nw, y + nh))

        hDC.EndPage()
        hDC.EndDoc()
    finally:
        hDC.DeleteDC()


# ─────────── ESC/POS RAW 인쇄 (폴백) ───────────

def _escpos_init() -> bytes:
    data = bytearray()
    data += b"\x1b\x40"                       # ESC @  초기화
    data += b"\x1d\x4c\x00\x00"               # GS L 좌마진 0
    data += bytes([0x1d, 0x57, PRINT_DOTS & 0xFF, (PRINT_DOTS >> 8) & 0xFF])
    data += b"\x1d\x28\x4b\x02\x00\x32\x02"   # GS ( K  인쇄 속도 낮춤
    return bytes(data)


def _image_to_escpos(pil_image, band_height: int = 2303) -> bytes:
    from PIL import ImageEnhance, ImageOps

    img = pil_image.convert("L")
    img = ImageOps.autocontrast(img, cutoff=1)
    img = ImageEnhance.Brightness(img).enhance(1.08)
    img = ImageEnhance.Contrast(img).enhance(1.25)
    img = ImageEnhance.Sharpness(img).enhance(1.8)
    img = img.convert("1")

    w, h = img.size
    byte_w = (w + 7) // 8
    pix = img.load()

    data = bytearray()
    y = 0
    while y < h:
        bh = min(band_height, h - y)
        data += b"\x1d\x76\x30\x00"
        data += bytes([byte_w & 0xFF, (byte_w >> 8) & 0xFF,
                       bh & 0xFF, (bh >> 8) & 0xFF])
        for row in range(y, y + bh):
            for cb in range(byte_w):
                byte = 0
                for bit in range(8):
                    x = cb * 8 + bit
                    if x < w and pix[x, row] == 0:
                        byte |= (0x80 >> bit)
                data.append(byte)
        y += bh
    return bytes(data)


def _escpos_tail() -> bytes:
    data = bytearray()
    data += b"\n\n\n\n\n\n"
    data += b"\x1d\x56\x00"
    return bytes(data)


def _send_raw_windows(printer_name: str, payload: bytes):
    import win32print
    name = printer_name or win32print.GetDefaultPrinter()
    h = win32print.OpenPrinter(name)
    try:
        win32print.StartDocPrinter(h, 1, ("Camera Snapshot", None, "RAW"))
        win32print.StartPagePrinter(h)
        win32print.WritePrinter(h, payload)
        win32print.EndPagePrinter(h)
        win32print.EndDocPrinter(h)
    finally:
        win32print.ClosePrinter(h)


# ─────────── PrinterManager ───────────

class PrinterManager:
    """기본 mode='gdi' (윈도우 드라이버, 가로줄 없음)
    필요 시 mode='escpos'로 변경 가능 (저수준 ESC/POS RAW)
    """

    def __init__(self, printer_name: str = "", interval: float = 1.0,
                 mode: str = "gdi"):
        self.printer_name = printer_name
        self.interval = interval
        self.mode = mode
        self._lock = threading.Lock()
        self._last_print = 0.0

    def set_options(self, printer_name=None, interval=None, mode=None):
        if printer_name is not None:
            self.printer_name = printer_name
        if interval is not None:
            self.interval = interval
        if mode is not None:
            self.mode = mode

    def can_print_now(self) -> bool:
        return (time.time() - self._last_print) >= self.interval

    def print_image(self, pil_image, rotate_180: bool = True) -> bool:
        """프린터를 거꾸로 설치한 경우 rotate_180=True 로 180도 회전 후 인쇄"""
        with self._lock:
            try:
                from PIL import Image as _Image
                img = (pil_image.transpose(_Image.ROTATE_180)
                       if rotate_180 else pil_image)
                if self.mode == "escpos":
                    payload = (_escpos_init() + _image_to_escpos(img)
                               + _escpos_tail())
                    _send_raw_windows(self.printer_name, payload)
                else:
                    _print_image_gdi(self.printer_name, img)
                self._last_print = time.time()
                logger.info("프린트 성공 (mode=%s, rotate=%s)",
                            self.mode, rotate_180)
                return True
            except Exception as e:
                logger.exception("프린트 실패: %s", e)
                return False


def list_windows_printers() -> list:
    try:
        import win32print
        flags = win32print.PRINTER_ENUM_LOCAL | win32print.PRINTER_ENUM_CONNECTIONS
        return [p[2] for p in win32print.EnumPrinters(flags)]
    except Exception:
        return []
