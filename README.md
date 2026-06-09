# K-Culture VR Tower Photo Printer

매장 키오스크 — 카메라 영상을 영수증 프린터로 즉시 인쇄.

## 구조
- `src/main.py` — 메인 앱 (Tkinter GUI, OpenCV, Mediapipe)
- `src/printer.py` — 영수증 프린터 (윈도우 GDI / ESC/POS)
- `assets/` — QR 코드 등
- `launcher.py` — 자동 업데이트 런처 (서버 → 코드 다운로드 → 실행)
- `requirements.txt` — Python 의존성

## 매장 PC 배포
1. `build_launcher.bat` 으로 `KVRTowerPhoto.exe` 빌드
2. 매장 PC에 EXE 복사 + Python 한 번 설치 (install.bat)
3. EXE 실행 → 매번 서버에서 최신 코드 받아 실행
