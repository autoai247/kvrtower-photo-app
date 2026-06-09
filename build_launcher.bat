@echo off
title launcher.exe 빌드
setlocal
cd /d "%~dp0"

echo.
echo ============================================================
echo  launcher.exe 빌드
echo  (이 파일은 매장 PC에 배포할 작은 실행파일을 만듭니다)
echo ============================================================
echo.

python --version >nul 2>&1
if errorlevel 1 (
    echo [오류] Python 이 필요합니다.
    pause
    exit /b 1
)

python -m pip install -q pyinstaller
if errorlevel 1 (
    echo [오류] pyinstaller 설치 실패
    pause
    exit /b 1
)

if exist build rmdir /s /q build
if exist launcher_dist rmdir /s /q launcher_dist
python -m PyInstaller --noconfirm --clean --onefile --console ^
    --name KVRTowerPhoto ^
    --distpath launcher_dist ^
    launcher.py

if not exist launcher_dist\KVRTowerPhoto.exe (
    echo [오류] 빌드 실패
    pause
    exit /b 1
)

echo.
echo ============================================================
echo  완료! launcher_dist\KVRTowerPhoto.exe 가 생성되었습니다.
echo.
echo  매장 PC 에 이 파일 하나만 복사해서 두면 됩니다.
echo  실행할 때마다 GitHub 에서 최신 코드를 받아 실행합니다.
echo ============================================================
echo.
pause
