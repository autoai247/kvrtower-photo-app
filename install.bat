@echo off
title 카메라 영수증 프린터 설치
cd /d "%~dp0"

echo.
echo ============================================================
echo  카메라 영수증 프린터 자동 설치
echo  폴더: %CD%
echo ============================================================
echo.
echo 시작하려면 아무 키나 누르세요.
pause >nul
echo.

REM ============================================================
REM  [1/3] Python 확인 (없으면 winget으로 설치 시도)
REM ============================================================
echo [1/3] Python 확인...
python --version 2>nul
if errorlevel 1 (
    echo     Python 이 없습니다. winget 으로 설치 시도...
    where winget >nul 2>&1
    if errorlevel 1 (
        echo.
        echo     [수동 설치 필요] https://www.python.org/downloads/ 에서
        echo     Python 3.12 를 받아 설치 후 이 창을 닫고 다시 실행하세요.
        pause
        exit /b 1
    )
    winget install -e --id Python.Python.3.12 --silent --accept-source-agreements --accept-package-agreements
    set "PATH=%PATH%;%LOCALAPPDATA%\Programs\Python\Python312;%LOCALAPPDATA%\Programs\Python\Python312\Scripts"
    python --version 2>nul
    if errorlevel 1 (
        echo     Python 설치는 됐지만 PATH 인식이 안됩니다. PC 재부팅 후 다시 실행하세요.
        pause
        exit /b 1
    )
)
echo     OK
echo.

REM ============================================================
REM  [2/3] Python 패키지 설치
REM ============================================================
echo [2/3] 라이브러리 설치 중 (1~2분 소요)...
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo     [오류] pip install 실패
    pause
    exit /b 1
)
echo     OK
echo.

REM ============================================================
REM  [3/3] EXE 빌드
REM ============================================================
echo [3/3] 실행파일 만드는 중 (2~3분 소요)...
if exist build rmdir /s /q build
python -m PyInstaller --noconfirm --clean --onefile --windowed --name CameraReceiptPrinter --paths src --collect-submodules PIL --collect-submodules cv2 --add-data "assets;assets" --collect-all mediapipe --collect-all pyttsx3 --collect-all screeninfo src\main.py
if not exist "dist\CameraReceiptPrinter.exe" (
    echo     [오류] 빌드 실패
    pause
    exit /b 1
)
echo     OK
echo.

REM ============================================================
REM  바탕화면 바로가기 생성
REM ============================================================
echo 바탕화면 바로가기 생성...
set "EXE=%CD%\dist\CameraReceiptPrinter.exe"
set "WORK=%CD%\dist"
powershell -NoProfile -Command "$s=New-Object -ComObject WScript.Shell; $lnk=$s.CreateShortcut([Environment]::GetFolderPath('Desktop')+'\카메라영수증프린터.lnk'); $lnk.TargetPath='%EXE%'; $lnk.WorkingDirectory='%WORK%'; $lnk.IconLocation='%EXE%,0'; $lnk.Save()"
echo     OK
echo.

echo ============================================================
echo  ★ 설치 완료! ★
echo  바탕화면의 [카메라영수증프린터] 아이콘을 더블클릭하세요.
echo ============================================================
echo.
pause
