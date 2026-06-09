@echo off
REM 개발/테스트 실행 - .exe 빌드 없이 바로 돌려보기
title 카메라 영수증 프린터 (개발 실행)
cd /d "%~dp0"
python -m pip install -q -r requirements.txt
python "src\main.py"
pause
