"""K-Culture VR Tower 사진 프린터 — 자동 업데이트 런처 (서버형)

매장 PC는 이 파일(또는 빌드된 KVRTowerPhoto.exe) 하나만 실행하면 됩니다.
실행 시:
  1. 자기 자신(launcher) 새 버전 있으면 자동 갱신 (다음 실행부터 적용)
  2. 서버(Railway)에서 최신 앱 코드 메타데이터 받음
  3. 변경된 파일만 다운로드
  4. 의존성 자동 설치 (변경 시)
  5. main.py 실행
인터넷 안 되면 마지막 캐시본으로 실행.
"""
import os
import sys
import json
import hashlib
import shutil
import subprocess
import urllib.request
from pathlib import Path

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  버전 — push 할 때마다 +1
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
LAUNCHER_VERSION = "2026.06.11.1"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
#  설정 — 서버 URL만 채우면 됨 (토큰은 서버가 보관)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SERVER_URL = "https://web-production-36aac.up.railway.app"
# 매장 식별용 (선택, 통계·접근제어 용도)
STORE_ID = "nseoultower"
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

CACHE_DIR = Path(os.environ.get("LOCALAPPDATA", str(Path.home()))) / "KVRTowerPhoto"
CACHE_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_CACHE = CACHE_DIR / "manifest.json"
LAST_REQ_HASH_FILE = CACHE_DIR / ".last_req_hash"


def log(msg):
    print(msg)
    try:
        sys.stdout.flush()
    except Exception:
        pass


def http_get(url: str, timeout: int = 15) -> bytes:
    req = urllib.request.Request(url)
    req.add_header("User-Agent", f"KVRTowerPhoto-Launcher/{STORE_ID}")
    req.add_header("Cache-Control", "no-cache")
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def get_manifest() -> dict:
    """서버에서 최신 파일 목록 + sha 받기"""
    url = f"{SERVER_URL}/manifest?store={STORE_ID}"
    data = http_get(url, timeout=10)
    return json.loads(data.decode("utf-8"))


def file_sha(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha1(path.read_bytes()).hexdigest()


def update_all(manifest: dict) -> int:
    """서버 manifest 와 로컬 캐시 비교 — 변경된 파일만 다운로드"""
    log("=" * 50)
    log("  최신 코드 확인 중...")
    log("=" * 50)
    files = manifest.get("files", [])
    success = 0
    for entry in files:
        rel = entry["path"]            # 로컬 상대 경로
        remote_sha = entry.get("sha", "")
        dest = CACHE_DIR / rel
        if file_sha(dest) == remote_sha and dest.exists():
            log(f"  ·  {rel} (최신)")
            success += 1
            continue
        try:
            url = f"{SERVER_URL}/file/{rel}?store={STORE_ID}"
            data = http_get(url, timeout=30)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(data)
            log(f"  ✓  {rel} (업데이트)")
            success += 1
        except Exception as e:
            if dest.exists():
                log(f"  ⚠  {rel} — 다운로드 실패, 캐시본 사용")
            else:
                log(f"  ✗  {rel} — {e}")
    return success


def install_requirements_if_changed():
    req_file = CACHE_DIR / "requirements.txt"
    if not req_file.exists():
        return
    content = req_file.read_bytes()
    h = hashlib.sha1(content).hexdigest()
    if LAST_REQ_HASH_FILE.exists():
        if LAST_REQ_HASH_FILE.read_text().strip() == h:
            return
    log("의존성 설치 (변경 감지)...")
    try:
        subprocess.run([sys.executable, "-m", "pip", "install", "-q",
                        "-r", str(req_file)], check=False)
        LAST_REQ_HASH_FILE.write_text(h)
    except Exception as e:
        log(f"  ⚠  pip install 실패: {e}")


def apply_pending_self_update():
    """이전 실행에서 받아둔 새 launcher 가 있으면 자기 자신 교체.
    .py 파일일 때만 작동 (EXE 는 사용 중 잠금 때문에 별도 처리).
    """
    try:
        my_path = Path(sys.argv[0]).resolve()
        if my_path.suffix.lower() != ".py":
            return
        pending = my_path.with_suffix(".py.new")
        if not pending.exists():
            return
        # 백업 → 교체
        backup = my_path.with_suffix(".py.bak")
        shutil.copy2(my_path, backup)
        shutil.move(str(pending), str(my_path))
        log(f"  ✓ launcher 자체 업데이트 적용 완료")
    except Exception as e:
        log(f"  ⚠ launcher 적용 실패: {e}")


def self_update_check():
    """manifest 의 launcher.py sha 와 자기 자신 sha 비교.
    다르면 새 launcher.py 받아서 옆에 .new 로 저장 (다음 실행 시 적용).
    """
    try:
        my_path = Path(sys.argv[0]).resolve()
        if my_path.suffix.lower() != ".py":
            return  # EXE 는 별도
        my_sha = file_sha(my_path)
        # 서버에서 새 launcher.py 받아 sha 비교
        new_bytes = http_get(f"{SERVER_URL}/launcher.py", timeout=15)
        new_sha = hashlib.sha1(new_bytes).hexdigest()
        if my_sha == new_sha:
            return
        pending = my_path.with_suffix(".py.new")
        pending.write_bytes(new_bytes)
        log(f"  ★ launcher 새 버전 다운로드 — 다음 실행부터 적용")
    except Exception as e:
        log(f"  ⚠ launcher 자체 업데이트 확인 실패: {e}")


def ensure_desktop_shortcut():
    """첫 실행 시 바탕화면에 아이콘 바로가기 생성. 이미 있으면 skip.

    매장 직원이 launcher.py 위치 모르게 — 바탕화면 아이콘 하나만 더블클릭하면
    실행되도록.
    """
    if os.name != "nt":
        return
    try:
        desktop = (Path(os.environ.get("USERPROFILE", str(Path.home())))
                   / "Desktop")
        if not desktop.exists():
            desktop = (Path(os.environ.get("PUBLIC", "C:\\Users\\Public"))
                       / "Desktop")
        link_path = desktop / "K-Culture VR Tower 사진프린터.lnk"
        if link_path.exists():
            return

        # 아이콘 캐시 (없으면 서버에서 받음)
        icon_path = CACHE_DIR / "kvrtower.ico"
        if not icon_path.exists():
            try:
                data = http_get(
                    f"{SERVER_URL}/file/assets/kvrtower.ico"
                    f"?store={STORE_ID}",
                    timeout=20)
                icon_path.write_bytes(data)
            except Exception:
                icon_path = None

        launcher_path = Path(sys.argv[0]).resolve()
        # 콘솔창 안 뜨도록 pythonw.exe 우선
        py_exe = sys.executable
        pyw = Path(py_exe).with_name("pythonw.exe")
        target_exe = str(pyw) if pyw.exists() else py_exe

        icon_part = (f"; $lnk.IconLocation='{icon_path}'"
                     if icon_path else "")
        ps_cmd = (
            f"$s=New-Object -ComObject WScript.Shell; "
            f"$lnk=$s.CreateShortcut('{link_path}'); "
            f"$lnk.TargetPath='{target_exe}'; "
            f"$lnk.Arguments='\"{launcher_path}\"'; "
            f"$lnk.WorkingDirectory='{launcher_path.parent}'"
            f"{icon_part}; "
            f"$lnk.Save()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", ps_cmd],
                       check=False, capture_output=True, timeout=15)
        log(f"  ✓ 바탕화면 바로가기 생성: {link_path.name}")
    except Exception as e:
        log(f"  ⚠ 바로가기 생성 실패: {e}")


def run_main():
    main_py = CACHE_DIR / "src" / "main.py"
    if not main_py.exists():
        log("")
        log("  ✗ main.py 못 찾음 — 첫 실행 인터넷 확인")
        log(f"    캐시 위치: {CACHE_DIR}")
        input("\n  엔터로 종료...")
        return 1
    log("")
    log("=" * 50)
    log("  프로그램 시작...")
    log("=" * 50)
    log("")
    os.chdir(CACHE_DIR)
    return subprocess.run([sys.executable, str(main_py)]).returncode


def main():
    log("")
    log("  ============================================================")
    log("    K-CULTURE VR TOWER · PHOTO PRINTER")
    log(f"    Launcher v{LAUNCHER_VERSION}")
    log(f"    서버: {SERVER_URL}")
    log("  ============================================================")
    log("")
    apply_pending_self_update()
    try:
        manifest = get_manifest()
        update_all(manifest)
        install_requirements_if_changed()
        self_update_check()
    except Exception as e:
        log(f"  ⚠  서버 통신 실패: {e}")
        log(f"     마지막 캐시본으로 실행합니다.")
    ensure_desktop_shortcut()
    return run_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
