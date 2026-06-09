"""K-Culture VR Tower 사진 프린터 — 자동 업데이트 런처 (서버형)

매장 PC는 이 파일(또는 빌드된 KVRTowerPhoto.exe) 하나만 실행하면 됩니다.
실행 시:
  1. 우리 서버(Railway)에서 최신 코드 메타데이터 받음
  2. 변경된 파일만 다운로드
  3. 의존성 자동 설치 (변경 시)
  4. main.py 실행
인터넷 안 되면 마지막 캐시본으로 실행.
"""
import os
import sys
import json
import hashlib
import subprocess
import urllib.request
from pathlib import Path

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
    log("  K-CULTURE VR TOWER · PHOTO PRINTER")
    log("  서버: " + SERVER_URL)
    log("")
    try:
        manifest = get_manifest()
        update_all(manifest)
        install_requirements_if_changed()
    except Exception as e:
        log(f"  ⚠  서버 통신 실패: {e}")
        log(f"     마지막 캐시본으로 실행합니다.")
    return run_main()


if __name__ == "__main__":
    sys.exit(main() or 0)
