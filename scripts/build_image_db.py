"""
build_image_db.py - V-Archive 자켓 이미지 수집 및 image_index.db 빌드

songs.json에서 song_id 목록을 가져와서
image_index.db에 없는 곡만 증분으로 추가한다.

Usage:
    python scripts/build_image_db.py [--db-path PATH] [--force-all]
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from dataclasses import dataclass, field

import httpx

# ------------------------------------------------------------------
# 설정
# ------------------------------------------------------------------

SONGS_JSON_URL  = "https://v-archive.net/db/v2/songs.json"
JACKET_URL_TMPL = "https://v-archive.net/s3/images/jackets/{song_id}.jpg"

DEFAULT_DB_PATH = Path("image_index.db")
REQUEST_TIMEOUT = 15.0
DOWNLOAD_INTERVAL_SEC = 0.1   # V-Archive 서버 부하 방지용 딜레이


# ------------------------------------------------------------------
# image_index.db 관련 (image_db.py 로직 인라인)
# ------------------------------------------------------------------




def _get_existing_ids(db_path: Path) -> set[str]:
    if not db_path.exists():
        return set()
    try:
        with sqlite3.connect(db_path) as conn:
            rows = conn.execute("SELECT image_id FROM images").fetchall()
        return {str(r[0]) for r in rows}
    except Exception as e:
        print(f"[DB] 기존 ID 조회 실패: {e}")
        return set()





# ------------------------------------------------------------------
# V-Archive 데이터 수집
# ------------------------------------------------------------------

def fetch_songs() -> list[dict]:
    print(f"[Fetch] songs.json 다운로드: {SONGS_JSON_URL}")
    try:
        resp = httpx.get(SONGS_JSON_URL, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        songs = resp.json()
        songs = [s for s in songs if s.get("title") is not None]
        print(f"[Fetch] {len(songs)}곡 확인")
        return songs
    except Exception as e:
        print(f"[Fetch] songs.json 다운로드 실패: {e}")
        sys.exit(1)


def download_jacket_bytes(song_id: str, client: httpx.Client) -> bytes | None:
    url = JACKET_URL_TMPL.format(song_id=song_id)
    try:
        resp = client.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[Download] 실패 ({song_id}): {e}")
        return None


# ------------------------------------------------------------------
# 메인 빌드 로직
# ------------------------------------------------------------------

@dataclass
class BuildResult:
    total: int
    added: list[dict]
    fail: int
    skip: int


def build(db_path: Path, force_all: bool) -> BuildResult:
    db_path.parent.mkdir(parents=True, exist_ok=True)

    with sqlite3.connect(db_path) as conn:
        _ensure_schema(conn)
        conn.commit()

    songs = fetch_songs()
    song_map = {str(s["title"]): s for s in songs}
    all_ids = list(song_map.keys())
    existing_ids = set() if force_all else _get_existing_ids(db_path)

    targets = [sid for sid in all_ids if sid not in existing_ids]
    print(f"[Build] 전체 {len(all_ids)}곡 / 기존 {len(existing_ids)}곡 / 신규 {len(targets)}곡")

    # ✅ 추가 (변경 없음)
    if not targets:
        print("[Build] 추가할 곡 없음 - 완료")
        Path("no_changes").write_text("1")
        return BuildResult(total=len(all_ids), added=[], fail=0, skip=0)

    skip = 0
    downloaded_files = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        print(f"[Build] 임시 자켓 디렉토리 생성: {tmp_path}")

        # 1. 이미지 다운로드 진행 및 임시 저장
        with httpx.Client() as client:
            for i, song_id in enumerate(targets, 1):
                img_data = download_jacket_bytes(song_id, client)
                if img_data is None:
                    print(f"[{i}/{len(targets)}] SKIP  {song_id} (이미지 없음)")
                    skip += 1
                else:
                    file_path = tmp_path / f"{song_id}.jpg"
                    file_path.write_bytes(img_data)
                    downloaded_files.append(song_id)
                    print(f"[{i}/{len(targets)}] DOWNLOADED {song_id}")
                time.sleep(DOWNLOAD_INTERVAL_SEC)

        # 2. Rust db_builder CLI 실행
        if downloaded_files:
            # 메인 overmax 저장소는 상대경로로 ../overmax 에 있음
            cmd = [
                "cargo", "run", "--manifest-path", "../overmax/Cargo.toml",
                "-p", "overmax-data", "--bin", "db_builder", "--",
                "--image-dir", str(tmp_path), "--db-path", str(db_path)
            ]
            print(f"[Build] Rust db_builder 실행: {' '.join(cmd)}")
            try:
                result = subprocess.run(cmd, capture_output=True, text=True, check=True)
                print(result.stdout)
                if result.stderr:
                    print(f"[Warn] CLI 표준 에러 출력:\n{result.stderr}")
            except subprocess.CalledProcessError as e:
                print(f"[Build] Rust db_builder 실행 실패 (exit={e.returncode}):\n{e.stderr}")
                raise e

    # 3. 최종 DB 상태를 확인하여 추가된 곡 계산
    new_existing_ids = _get_existing_ids(db_path)
    newly_added_ids = new_existing_ids - existing_ids

    added = []
    for song_id in newly_added_ids:
        song = song_map.get(song_id)
        if song:
            added.append({
                "song_id": song_id,
                "name": song.get("name", ""),
                "composer": song.get("composer", ""),
            })

    fail = len(downloaded_files) - len(added)

    print(f"\n[Build] 완료: success={len(added)}, fail={fail}, skip={skip}")

    # ✅ 추가 (혹시라도 추가 0이면)
    if not added:
        Path("no_changes").write_text("1")

    return BuildResult(total=len(all_ids), added=added, fail=fail, skip=skip)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="V-Archive 자켓 이미지 DB 빌드")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH), help="출력 DB 경로")
    parser.add_argument("--force-all", action="store_true", help="기존 항목 포함 전체 재빌드")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = build(Path(args.db_path), args.force_all)

    # GitHub Actions에서 릴리즈 노트로 사용할 파일 생성
    notes_path = Path("release_notes.md")
    lines = [
        f"## Image DB",
        f"",
        f"- 총 **{result.total}곡** 등록",
        f"- 이번 업데이트: **{len(result.added)}곡 추가**",
    ]
    if result.added:
        lines += ["", "### 추가된 곡", ""]
        for s in result.added:
            lines.append(f"- {s['name']} — {s['composer']}")

    notes_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\n[Build] 릴리즈 노트 생성: {notes_path}")
