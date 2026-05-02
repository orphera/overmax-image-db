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
import sys
import time
from pathlib import Path
from dataclasses import dataclass, field

import cv2
import httpx
import numpy as np

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

def _ensure_schema(conn: sqlite3.Connection):
    conn.execute("""
        CREATE TABLE IF NOT EXISTS images (
            id       INTEGER PRIMARY KEY AUTOINCREMENT,
            image_id TEXT NOT NULL,
            phash    TEXT NOT NULL,
            dhash    TEXT NOT NULL,
            ahash    TEXT NOT NULL,
            hog      BLOB NOT NULL,
            orb      BLOB
        )
    """)
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_images_image_id ON images (image_id)"
    )
    conn.execute("""
        DELETE FROM images
        WHERE id NOT IN (
            SELECT MAX(id) FROM images GROUP BY image_id
        )
    """)
    conn.execute(
        "CREATE UNIQUE INDEX IF NOT EXISTS uq_images_image_id ON images (image_id)"
    )


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


def _upsert_entry(conn: sqlite3.Connection, song_id: str, img: np.ndarray) -> bool:
    gray = _to_gray(img)
    if gray is None:
        return False
    ph, dh, ah = _compute_hashes(gray)
    hog = _compute_hog(gray)
    try:
        conn.execute(
            """
            INSERT INTO images (image_id, phash, dhash, ahash, hog, orb)
            VALUES (?, ?, ?, ?, ?, NULL)
            ON CONFLICT(image_id) DO UPDATE SET
                phash = excluded.phash,
                dhash = excluded.dhash,
                ahash = excluded.ahash,
                hog   = excluded.hog,
                orb   = NULL
            """,
            (song_id, ph, dh, ah, hog.tobytes()),
        )
        return True
    except Exception as e:
        print(f"[DB] upsert 실패 ({song_id}): {e}")
        return False


# ------------------------------------------------------------------
# 이미지 처리 (image_db.py에서 복사)
# ------------------------------------------------------------------

def _to_gray(img: np.ndarray) -> np.ndarray | None:
    if img is None or img.size == 0:
        return None
    if img.ndim == 2:
        return img
    if img.ndim == 3 and img.shape[2] == 4:
        return cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    if img.ndim == 3 and img.shape[2] == 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return None


def _bits_to_hex(bits: np.ndarray) -> str:
    packed = np.packbits(bits.reshape(-1).astype(np.uint8), bitorder="big")
    return "".join(f"{b:02x}" for b in packed)


def _phash(gray: np.ndarray) -> str:
    r = cv2.resize(gray, (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32)
    dct = cv2.dct(r)
    low = dct[:8, :8]
    median = float(np.median(low.reshape(-1)[1:]))
    return _bits_to_hex(low > median)


def _dhash(gray: np.ndarray) -> str:
    r = cv2.resize(gray, (9, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _bits_to_hex(r[:, 1:] > r[:, :-1])


def _ahash(gray: np.ndarray) -> str:
    r = cv2.resize(gray, (8, 8), interpolation=cv2.INTER_AREA).astype(np.float32)
    return _bits_to_hex(r > float(np.mean(r)))


def _compute_hashes(gray: np.ndarray) -> tuple[str, str, str]:
    return _phash(gray), _dhash(gray), _ahash(gray)


def _compute_hog(gray: np.ndarray) -> np.ndarray:
    resized = cv2.resize(gray, (64, 64), interpolation=cv2.INTER_AREA)
    descriptor = cv2.HOGDescriptor(
        _winSize=(64, 64), _blockSize=(16, 16),
        _blockStride=(8, 8), _cellSize=(8, 8), _nbins=9,
    )
    features = descriptor.compute(resized)
    if features is None:
        return np.zeros((1764,), dtype=np.float32)
    return features.reshape(-1).astype(np.float32)


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


def download_jacket(song_id: str, client: httpx.Client) -> np.ndarray | None:
    url = JACKET_URL_TMPL.format(song_id=song_id)
    try:
        resp = client.get(url, timeout=REQUEST_TIMEOUT)
        if resp.status_code == 404:
            return None
        resp.raise_for_status()
        img_array = np.frombuffer(resp.content, dtype=np.uint8)
        img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
        return img
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

    added = []
    fail = 0
    skip = 0

    with httpx.Client() as client:
        with sqlite3.connect(db_path) as conn:
            _ensure_schema(conn)
            for i, song_id in enumerate(targets, 1):
                img = download_jacket(song_id, client)
                if img is None:
                    print(f"[{i}/{len(targets)}] SKIP  {song_id} (이미지 없음)")
                    skip += 1
                elif _upsert_entry(conn, song_id, img):
                    song = song_map[song_id]
                    print(f"[{i}/{len(targets)}] OK    {song_id} ({song.get('name', '')})")
                    added.append({
                        "song_id": song_id,
                        "name": song.get("name", ""),
                        "composer": song.get("composer", ""),
                    })
                else:
                    print(f"[{i}/{len(targets)}] FAIL  {song_id}")
                    fail += 1

                conn.commit()
                time.sleep(DOWNLOAD_INTERVAL_SEC)

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
