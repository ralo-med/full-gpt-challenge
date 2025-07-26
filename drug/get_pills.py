#!/usr/bin/env python3
"""
MFDS 의약품 낱알식별 전체 덤프 → CSV 저장 스크립트
  • 약 26 k 건(2025‑07 기준)¹을 전부 내려받아 `pill_id_db.csv` 로 저장
  • 요청당 최대 100 행(numOfRows) × 페이지루프
  • ‘type=json’ 로 받아 pandas DataFrame 으로 병합 → CSV

필요:
  pip install pandas requests tqdm python-dotenv
  export MFDS_SERVICE_KEY='발급받은 인증키(URL‑encode 안해도 됨)'

API spec 참고: https://apis.data.go.kr/1471000/MdcinGrnIdntfcInfoService02/getMdcinGrnIdntfcInfoList02
"""

import os, math, time, sys, json
from typing import List, Dict
import requests, pandas as pd
from tqdm import tqdm

# .env 자동 로드
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

SERVICE_KEY = os.getenv("MFDS_SERVICE_KEY")
if not SERVICE_KEY:
    sys.exit("❌  MFDS_SERVICE_KEY 환경변수부터 설정하세요! (.env 파일 또는 export 필요)")

ENDPOINT = (
    "https://apis.data.go.kr/1471000/"
    "MdcinGrnIdntfcInfoService02/getMdcinGrnIdntfcInfoList02"
)
ROWS = 100
TIMEOUT = 5

def fetch_page(page_no: int) -> Dict:
    params = {
        "serviceKey": SERVICE_KEY,
        "type": "json",
        "numOfRows": ROWS,
        "pageNo": page_no,
    }
    r = requests.get(ENDPOINT, params=params, timeout=TIMEOUT)
    r.raise_for_status()
    return r.json()["body"]

def main() -> None:
    print("🔎 MFDS 의약품 낱알식별 전체 덤프 시작")
    first = fetch_page(1)
    total = int(first["totalCount"])
    pages = math.ceil(total / ROWS)
    print(f"총 {total:,}건 ‑ 페이지 {pages}개 (rows={ROWS})")

    items: List[Dict] = first["items"]
    for p in tqdm(range(2, pages + 1), desc="Downloading", unit="page", ncols=80):
        try:
            body = fetch_page(p)
            items.extend(body.get("items", []))
        except Exception as e:
            print(f"⚠️  page {p} 실패 → 재시도 중...", file=sys.stderr)
            time.sleep(1)
            body = fetch_page(p)
            items.extend(body.get("items", []))

    print("🗂️  pandas DataFrame 병합 중...")
    df = pd.json_normalize(items)
    print("💾 CSV 저장 중...")
    df.to_csv("pill_id_db.csv", index=False, encoding="utf-8-sig")
    print(f"✅  저장 완료: pill_id_db.csv  ({len(df):,} rows)")

if __name__ == "__main__":
    main()
