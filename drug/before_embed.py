#!/usr/bin/env python3
"""
pill_id_db.csv  ➜  (1) pill_embed_texts.csv  (2) pill_metadata.parquet

embed_text 템플릿 (예)
─────────────────────
페라트라정2.5밀리그램(레트로졸). 모양:원형, 색:노랑.
식별문자 앞:YH 뒤:LT. 크기:6.1×6.1×3.5mm.
효능:항악성종양제. 겉모양:어두운 황색의 원형 필름코팅정
"""
import pandas as pd
import re

# ── 경로 ────────────────────────────────────────────────
SRC = "pill_id_db.csv"              # 원본 CSV
EMB = "pill_embed_texts.csv"        # ITEM_SEQ, embed_text
META = "pill_metadata.parquet"      # 추가 정보

# ── 컬럼 정의 ───────────────────────────────────────────
embed_cols = [
    "ITEM_NAME",
    "DRUG_SHAPE",
    "COLOR_CLASS1",
    "COLOR_CLASS2",
    "PRINT_FRONT",
    "PRINT_BACK",
    "LENG_LONG",
    "LENG_SHORT",
    "THICK",
    "CLASS_NAME",
    "CHART",
]

meta_cols = [
    "ITEM_SEQ",
    "ITEM_IMAGE",
    "ENTP_NAME",
    "LENG_LONG",
    "LENG_SHORT",
    "THICK",
    "FORM_CODE_NAME",
    "ETC_OTC_NAME",
]

# ── 전처리 ─────────────────────────────────────────────
df = pd.read_csv(SRC)

# meta_cols에 들어가는 모든 컬럼을 문자열로 변환, NaN은 빈 문자열로
for col in meta_cols:
    if col in df.columns:
        df[col] = df[col].fillna("").astype(str)

def tmpl(row) -> str:
    name       = row.ITEM_NAME
    shape      = row.DRUG_SHAPE
    color1     = row.COLOR_CLASS1 or ""
    color2     = f"/{row.COLOR_CLASS2}" if pd.notna(row.COLOR_CLASS2) and row.COLOR_CLASS2 else ""
    pf, pb     = row.PRINT_FRONT or "-", row.PRINT_BACK or "-"
    size       = f"{row.LENG_LONG}×{row.LENG_SHORT}×{row.THICK}mm"
    cls        = row.CLASS_NAME or ""
    chart      = row.CHART or ""
    txt = (
        f"{name}. 모양:{shape}, 색:{color1}{color2}. "
        f"식별문자 앞:{pf} 뒤:{pb}. 크기:{size}. "
        f"효능:{cls}. 겉모양:{chart}"
    )
    return re.sub(r"\s+", " ", txt).strip()

df["embed_text"] = df.apply(tmpl, axis=1)

# ── 저장 ───────────────────────────────────────────────
df[["ITEM_SEQ", "embed_text"]].to_csv(EMB, index=False, encoding="utf-8-sig")
df[meta_cols].to_parquet(META, index=False)

print(f"✅ {len(df):,} rows processed -> {EMB}, {META}")
