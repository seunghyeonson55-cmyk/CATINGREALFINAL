"""
STEP 2: 얼굴 사진 → 멀티모달 AI 인상 묘사 → 임베딩 → 검색 가능하게 저장.
- 입력: faces/*.png
- 출력: data/test_actors.json (묘사 포함) + data/test_embeddings_openai.npy
"""
import os
import glob
import json
import base64
import numpy as np
from openai import OpenAI
from dotenv import load_dotenv
from engine import get_embedder

load_dotenv("/Users/ve/Desktop/클로드코드/cating/.env")
client = OpenAI()
VISION = "gpt-5.4-mini"

PROMPT = (
    "이 얼굴 사진을 보고 캐스팅 감독을 위한 첫인상을 분석해줘.\n"
    "- desc: 감각적인 인상 묘사 1~2문장(한국어). 실존 연예인 이름 금지.\n"
    "- gender: '남' 또는 '여'\n"
    "- age: 추정 나이(정수)\n"
    '반드시 JSON으로만: {"desc":"...","gender":"남","age":28}'
)

actors = []
for i, path in enumerate(sorted(glob.glob("faces/*.png"))):
    b64 = base64.b64encode(open(path, "rb").read()).decode()
    url = f"data:image/png;base64,{b64}"
    r = client.chat.completions.create(
        model=VISION,
        messages=[{"role": "user", "content": [
            {"type": "text", "text": PROMPT},
            {"type": "image_url", "image_url": {"url": url}}]}],
        response_format={"type": "json_object"},
    )
    data = json.loads(r.choices[0].message.content)
    actors.append({
        "id": 1000 + i,
        "name": f"지원자{chr(65+i)}",          # 지원자A, B, ...
        "photo": path,
        "gender": data.get("gender", "?"),
        "age": data.get("age", 0),
        "desc": data["desc"].strip(),
    })
    print(f"📸 {os.path.basename(path)} → {actors[-1]['name']} ({actors[-1]['gender']}·{actors[-1]['age']})")
    print(f"     “{actors[-1]['desc']}”")

json.dump(actors, open("data/test_actors.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)

# 묘사 → 임베딩
embedder = get_embedder("openai")
emb = embedder.encode([a["desc"] for a in actors])
np.save("data/test_embeddings_openai.npy", emb)
print(f"\n완료: {len(actors)}명 분석+임베딩 → data/test_actors.json, test_embeddings_openai.npy")
