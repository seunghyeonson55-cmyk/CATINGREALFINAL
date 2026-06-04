"""
배우 설명 → 임베딩 → 저장.
사용법:
  python build_index.py          # 무료 로컬 모델
  python build_index.py openai   # OpenAI (.env에 키 필요)
결과: data/embeddings_<provider>.npy
"""
import sys
import json
import numpy as np
from engine import get_embedder, build_embeddings, actor_to_text

provider = sys.argv[1] if len(sys.argv) > 1 else "local"

actors = json.load(open("data/actors.json", encoding="utf-8"))
print(f"배우 {len(actors)}명 로드. 제공자: {provider}")
print("예시 임베딩 텍스트:", actor_to_text(actors[0]))

embedder = get_embedder(provider)
emb = build_embeddings(actors, embedder)
out = f"data/embeddings_{provider}.npy"
np.save(out, emb)
print(f"임베딩 완료: {emb.shape[0]}명 × {emb.shape[1]}차원 → {out} 저장")
