"""
배우 100명의 인상 설명을 AI로 풍부하게 재생성.
- 입력: data/actors.json (얇은 설명)
- 출력: data/actors.json 의 'desc'를 교체, 원본은 'desc_orig'에 보관
(지금은 속성 기반 생성. 실제 제품에선 사진을 보고 멀티모달 AI가 씀 = STEP 2)
"""
import json
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv("/Users/ve/Desktop/클로드코드/cating/.env")
client = OpenAI()
MODEL = "gpt-5.4-mini"

actors = json.load(open("data/actors.json", encoding="utf-8"))

SYSTEM = (
    "너는 캐스팅 감독을 돕는 어시스턴트다. 배우의 인상 속성 데이터를 받아, "
    "감독이 한눈에 '이 느낌이다' 싶게 만드는 생생한 첫인상 묘사를 한국어로 쓴다.\n"
    "규칙:\n"
    "- 각 배우당 1~2문장, 감각적이고 구체적으로.\n"
    "- 배우마다 표현을 다르게. 똑같은 상투구('터프하고 든든한 존재감' 등) 반복 금지.\n"
    "- 속성 점수에 어울리면 색기/퇴폐/날티/풋풋/반항/우아/나른함 같은 미묘한 뉘앙스도 자연스럽게 녹여라.\n"
    "- 실존 연예인 이름은 절대 쓰지 마라.\n"
    '- 반드시 JSON으로만 답하라: {"items":[{"id":정수,"desc":"묘사"}]}'
)


def make_block(batch):
    lines = []
    for a in batch:
        traits = ", ".join(f"{t} {v}" for t, v in
                            sorted(a["traits"].items(), key=lambda x: -x[1]))
        g = "남자" if a["gender"] == "남" else "여자"
        lines.append(f'id {a["id"]}: {g}, {a["age"]}세, 유형 {a["arch"]}, 인상속성[{traits}]')
    return "\n".join(lines)


new_desc = {}
B = 20
for i in range(0, len(actors), B):
    batch = actors[i:i + B]
    user = "다음 배우들의 인상을 묘사해줘:\n" + make_block(batch)
    resp = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "system", "content": SYSTEM},
                  {"role": "user", "content": user}],
        response_format={"type": "json_object"},
    )
    items = json.loads(resp.choices[0].message.content)["items"]
    for it in items:
        new_desc[int(it["id"])] = it["desc"].strip()
    print(f"  배치 {i//B+1}: {len(items)}명 완료")

for a in actors:
    if a["id"] in new_desc:
        a["desc_orig"] = a["desc"]
        a["desc"] = new_desc[a["id"]]

json.dump(actors, open("data/actors.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=2)
print(f"\n완료: {len(new_desc)}명 설명 재생성 → data/actors.json 저장")
print("\n샘플 3명:")
for a in actors[:3]:
    print(f"  [{a['arch']}] {a['name']}: {a['desc']}")
