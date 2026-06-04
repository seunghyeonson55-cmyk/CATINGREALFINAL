"""
배우 데이터에 정량 필터용 필드 추가: height_cm, voice(목소리 유형).
- 성별/나이는 이미 있음 (사진에서 추정).
- 키/목소리는 사진으로 알 수 없으므로, 실제 제품에선 '지원서 입력값'.
  여기선 데모를 위해 유형(arch)·성별에 어울리게 그럴듯한 값을 채운다.
"""
import json
import random

random.seed(42)

# 유형별 목소리 경향 (데모용)
VOICE_BY_ARCH = {
    "야성미남": ["저음", "중저음"],
    "중후남":   ["저음", "중저음"],
    "도시미남": ["중저음", "중음"],
    "청량남":   ["미성", "중음"],
    "카리스마녀": ["중저음", "허스키"],
    "시크녀":   ["중저음", "허스키"],
    "지적녀":   ["중음", "미성"],
    "청순녀":   ["미성", "중음"],
    "귀여운녀": ["미성"],
}


def height_for(gender, arch):
    if gender == "남":
        base = random.randint(172, 188)
        if arch in ("야성미남", "중후남"):
            base = random.randint(176, 190)
    else:
        base = random.randint(160, 173)
        if arch in ("카리스마녀", "시크녀"):
            base = random.randint(166, 175)
    return base


def enrich(path, has_arch=True):
    actors = json.load(open(path, encoding="utf-8"))
    for a in actors:
        arch = a.get("arch", "")
        a["height_cm"] = height_for(a["gender"], arch)
        choices = VOICE_BY_ARCH.get(arch, ["중음", "중저음", "미성", "허스키"])
        a["voice"] = random.choice(choices)
    json.dump(actors, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"{path}: {len(actors)}명에 height_cm·voice 추가")
    for a in actors[:3]:
        print(f"   {a['name']} ({a['gender']}·{a['age']}) {a['height_cm']}cm, 목소리 {a['voice']}")


enrich("data/actors.json")
enrich("data/test_actors.json")
