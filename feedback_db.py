"""
피드백(정답 신호) 저장소 — SQLite
================================================
■ 가장 중요한 설계 원칙(절대 어기지 말 것)
  - '얼굴 분석(인상 점수)'은 고정된 자(ruler)다. 이 파일은 그 점수를 **절대 건드리지 않는다.**
  - 여기 쌓는 것은 오직 "검색 결과를 어떤 순서로 보여줄지"(=랭킹)를 나중에 학습할 신호다.
    → 학습은 '랭킹'에만 적용. '인상 점수'에는 적용 안 함. (둘을 완전히 분리)

■ 지금 단계
  - '기록만' 한다. 학습(재랭킹)은 데이터가 충분히 쌓인 뒤 별도로 켠다.
  - 이 파일은 신호를 차곡차곡 쌓는 '창고'일 뿐, 검색 순위를 바꾸지 않는다.

■ 신호의 종류
  - 명시적(explicit): 👍 up / 👎 down  — 감독이 직접 누른 평가
  - 암묵적(implicit): click(자세히 봄) / shortlist(찜) / contact(컨택)
    → 말로 안 해도 '관심의 세기'를 드러내는 행동. 보통 contact > shortlist > click 순으로 강한 신호.

■ 테이블 (CLAUDE.md 스키마 초안과 동일한 방향)
  searches : 어떤 '검색어'로 검색했는가
  events   : 그 검색에서 '누구'에게 '무슨 신호'를 줬는가 (+ 그때의 순위·점수)
"""
from __future__ import annotations
import os
import sqlite3
import datetime
import numpy as np

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback.db")

# ■ 학습 스위치 — 지금은 '기록만' 한다(꺼짐). 데이터가 충분히 쌓인 뒤에만 True로 켠다.
#   이 값이 False인 동안 아래 재랭킹 함수들은 항상 '보정 없음(0)'을 돌려준다.
LEARNING_ENABLED = False

# 신호 종류와 '관심 세기'(나중에 재랭킹 가중치로 쓸 참고값. 지금은 저장만, 학습엔 미사용)
SIGNAL_WEIGHTS = {
    "down": -1.0,      # 명시적 부정
    "click": 0.3,      # 자세히 봄(약한 관심)
    "shortlist": 0.7,  # 찜(중간 관심)
    "up": 1.0,         # 명시적 긍정
    "contact": 1.5,    # 컨택(가장 강한 관심 — 실제 섭외 의사)
}


# ---------- 벡터(의미 좌표) ↔ 저장용 바이트 변환 ----------
# 검색어의 임베딩 벡터를 DB에 그대로(BLOB) 넣고 빼기 위한 도우미.
# float32로 통일해 용량을 아낀다(검색 엔진 벡터는 정규화돼 있어 코사인=내적).

def _vec_to_blob(vec) -> bytes | None:
    if vec is None:
        return None
    return np.asarray(vec, dtype=np.float32).tobytes()


def _blob_to_vec(blob) -> np.ndarray | None:
    if not blob:
        return None
    return np.frombuffer(blob, dtype=np.float32)


def _unit(v: np.ndarray) -> np.ndarray:
    """방향만 남기려고 길이를 1로 맞춘다(0 벡터는 그대로)."""
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _conn():
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute("PRAGMA journal_mode=WAL")   # 동시 접근에 안정적
    return c


def init_db():
    """테이블이 없으면 만든다(있으면 그대로 둠). 앱 시작 때 한 번 호출."""
    with _conn() as c:
        c.executescript(
            """
            CREATE TABLE IF NOT EXISTS searches (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text     TEXT NOT NULL,      -- 감독이 입력한 검색어 원문
                expanded_text  TEXT,               -- AI가 풀어 쓴 해석문(있으면)
                query_embedding BLOB,              -- (나중에 채움) 비슷한 검색어끼리 묶기용
                created_at     TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                search_id   INTEGER,               -- 어떤 검색에서 일어난 신호인가(searches.id)
                actor_uid   TEXT,                  -- 대상 지원자 식별자
                actor_name  TEXT,
                type        TEXT NOT NULL,         -- click/shortlist/contact/up/down
                rank        INTEGER,               -- 그 검색에서 보여준 순위(1=맨 위)
                score       REAL,                  -- 그때의 분위기 점수(코사인)
                created_at  TEXT NOT NULL,
                FOREIGN KEY (search_id) REFERENCES searches(id)
            );
            CREATE INDEX IF NOT EXISTS idx_events_search ON events(search_id);
            CREATE INDEX IF NOT EXISTS idx_events_actor  ON events(actor_uid);
            CREATE INDEX IF NOT EXISTS idx_events_type   ON events(type);

            -- 배우(지원자)별 피드백 누적 — '이 배우는 어떤 의미의 검색에서 평이 좋은가'를
            -- 한 곳에 모은다. 검색어 글자는 매번 달라도, 신호가 들어올 때마다
            --   pref_vec += (신호세기) × (그 검색의 의미 벡터)
            -- 로 더해두면, pref_vec의 '방향'이 곧 그 배우가 호평받는 검색 의미가 된다.
            -- (배우는 고정이라 데이터가 흩어지지 않고 여기 쌓인다)
            CREATE TABLE IF NOT EXISTS actor_signals (
                actor_uid    TEXT PRIMARY KEY,    -- 지원자 식별자
                actor_name   TEXT,
                pref_vec     BLOB,                -- Σ (신호세기 × 검색벡터)  ← 호평 '방향'
                sum_weight   REAL DEFAULT 0,      -- Σ 신호세기(부호 포함) — 전반적 호/불호
                abs_weight   REAL DEFAULT 0,      -- Σ |신호세기| — 데이터가 얼마나 모였나(신뢰도)
                pos_count    INTEGER DEFAULT 0,   -- 긍정 신호 수(up/click/shortlist/contact)
                neg_count    INTEGER DEFAULT 0,   -- 부정 신호 수(down)
                event_count  INTEGER DEFAULT 0,   -- 반영된 신호 총 건수
                updated_at   TEXT
            );
            """
        )


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def get_or_create_search(query_text: str, expanded_text: str | None = None,
                         embedding=None) -> int | None:
    """검색어 한 건을 searches에 기록(같은 검색어면 기존 행 재사용). search_id 반환.
    검색어가 비어 있으면 기록하지 않고 None.
    embedding(검색 의미 벡터)을 주면 함께 저장한다 — 비슷한 의미 검색끼리 묶는 핵심."""
    q = (query_text or "").strip()
    if not q:
        return None
    blob = _vec_to_blob(embedding)
    with _conn() as c:
        row = c.execute(
            "SELECT id, query_embedding FROM searches WHERE query_text=? ORDER BY id LIMIT 1", (q,)
        ).fetchone()
        if row:
            # 기존 행에 벡터가 비어 있고 이번에 벡터가 들어오면 채워준다.
            if blob is not None and not row[1]:
                c.execute("UPDATE searches SET query_embedding=? WHERE id=?", (blob, row[0]))
            return row[0]
        cur = c.execute(
            "INSERT INTO searches(query_text, expanded_text, query_embedding, created_at) "
            "VALUES(?,?,?,?)",
            (q, expanded_text, blob, _now()),
        )
        return cur.lastrowid


def set_search_embedding(search_id, embedding) -> None:
    """이미 만든 검색 행에 의미 벡터를 채워 넣는다(비어 있을 때만).
    앱은 search_id를 먼저 만들고 벡터를 그 다음에 계산하므로 이 함수로 뒤늦게 붙인다."""
    if search_id is None or embedding is None:
        return
    blob = _vec_to_blob(embedding)
    with _conn() as c:
        c.execute(
            "UPDATE searches SET query_embedding=? WHERE id=? AND query_embedding IS NULL",
            (blob, search_id),
        )


def log_event(search_id, actor_uid, actor_name, etype,
              rank: int | None = None, score: float | None = None):
    """신호 한 건을 events에 기록한다. (학습은 하지 않음 — 저장만)
    덤으로, 그 검색에 의미 벡터가 저장돼 있고 대상 배우가 있으면
    actor_signals(배우별 누적)에도 '검색 의미 × 신호세기'를 더해 둔다.
    → 이것도 '기록'일 뿐, 검색 순위는 바꾸지 않는다."""
    with _conn() as c:
        c.execute(
            "INSERT INTO events(search_id, actor_uid, actor_name, type, rank, score, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (search_id, actor_uid, actor_name, etype,
             rank, float(score) if score is not None else None, _now()),
        )
        # 배우별 누적: 대상 배우가 있고, 점수화 가능한 신호이며, 그 검색에 벡터가 있을 때만.
        if actor_uid and etype in SIGNAL_WEIGHTS and search_id is not None:
            row = c.execute(
                "SELECT query_embedding FROM searches WHERE id=?", (search_id,)
            ).fetchone()
            qvec = _blob_to_vec(row[0]) if row else None
            if qvec is not None:
                _accumulate_actor_signal(c, actor_uid, actor_name, etype, qvec)


def _accumulate_actor_signal(c, actor_uid, actor_name, etype, qvec: np.ndarray) -> None:
    """배우 한 명의 누적 칸을 갱신한다(있으면 더하고, 없으면 새로 만든다).
    pref_vec 에 '신호세기(부호 포함) × 검색벡터'를 더해 호평 방향을 쌓는다."""
    w = float(SIGNAL_WEIGHTS[etype])
    qvec = np.asarray(qvec, dtype=np.float32)
    row = c.execute(
        "SELECT pref_vec, sum_weight, abs_weight, pos_count, neg_count, event_count "
        "FROM actor_signals WHERE actor_uid=?", (actor_uid,),
    ).fetchone()
    if row and row[0]:
        pref = _blob_to_vec(row[0]).astype(np.float32).copy()
        sum_w, abs_w, pos_c, neg_c, ev_c = row[1], row[2], row[3], row[4], row[5]
        if pref.shape != qvec.shape:          # 차원이 다르면(모델 교체 등) 새로 시작
            pref = np.zeros_like(qvec)
            sum_w = abs_w = 0.0; pos_c = neg_c = ev_c = 0
    else:
        pref = np.zeros_like(qvec)
        sum_w = abs_w = 0.0; pos_c = neg_c = ev_c = 0
    pref = pref + w * qvec
    sum_w += w
    abs_w += abs(w)
    ev_c += 1
    if w > 0:
        pos_c += 1
    elif w < 0:
        neg_c += 1
    c.execute(
        "INSERT OR REPLACE INTO actor_signals"
        "(actor_uid, actor_name, pref_vec, sum_weight, abs_weight, pos_count, neg_count, event_count, updated_at) "
        "VALUES(?,?,?,?,?,?,?,?,?)",
        (actor_uid, actor_name, _vec_to_blob(pref), sum_w, abs_w, pos_c, neg_c, ev_c, _now()),
    )


# ==================================================================
#  미래의 '의미 기반 재랭킹' — 지금은 설계/조회만, 실제 순위 보정은 LEARNING_ENABLED=False라 꺼짐.
#  데이터가 충분히 쌓이면 LEARNING_ENABLED=True 로 켜고, 검색 단계에서 rerank_bonus()를 더하면 된다.
# ==================================================================

def get_actor_signal(actor_uid: str) -> dict | None:
    """배우 한 명의 누적 피드백을 돌려준다(없으면 None).
    pref_vec(호평 방향, 단위벡터), 신뢰도, 긍/부정 수 등."""
    with _conn() as c:
        row = c.execute(
            "SELECT actor_name, pref_vec, sum_weight, abs_weight, pos_count, neg_count, event_count "
            "FROM actor_signals WHERE actor_uid=?", (actor_uid,),
        ).fetchone()
    if not row:
        return None
    pref = _blob_to_vec(row[1])
    return {
        "actor_name": row[0],
        "pref_unit": _unit(pref) if pref is not None else None,  # 호평 '방향'(길이1)
        "sum_weight": row[2], "abs_weight": row[3],
        "pos_count": row[4], "neg_count": row[5], "event_count": row[6],
    }


def nearby_feedback(query_vec, top_searches: int = 50, min_sim: float = 0.55) -> dict:
    """이번 검색 벡터와 '의미가 가까운' 과거 검색들을 찾아, 거기서 나온 신호를
    배우별로 모아 돌려준다. (글자가 달라도 의미가 가까우면 피드백이 공유되는 핵심 조회)
    반환: { actor_uid: {"name":.., "score":누적보정점수, "events":건수} }
    ※ 조회만 한다 — 실제 순위 반영은 호출하는 쪽에서 LEARNING_ENABLED일 때만."""
    q = np.asarray(query_vec, dtype=np.float32)
    qn = _unit(q)
    out: dict[str, dict] = {}
    with _conn() as c:
        rows = c.execute(
            "SELECT id, query_embedding FROM searches WHERE query_embedding IS NOT NULL"
        ).fetchall()
        sims = []
        for sid, blob in rows:
            v = _blob_to_vec(blob)
            if v is None or v.shape != qn.shape:
                continue
            sim = float(_unit(v) @ qn)            # 검색끼리 코사인 유사도
            if sim >= min_sim:
                sims.append((sim, sid))
        sims.sort(reverse=True)
        for sim, sid in sims[:top_searches]:
            evs = c.execute(
                "SELECT actor_uid, actor_name, type FROM events "
                "WHERE search_id=? AND actor_uid IS NOT NULL", (sid,),
            ).fetchall()
            for uid, name, etype in evs:
                w = SIGNAL_WEIGHTS.get(etype)
                if w is None:
                    continue
                rec = out.setdefault(uid, {"name": name, "score": 0.0, "events": 0})
                rec["score"] += w * sim          # 의미가 가까울수록(sim 큰) 더 세게 반영
                rec["events"] += 1
    return out


def rerank_bonus(query_vec, actor_uids: list[str], strength: float = 0.05) -> dict:
    """검색 결과 순위에 '더해줄 보정값'을 배우별로 돌려준다.
    LEARNING_ENABLED=False 인 지금은 항상 0(=순위 안 바뀜). 켜면 비로소 작동.
    조합: (1) 비슷한 의미 과거검색의 신호 + (2) 그 배우의 누적 호평방향 일치도."""
    zeros = {uid: 0.0 for uid in actor_uids}
    if not LEARNING_ENABLED or query_vec is None:
        return zeros                              # ← 지금 단계: 항상 보정 없음
    qn = _unit(np.asarray(query_vec, dtype=np.float32))
    near = nearby_feedback(query_vec)
    bonus = {}
    for uid in actor_uids:
        b = 0.0
        if uid in near:
            b += near[uid]["score"]               # 비슷한 검색에서의 평
        sig = get_actor_signal(uid)
        if sig and sig["pref_unit"] is not None and sig["pref_unit"].shape == qn.shape:
            # 이 배우가 호평받던 '방향'과 이번 검색 방향이 얼마나 일치하나 × 신뢰도
            conf = min(1.0, sig["abs_weight"] / 5.0)
            b += float(sig["pref_unit"] @ qn) * conf
        bonus[uid] = b * strength
    return bonus


def stats() -> dict:
    """쌓인 양을 한눈에 보기 위한 요약(화면에 '눈으로 보여주기'용)."""
    with _conn() as c:
        n_searches = c.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        n_distinct_q = c.execute("SELECT COUNT(DISTINCT query_text) FROM searches").fetchone()[0]
        n_vec = c.execute(
            "SELECT COUNT(*) FROM searches WHERE query_embedding IS NOT NULL"
        ).fetchone()[0]
        n_events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_type = dict(c.execute("SELECT type, COUNT(*) FROM events GROUP BY type").fetchall())
        n_actors = c.execute("SELECT COUNT(*) FROM actor_signals").fetchone()[0]
    return {"searches": n_searches, "distinct_queries": n_distinct_q,
            "searches_with_vector": n_vec, "events": n_events, "by_type": by_type,
            "actors_with_signal": n_actors, "learning_enabled": LEARNING_ENABLED}


def actor_activity(actor_uid: str) -> dict:
    """배우 한 명이 받은 신호를 종류별로 센다(배우 활동지표 표시용).
    예: {'shortlist': 3, 'contact': 1, 'click': 5, 'up': 2, 'down': 0}."""
    if not actor_uid:
        return {}
    with _conn() as c:
        rows = c.execute(
            "SELECT type, COUNT(*) FROM events WHERE actor_uid=? GROUP BY type",
            (actor_uid,),
        ).fetchall()
    return dict(rows)


def recent_events(limit: int = 20):
    """최근 신호 몇 건(검색어·대상·종류·순위)을 화면 확인용으로 가져온다."""
    with _conn() as c:
        return c.execute(
            """SELECT e.created_at, COALESCE(s.query_text,'(검색어 없음)'),
                      e.actor_name, e.type, e.rank, e.score
               FROM events e LEFT JOIN searches s ON e.search_id = s.id
               ORDER BY e.id DESC LIMIT ?""",
            (limit,),
        ).fetchall()
