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

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "feedback.db")

# 신호 종류와 '관심 세기'(나중에 재랭킹 가중치로 쓸 참고값. 지금은 저장만, 학습엔 미사용)
SIGNAL_WEIGHTS = {
    "down": -1.0,      # 명시적 부정
    "click": 0.3,      # 자세히 봄(약한 관심)
    "shortlist": 0.7,  # 찜(중간 관심)
    "up": 1.0,         # 명시적 긍정
    "contact": 1.5,    # 컨택(가장 강한 관심 — 실제 섭외 의사)
}


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
            """
        )


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


def get_or_create_search(query_text: str, expanded_text: str | None = None) -> int | None:
    """검색어 한 건을 searches에 기록(같은 검색어면 기존 행 재사용). search_id 반환.
    검색어가 비어 있으면 기록하지 않고 None."""
    q = (query_text or "").strip()
    if not q:
        return None
    with _conn() as c:
        row = c.execute(
            "SELECT id FROM searches WHERE query_text=? ORDER BY id LIMIT 1", (q,)
        ).fetchone()
        if row:
            return row[0]
        cur = c.execute(
            "INSERT INTO searches(query_text, expanded_text, query_embedding, created_at) "
            "VALUES(?,?,?,?)",
            (q, expanded_text, None, _now()),
        )
        return cur.lastrowid


def log_event(search_id, actor_uid, actor_name, etype,
              rank: int | None = None, score: float | None = None):
    """신호 한 건을 events에 기록한다. (학습은 하지 않음 — 저장만)"""
    with _conn() as c:
        c.execute(
            "INSERT INTO events(search_id, actor_uid, actor_name, type, rank, score, created_at) "
            "VALUES(?,?,?,?,?,?,?)",
            (search_id, actor_uid, actor_name, etype,
             rank, float(score) if score is not None else None, _now()),
        )


def stats() -> dict:
    """쌓인 양을 한눈에 보기 위한 요약(화면에 '눈으로 보여주기'용)."""
    with _conn() as c:
        n_searches = c.execute("SELECT COUNT(*) FROM searches").fetchone()[0]
        n_distinct_q = c.execute("SELECT COUNT(DISTINCT query_text) FROM searches").fetchone()[0]
        n_events = c.execute("SELECT COUNT(*) FROM events").fetchone()[0]
        by_type = dict(c.execute("SELECT type, COUNT(*) FROM events GROUP BY type").fetchall())
    return {"searches": n_searches, "distinct_queries": n_distinct_q,
            "events": n_events, "by_type": by_type}


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
