"""
CATING 화면 (Streamlit)
- 탭 1 검색: 문장 검색(분위기) + 정량 필터(성별/나이/키/목소리) + top-k
- 탭 2 지원서 업로드: PDF/PNG/JPG/DOCX → 얼굴 캡쳐 + AI 인상묘사 + 나이·키 자동분류
실행:  streamlit run app.py
"""
import io
import os
import json
import html
import base64
import hashlib
import numpy as np
import streamlit as st
from engine import get_embedder, search_filtered, passes_filters
import intake
import feedback_db

# ---------- API 키 연결: 로컬은 .env 파일, 인터넷 배포(Streamlit Cloud)는 Secrets ----------
# 로컬 PC에서는 .env에서, 배포 서버에서는 Streamlit '비밀(secrets)'에서 키를 읽는다.
# (둘 다 코드에 키를 직접 적지 않음 — 보안 원칙 준수)
from dotenv import load_dotenv
load_dotenv()  # 현재 폴더의 .env가 있으면 읽음(로컬). 서버엔 .env가 없으니 그냥 넘어감.
try:
    if not os.environ.get("OPENAI_API_KEY") and "OPENAI_API_KEY" in st.secrets:
        os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
except Exception:
    pass  # secrets 설정이 없는 환경(예: 로컬)에서는 조용히 넘어감

feedback_db.init_db()   # 피드백 저장용 DB 테이블 준비(없으면 생성)

# ---------- 브랜드 로고 (reference/logo.png 있으면 사용, 없으면 텍스트로 대체) ----------
LOGO_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reference", "logo.png")


@st.cache_data
def logo_b64() -> str | None:
    if os.path.exists(LOGO_PATH):
        return base64.b64encode(open(LOGO_PATH, "rb").read()).decode()
    return None

# ---------- 데이터 로드 (캐시) ----------
@st.cache_data
def load_data():
    actors = json.load(open("data/actors.json", encoding="utf-8"))
    emb = np.load("data/embeddings_openai.npy")
    return actors, emb

@st.cache_resource
def load_embedder():
    return get_embedder("openai")

@st.cache_resource
def load_vision_client():
    # API 키는 파일 맨 위에서 .env(로컬) 또는 st.secrets(서버) 로 이미 os.environ 에 넣어둠.
    from openai import OpenAI
    return OpenAI()


ANALYSIS_CACHE_PATH = "data/analysis_cache.json"


@st.cache_resource
def _analysis_cache():
    """사진 내용 → 분석결과(인상 텍스트·점수) 캐시. 같은 사진은 다시 분석하지 않는다
    = 돌릴 때마다 결과가 동일해지고(결정적) 재분석 비용도 안 든다. 사진 자체는 저장 안 함."""
    import os
    if os.path.exists(ANALYSIS_CACHE_PATH):
        try:
            return json.load(open(ANALYSIS_CACHE_PATH, encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _cache_key(members) -> str:
    """그 사람 사진들의 내용 해시 → 안정적인 식별 키(파일명·순서 무관)."""
    return hashlib.md5(b"".join(sorted(hashlib.md5(d).digest()
                                       for _, d in members))).hexdigest()


def _cache_put(key: str, val: dict):
    c = _analysis_cache()
    c[key] = val
    try:
        json.dump(c, open(ANALYSIS_CACHE_PATH, "w", encoding="utf-8"), ensure_ascii=False)
    except Exception:
        pass


def _chat_json(client, messages):
    """JSON 응답 채팅 호출. 결과를 매번 동일하게(결정적) 만들기 위해
    temperature=0·seed 고정. 일부 모델이 이 인자를 막으면 빼고 한 번 더 시도."""
    from openai import BadRequestError
    kw = dict(model=VISION, messages=messages,
              response_format={"type": "json_object"})
    try:
        return client.chat.completions.create(temperature=0, seed=7, **kw)
    except BadRequestError:
        return client.chat.completions.create(**kw)

actors, EMB = load_data()
embedder = load_embedder()

VISION = "gpt-5.4-mini"
# 기본은 '있는 사진 전부' 분석(001~003 → 3장, 004까지 → 4장).
# 아래 값은 비정상적으로 많을 때만 발동하는 안전 천장(요청당 이미지 과다 → API 실패 방지).
MAX_FACES_PER_PERSON = 30

# 인상 '정도'를 0~1로 채점할 고정 축. 사람마다 같은 잣대로 매겨야 비교가 된다
# (예: 남성성 0.7 vs 0.9). 랭킹엔 안 쓰고 '보여주기용 라벨'로만 쓴다(설계: 하이브리드).
TRAIT_AXES = ["남성성", "여성성", "카리스마", "청순함", "부드러움", "강인함", "도시세련", "따뜻함"]

# ---------- 인상 유형별 색감 (랜딩 팔레트) ----------
PALETTE = {
    "야성미남": ("#c0653a", "#7a2e18"), "도시미남": ("#4a90c0", "#1f4a6e"),
    "청량남": ("#3fb5a8", "#15605a"), "중후남": ("#b08a55", "#5e4525"),
    "청순녀": ("#c89ad8", "#6e4a85"), "시크녀": ("#8088b0", "#3a3e5e"),
    "귀여운녀": ("#e08aa8", "#85405e"), "지적녀": ("#5aa0a0", "#256060"),
    "카리스마녀": ("#b060a0", "#6e2a60"),
}
VOICES = ["저음", "중저음", "중음", "미성", "허스키"]
VOICE_HINT = {"저음": "중후함", "중저음": "차분", "중음": "표준", "미성": "하이톤", "허스키": "허스키"}

st.set_page_config(page_title="CATING — Cast Catching", page_icon="🎬", layout="wide")

# ===== 공통 디자인 토큰 (cating_workflow.html 시안 기준) =====
# 색: 네이비 #1f3a5f · 청록 #52b3a8 · 베이지 #efe9dd · 종이배경 #f6f2ea
# 폰트: 제목 Fraunces(세리프) · 본문 Pretendard
# 질감: 종이배경 + 그레인 노이즈 + 부드러운 그림자 + 둥근 카드
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,500;0,9..144,600;1,9..144,400&display=swap');
@import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard@v1.3.9/dist/web/static/pretendard.css');

:root {
  --navy:#1f3a5f; --navy-d:#152a3f; --navy-l:#2f5180;
  --teal:#52b3a8; --teal-d:#3d8d84;
  --beige:#efe9dd; --paper:#f6f2ea;
  --ink:#1a2e44; --muted:#6f7d8c; --faint:#9aa6b2; --card:#fff; --line:#e6e0d3;
  --shadow-s:0 2px 10px rgba(31,58,95,.06);
  --shadow-m:0 12px 40px rgba(31,58,95,.10);
  --shadow-l:0 24px 70px rgba(31,58,95,.16);
  --serif:'Fraunces',serif;
  --sans:'Pretendard',-apple-system,system-ui,sans-serif;
}

/* 종이질감 배경 + 위쪽 청록 글로우 */
.stApp {
  background:var(--paper);
  background-image:radial-gradient(ellipse 80% 50% at 50% -10%, rgba(82,179,168,.10), transparent 60%);
}
/* 그레인(노이즈) 오버레이 — 화면 전체에 은은하게 */
.stApp::after {
  content:''; position:fixed; inset:0; pointer-events:none; z-index:9999; opacity:.4;
  background-image:url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='200' height='200'%3E%3Cfilter id='n'%3E%3CfeTurbulence type='fractalNoise' baseFrequency='0.8' numOctaves='4'/%3E%3C/filter%3E%3Crect width='200' height='200' filter='url(%23n)' opacity='0.018'/%3E%3C/svg%3E");
}
.block-container { padding-top:1.4rem; }

/* 본문 폰트: Pretendard */
html, body, .stApp, [class*="css"], .stMarkdown, p, span, div, label, input, textarea, button {
  font-family:var(--sans);
}
.stApp { color:var(--ink); }
/* 제목 폰트: Fraunces(세리프) */
h1, h2, h3, h4 { font-family:var(--serif) !important; color:var(--navy); letter-spacing:-.3px; }

/* 버튼: 둥근 알약 모양 + 부드러운 그림자 */
.stButton > button, .stFormSubmitButton > button {
  border-radius:100px; font-weight:600; border:1.5px solid var(--line);
  transition:all .2s; box-shadow:var(--shadow-s);
}
.stButton > button:hover, .stFormSubmitButton > button:hover { transform:translateY(-1px); }
.stButton > button[kind="primary"], .stFormSubmitButton > button[kind="primary"] {
  background:var(--navy); color:#fff; border-color:var(--navy); box-shadow:var(--shadow-m);
}
.stButton > button[kind="primary"]:hover, .stFormSubmitButton > button[kind="primary"]:hover {
  background:var(--navy-l);
}

/* 입력창: 둥근 모서리 + 청록 포커스 */
.stTextInput input, .stTextArea textarea, .stNumberInput input {
  border-radius:12px !important;
}

/* 상단 브랜드 바 */
.brandbar { display:flex; align-items:center; gap:14px; padding:2px 2px 14px;
            border-bottom:1px solid var(--line); margin-bottom:14px; }
.brandbar img { height:48px; }
.brandbar .logotxt { font-family:var(--serif); font-size:26px; font-weight:600;
                     color:var(--navy); letter-spacing:1px; }
.brandbar .tag { font-size:13px; font-weight:700; color:var(--teal-d); letter-spacing:.06em;
                 border-left:1px solid var(--line); padding-left:14px; }
/* 결과 카드 (cating_workflow 시안 톤) */
.card { background:var(--card); border:1px solid var(--line); border-radius:18px;
        padding:14px; margin-bottom:12px; box-shadow:var(--shadow-s);
        position:relative; transition:all .3s cubic-bezier(.2,.8,.2,1); }
.card:hover { transform:translateY(-4px); box-shadow:var(--shadow-l); }
.card.best { border-color:var(--teal); box-shadow:0 0 0 2px var(--teal), var(--shadow-m); }
/* 얼굴: 세로 4:5 비율 */
.avatar { width:100%; aspect-ratio:4/5; border-radius:14px; display:flex;
          align-items:center; justify-content:center; font-size:38px; font-weight:800;
          color:#FFFFFF; margin-bottom:11px; overflow:hidden; }
.avatar img { width:100%; height:100%; object-fit:cover; transition:transform .5s; }
.card:hover .avatar img { transform:scale(1.05); }
/* 얼굴 위 떠 있는 순위 뱃지 */
.rankb { position:absolute; top:21px; right:21px; font-family:var(--serif); font-weight:600;
         font-size:11px; padding:4px 11px; border-radius:100px; color:#fff;
         background:rgba(14,29,44,.6); backdrop-filter:blur(8px); z-index:2; }
.rankb.best { background:var(--teal); }
.nm { font-family:var(--sans); font-size:15px; font-weight:700; color:var(--ink);
      display:flex; align-items:baseline; gap:6px; }
.nm .m { font-size:11px; color:var(--faint); font-weight:500; }
.meta { font-size:12px; color:var(--muted); margin:2px 0 6px; }
/* 매칭도: 세리프 이탤릭 + 큰 퍼센트 */
.match { font-family:var(--serif); font-style:italic; font-size:13px; color:var(--teal-d);
         font-weight:600; margin:4px 0 9px; }
.match .pct { font-style:normal; font-size:17px; }
.desc { font-size:12.5px; color:#46566F; line-height:1.55; min-height:42px; }
/* 인상 점수 가로 바 */
.bars { display:flex; flex-direction:column; gap:5px; margin-top:9px; }
.bar { display:flex; align-items:center; gap:7px; }
.bar .bt { width:54px; font-size:10px; color:var(--faint); }
.bar .bk { flex:1; height:5px; background:var(--beige); border-radius:100px; overflow:hidden; }
.bar .bf { height:100%; background:linear-gradient(90deg,var(--teal),var(--navy)); }
.bar .bv { width:24px; text-align:right; font-size:10px; color:var(--muted); font-weight:600; }
.barlbl { font-size:10px; color:var(--faint); }
.badge { display:inline-block; background:var(--navy); color:#fff; font-size:11px;
         font-weight:800; padding:2px 9px; border-radius:99px; margin-bottom:6px; }
.rank { display:inline-block; background:var(--beige); color:var(--navy); font-size:12px;
        font-weight:900; padding:2px 10px; border-radius:99px; margin-right:6px; }
.rank.top3 { background:var(--navy); color:#fff; }
.tagnew { display:inline-block; background:var(--teal); color:#fff; font-size:11px;
          font-weight:800; padding:2px 9px; border-radius:99px; margin-bottom:6px; }
/* 검색어 해석 박스 (.parse) */
.parse { font-size:13px; color:var(--muted); background:rgba(82,179,168,.07);
         border:1px solid rgba(82,179,168,.22); border-radius:14px; padding:14px 16px;
         margin-bottom:16px; }
.parse .ptitle { font-family:var(--serif); font-style:italic; color:var(--navy);
                 font-size:14px; margin-bottom:7px; }
.parse .pv { display:inline-block; background:var(--beige); color:var(--teal-d);
             padding:3px 10px; border-radius:100px; font-weight:700; font-size:12px;
             margin:3px 4px 0 0; }
.parse .pdesc { color:var(--ink); line-height:1.6; margin:6px 0; }
/* 검색 입력창 청록 포커스 링 */
.stTextInput input:focus, .stTextArea textarea:focus {
  border-color:var(--teal) !important; box-shadow:0 0 0 4px rgba(82,179,168,.12) !important; }
.chip { display:inline-block; background:var(--beige); color:var(--navy); font-size:11px;
        font-weight:700; padding:3px 10px; border-radius:99px; margin:2px 4px 0 0; }
</style>
""", unsafe_allow_html=True)


def render_brandbar(subtitle: str = "CAST CATCHING"):
    """상단 로고 바. reference/logo.png 있으면 이미지, 없으면 텍스트 로고."""
    b64 = logo_b64()
    if b64:
        st.markdown(f'<div class="brandbar"><img src="data:image/png;base64,{b64}">'
                    f'<span class="tag">{html.escape(subtitle)}</span></div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="brandbar"><span class="logotxt">CATING</span>'
                    f'<span class="tag">{html.escape(subtitle)}</span></div>',
                    unsafe_allow_html=True)

if "applicants" not in st.session_state:
    st.session_state.applicants = []      # 업로드·분석된 지원자(actor 형태 dict)
    st.session_state.app_embs = []        # 각 지원자의 임베딩 벡터
    st.session_state.app_seen = set()     # 이미 처리한 파일 서명(중복 분석 방지)
    st.session_state.shortlist = set()    # 찜한 사람들(uid 집합) — UI 표시용
    st.session_state.feedback = {}        # uid → 'up'/'down' — UI 표시용
    st.session_state.contacted = set()    # 컨택한 사람들(uid 집합) — UI 표시용
    st.session_state.open_cards = set()   # 상세가 펼쳐진 카드(클릭 신호 토글)
    st.session_state.interp_feedback = {} # 검색어 → 'up'/'down' (AI 해석이 맞았는지)
    st.session_state.detail_embs = {}     # uid → 정밀 분석 임베딩(재랭킹용, 한 번 계산 후 재사용)
    st.session_state.detail_info = {}     # uid → 정밀 분석 결과(desc/keywords/traits, 카드 표시용)
    st.session_state.detail_active = set()# '세밀 분석'을 켠 검색 키 모음
    st.session_state.profiles = {}        # user_id → 프로필 dict(role·이름 등) — 로그인 사용자별
    st.session_state.demo_user = None     # 카카오 미설정 시 테스트용 임시 로그인
    st.session_state.pending_role = None  # 프로필 작성 중 고른 역할(감독/배우)
    st.session_state.show_login = False   # 랜딩페이지에서 '시작하기'를 누르면 True(로그인 화면으로)


# ---------- 신호 기록 (DB에 영구 저장) ----------
# 화면 상태(찜됨/투표 등)는 session_state로 즉시 표시하고,
# '정답 신호' 자체는 feedback_db(SQLite)에 차곡차곡 쌓는다. 학습은 아직 끔(기록만).
def _log(search_id, a, etype, rank=None, score=None):
    feedback_db.log_event(search_id, a.get("uid"), a.get("name"), etype, rank=rank, score=score)


# ---------- 카드 렌더 ----------
def pick_face(a, qvec):
    """그 사람 사진 중 '검색어와 가장 닮은' 사진의 b64를 고른다. 정보 없으면 첫 사진."""
    faces = a.get("all_faces_b64")
    embs = a.get("photo_embs")
    if qvec is not None and faces and embs is not None and len(faces) == len(embs):
        sims = np.asarray(embs) @ qvec            # 사진별 코사인 유사도
        return faces[int(np.argmax(sims))]
    return a.get("face_b64")


@st.dialog("배우 상세 정보", width="large")
def actor_detail_dialog(a, qvec):
    """카드의 '상세 보기'로 열리는 큰 화면: 큰 얼굴 + 기본정보 + AI 인상 분석 + 특기 + 사진들."""
    best = pick_face(a, qvec)
    faces = a.get("all_faces_b64") or []
    left, right = st.columns([1, 1.3])
    with left:
        if best:
            st.image(base64.b64decode(best), use_container_width=True)
        else:
            st.markdown(f"<div class='avatar' style='font-size:64px;"
                        f"background:linear-gradient(135deg,#1F3A5F,#4AA9A0)'>"
                        f"{html.escape(a['name'][0])}</div>", unsafe_allow_html=True)
    with right:
        st.markdown(f"### {html.escape(a['name'])}")
        st.markdown(f"**기본정보** · {a['gender']} · {a['age']}세 · "
                    f"{a.get('height_cm') or '?'}cm · "
                    f"{(a.get('voice') or '목소리 미입력')}")
        spec = a.get("specialty")
        st.markdown(f"**특기** · {html.escape(spec) if spec else '지원서에 기재 시 표시'}")
        kw = a.get("keywords") or []
        if kw:
            st.markdown("**인상 태그**")
            st.markdown("".join(f'<span class="chip">{html.escape(t)}</span>' for t in kw),
                        unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("**AI 인상 분석 (0~1 점수)** — 고정된 분석 기준. 피드백으로 바뀌지 않습니다.")
    tr = a.get("traits") or {}
    tcols = st.columns(2)
    shown = [ax for ax in TRAIT_AXES if ax in tr]
    for idx, ax in enumerate(shown):
        with tcols[idx % 2]:
            st.progress(tr[ax], text=f"{ax} {tr[ax]:.2f}")
    if a.get("desc"):
        st.markdown("**인상 묘사**")
        st.write(a["desc"])
    if faces:
        st.markdown(f"**지원서 사진 {len(faces)}장**")
        if qvec is not None and len(faces) > 1:
            st.caption("⭐ = 검색어와 가장 닮아 프로필로 뽑힌 사진")
        pcols = st.columns(min(4, len(faces)))
        for j, b64 in enumerate(faces):
            with pcols[j % len(pcols)]:
                st.image(base64.b64decode(b64), use_container_width=True,
                         caption=("⭐ 프로필" if b64 == best and qvec is not None
                                  and len(faces) > 1 else None))


def render_cards(results, prefix="x", qvec=None, search_id=None, ranked=True):
    cols = st.columns(4)
    for i, (a, score) in enumerate(results):
        c1, c2 = PALETTE.get(a.get("arch", ""), ("#1F3A5F", "#4AA9A0"))
        top = sorted(a.get("traits", {}).items(), key=lambda x: -x[1])[:3]
        bars = ""
        for t, v in top:
            bars += (f'<div class="bar"><span class="bt">{html.escape(t)}</span>'
                     f'<span class="bk"><span class="bf" style="width:{v*100:.0f}%"></span></span>'
                     f'<span class="bv">{v:.1f}</span></div>')
        bars = f'<div class="bars">{bars}</div>' if bars else ''
        # 얼굴 위에 떠 있는 순위 뱃지 (1등은 청록) — 둘러보기(ranked=False)면 숨김
        if ranked:
            rankb = (f'<div class="rankb {"best" if i==0 and score is not None else ""}">'
                     f'{i+1}위</div>')
        else:
            rankb = ''
        if score is not None:
            match = (f'<div class="match">매칭도 '
                     f'<span class="pct">{round(score*100)}%</span></div>')
        elif ranked:
            match = '<div class="match">(점수 없음)</div>'
        else:
            match = ''
        vc = a.get("voice") or "?"
        voice_label = f"{vc}({VOICE_HINT.get(vc, '')})" if vc != "?" else "목소리 미입력"
        profile_b64 = pick_face(a, qvec)
        if profile_b64:
            avatar = f'<div class="avatar"><img src="data:image/png;base64,{profile_b64}"></div>'
        else:
            avatar = (f'<div class="avatar" style="background:linear-gradient(135deg,{c1},{c2})">'
                      f'{html.escape(a["name"][0])}</div>')
        if a.get("is_applicant"):
            pc = a.get("photo_count", 1)
            uc = a.get("used_count", pc)
            label = f"사진 {pc}장" + (f"(분석 {uc}장)" if uc < pc else "")
            newtag = f'<span class="tagnew">지원자 · {label}</span><br>'
        else:
            newtag = ''
        meta_extra = ('<div class="meta" style="color:#C0792E">⚠️ 얼굴 자동검출 실패(전체 이미지)</div>'
                      if a.get("is_applicant") and not a.get("face_found") else '')
        kws = (a.get("keywords") or [])[:3]
        chips = "".join(f'<span class="chip">{html.escape(t)}</span>' for t in kws)
        with cols[i % 4]:
            st.markdown(f"""
            <div class="card {'best' if i==0 and score is not None else ''}">
              {rankb}
              {avatar}
              {newtag}
              <div class="nm">{html.escape(a['name'])}
                <span class="m">{a['gender']} · {a['age']}세 · {a.get('height_cm') or '?'}cm</span></div>
              <div class="meta">{voice_label}</div>
              {meta_extra}
              {match}
              <div class="desc">{html.escape(a['desc'])}</div>
              <div style="margin-top:8px">{chips}</div>
              {bars}
            </div>
            """, unsafe_allow_html=True)
            # ----- 정답 신호 버튼 (찜·컨택·👍·👎) → 화면 표시 + DB 기록 -----
            uid = a.get("uid") or f"demo{a.get('id','')}"
            k = f"{prefix}_{uid}_{i}"
            rank = i + 1                      # 그 검색에서 보여준 순위(1=맨 위)
            faved = uid in st.session_state.shortlist
            contacted = uid in st.session_state.contacted
            vote = st.session_state.feedback.get(uid)
            b1, b2, b3, b4 = st.columns(4)
            with b1:
                if st.button("💛" if faved else "🤍", key=f"fav_{k}",
                             use_container_width=True, help="찜(관심 후보로 담기)"):
                    if faved:
                        st.session_state.shortlist.discard(uid)
                    else:
                        st.session_state.shortlist.add(uid)
                        _log(search_id, a, "shortlist", rank, score)
                    st.rerun()
            with b2:
                if st.button("📞" if not contacted else "📞✓", key=f"con_{k}",
                             use_container_width=True, help="컨택(섭외 의사 — 가장 강한 신호)"):
                    if contacted:
                        st.session_state.contacted.discard(uid)
                    else:
                        st.session_state.contacted.add(uid)
                        _log(search_id, a, "contact", rank, score)
                    st.rerun()
            with b3:
                if st.button("👍" + ("✓" if vote == "up" else ""), key=f"up_{k}",
                             use_container_width=True, help="이 결과가 잘 맞아요(랭킹 학습용)"):
                    st.session_state.feedback[uid] = "up"
                    _log(search_id, a, "up", rank, score)
                    st.rerun()
            with b4:
                if st.button("👎" + ("✓" if vote == "down" else ""), key=f"down_{k}",
                             use_container_width=True, help="이 결과는 안 맞아요(랭킹 학습용)"):
                    st.session_state.feedback[uid] = "down"
                    _log(search_id, a, "down", rank, score)
                    st.rerun()

            # '상세 보기' = 큰 화면 모달 + 클릭(암묵적 관심) 신호 기록
            if st.button("🔎 상세 보기", key=f"det_{k}", use_container_width=True):
                _log(search_id, a, "click", rank, score)
                actor_detail_dialog(a, qvec)


# ---------- 한 사람(사진 여러 장) 분석 ----------
def analyze_person_faces(face_pngs: list[bytes], detailed: bool = False) -> dict:
    """동일 인물의 여러 사진을 한 번에 보고, 세밀한 인상을 묘사한다.
    detailed=True면 '정밀 모드' — 눈매·표정 등 인상에 직결되는 부분을 한 단계 더 자세히 본다."""
    client = load_vision_client()
    n = len(face_pngs)
    axes = ", ".join(TRAIT_AXES)
    detail_clause = (
        "■ 정밀 관찰 모드: 이번엔 평소보다 한 단계 더 깊이 본다. "
        "눈매(쌍꺼풀 유무·눈꼬리 각도·눈동자 크기·눈 사이 간격), 눈빛과 시선이 주는 느낌, "
        "미간·눈썹의 모양과 짙기, 입매의 미세한 올라감/다묾, 표정에서 풍기는 감정의 결, "
        "광대·턱선·이목구비 비율의 미묘한 균형까지 세밀하게 관찰해 desc에 녹이고, "
        "traits 점수도 더 신중하게 0.02 단위로 미세하게 구분해라.\n"
    ) if detailed else ""
    prompt = (
        f"아래 {n}장은 모두 '동일 인물'의 사진입니다(여러 각도/표정). "
        "한 사람으로 보고 종합해, 캐스팅 감독을 위한 첫인상을 분석해줘.\n"
        + detail_clause +
        "■ 가장 중요한 규칙: **이 사람이 누구인지 전혀 모른다고 가정**하고, "
        "오직 '얼굴 사진에서 직접 보이는 인상'만으로 묘사하라. 사람이 처음 보는 낯선 얼굴을 "
        "보고 즉각 느끼는 인상처럼. 설령 아는 얼굴 같아도 그 사람의 이름·직업·연예인 여부·"
        "평소 이미지·작품·평판 같은 배경지식은 **절대 떠올리지 말고 쓰지 마라.** "
        "눈앞의 눈매·표정·이목구비·전체 분위기에서 보이는 것만 말하라.\n"
        "- desc: 사람이 얼굴을 볼 때처럼, **생김새와 분위기를 따로 나누지 말고 하나의 종합 "
        "인상으로** 자연스럽게 녹여 3~4문장(한국어)으로 묘사해라. "
        "생김새 특징(눈매·쌍꺼풀·눈동자, 콧대, 입매, 턱선, 광대, 얼굴형, 이목구비 비율, "
        "피부톤 등)이 어떤 분위기(시크함·청순함·강렬함·부드러움 등)를 만들어내는지가 "
        "한 호흡으로 드러나게. 예: '또렷한 쌍꺼풀과 곧게 뻗은 콧대, 갸름한 턱선이 모여 "
        "차갑고 도회적인 인상을 만든다'처럼 생김새→분위기가 한 문장 안에 이어지게. "
        "비슷한 사람과 구분되도록 미묘한 차이까지 구체적으로. "
        "'터프하고 든든한' 같은 흔한 상투구는 피하고, 실존 연예인 이름·배경지식 사용 금지.\n"
        "- keywords: 이 인물을 구분 짓는 핵심 키워드 5~8개. "
        "생김새 특징과 분위기 단어를 **섞어서**(예: 둥근얼굴형, 짙은눈썹, 시크함, 청순함).\n"
        f"- traits: 다음 인상 축을 각각 0.0~1.0으로 채점한 객체. 축: {axes}\n"
        "  채점 눈금(반드시 이 기준으로, 0.5에 몰지 말 것): "
        "0.1 거의 없음 · 0.3 약함 · 0.5 보통 · 0.7 뚜렷함 · 0.9 매우 강함. "
        "소수 둘째 자리까지, 0.72와 0.88처럼 미세하게 구분해라.\n"
        f"- per_photo: 사진별 인상을 짧게(각 1문장) 적은 배열. **사진 순서대로 정확히 {n}개**. "
        "사진마다 표정·각도·분위기가 다르면 그 차이가 드러나게.\n"
        "- gender: '남' 또는 '여'\n"
        "- age: 추정 나이(정수)\n"
        '반드시 JSON으로만: {"desc":"...","keywords":["..."],'
        '"traits":{"남성성":0.0},"per_photo":["..."],"gender":"남","age":28}'
    )
    content = [{"type": "text", "text": prompt}]
    for png in face_pngs:
        b64 = base64.b64encode(png).decode()
        content.append({"type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}", "detail": "high"}})
    r = _chat_json(client, [{"role": "user", "content": content}])
    return json.loads(r.choices[0].message.content)


@st.cache_data(show_spinner=False)
def expand_query(q: str) -> dict:
    """검색 문장을 AI가 '세밀한 인상 묘사'로 풀어 쓴다.
    레퍼런스 배우/작품('김우빈의 살목지에서의 분위기' 등)도 그 인물·배역의
    구체적 인상으로 해석한다. 이 결과를 임베딩해 top-k를 매긴다."""
    ckey = "q::" + q.strip()
    cached = _analysis_cache().get(ckey)
    if cached is not None:
        return cached
    client = load_vision_client()
    axes = ", ".join(TRAIT_AXES)
    prompt = (
        "너는 캐스팅 감독을 돕는 인상 분석가다. 아래 검색 문장을 받아, "
        "그 안에 담긴 '원하는 배우의 얼굴 인상'을 최대한 구체적으로 풀어 써라.\n"
        "■ 가장 중요한 규칙: **오직 '얼굴에서 보이는 인상'만으로** 묘사한다. "
        "사람이 처음 보는 얼굴을 보고 즉각 느끼는 첫인상처럼.\n"
        "- 실존 배우/연예인 이름이 들어오면(예: '한소희 같은 분위기'), 그 사람의 "
        "**평소 이미지·연기 경력·작품 속 배역·성격·대중적 평판 같은 배경지식은 절대 떠올리지 마라.** "
        "대신 '그 이름이 일반적으로 연상시키는 얼굴 생김새와 인상'만 중립적인 인상 묘사로 변환하라. "
        "(이름은 해석에만 쓰고, desc·keywords 결과 텍스트에는 이름을 절대 남기지 마라.)\n"
        "- 작품·배역 이름이 들어와도 줄거리·캐릭터 서사가 아니라, 그 배역에서 '얼굴이 주는 인상'만 묘사하라.\n"
        "- desc: 사람이 얼굴을 떠올릴 때처럼 **생김새와 분위기를 하나로 녹여** 2~3문장(한국어)으로. "
        "원하는 눈매·콧대·입매·턱선·얼굴형·이목구비 비율 같은 생김새가 어떤 분위기를 "
        "만드는지가 한 호흡으로 드러나게.\n"
        "- keywords: 핵심 키워드 6~10개(생김새 특징과 분위기 단어를 섞어서. 이름·작품명 금지).\n"
        f"- traits: 원하는 인상의 강도를 다음 축마다 0.0~1.0으로 채점한 객체. 축: {axes}. "
        "강하게 원하는 축만 높게(0.7~0.95), 무관한 축은 낮게(0.0~0.3).\n"
        f'검색 문장: "{q}"\n'
        '반드시 JSON으로만: {"desc":"...","keywords":["..."],"traits":{"남성성":0.0}}'
    )
    r = _chat_json(client, [{"role": "user", "content": prompt}])
    data = json.loads(r.choices[0].message.content)
    desc = (data.get("desc") or "").strip()
    keywords = data.get("keywords") or []
    raw_traits = data.get("traits") or {}
    traits = {}
    for ax in TRAIT_AXES:
        v = raw_traits.get(ax)
        if isinstance(v, (int, float)):
            traits[ax] = max(0.0, min(1.0, float(v)))
    # 지원자 임베딩 텍스트와 같은 형식("강한 인상: ...")을 검색문에도 더해 대칭 매칭 → 정밀도↑
    strong = [ax for ax in TRAIT_AXES if traits.get(ax, 0) >= 0.65]
    parts = [desc]
    if keywords:
        parts.append(", ".join(keywords))
    if strong:
        parts.append("강한 인상: " + ", ".join(strong))
    expanded = " / ".join(parts)
    out = {"desc": desc, "keywords": keywords, "traits": traits, "expanded": expanded}
    _cache_put(ckey, out)
    return out


def process_person(person_name: str, members: list[tuple[str, bytes]]) -> dict | None:
    """한 사람의 파일 여러 개(사진 5장 등)를 묶어 분석한다."""
    face_pngs, texts, any_face = [], [], False
    for fname, fbytes in members:
        images, text = intake.extract_images_and_text(fbytes, fname)
        if text:
            texts.append(text)
        if not images:
            continue
        crop, src = intake.find_best_face(images)
        if crop is not None:
            any_face = True
        face_pngs.append(intake.to_png_bytes(crop if crop is not None else src))
    if not face_pngs:
        return {"error": "이미지를 찾지 못했어요", "name": person_name}

    total_photos = len(face_pngs)
    # 기본은 있는 사진 전부 분석. 안전 천장(30장)을 넘을 때만 고르게 샘플(요청 과부하 방지).
    used = face_pngs
    if total_photos > MAX_FACES_PER_PERSON:
        idx = sorted(set(np.linspace(0, total_photos - 1, MAX_FACES_PER_PERSON).round().astype(int)))
        used = [face_pngs[i] for i in idx]

    full_text = "\n".join(texts)
    # 같은 사진이면 저장된 분석을 재사용(결과 고정 + 비용 절감). 처음 보는 사진만 AI 호출.
    key = _cache_key(members)
    vis = _analysis_cache().get(key)
    if vis is None:
        vis = analyze_person_faces(used)                  # 비전 AI: 그 사람 사진 종합
        _cache_put(key, vis)
    age_text = intake.parse_age(full_text)               # 지원서 글자의 나이(우선)
    height_text = intake.parse_height(full_text)         # 키는 글자에서만(사진 불가)
    specialty = intake.parse_specialty(full_text)        # 특기(지원서 글자에 있을 때만)
    desc = vis.get("desc", "").strip()
    keywords = vis.get("keywords", []) or []
    raw_traits = vis.get("traits", {}) or {}
    # 정해둔 축만, 0~1로 범위 제한해서 보관(모델이 벗어난 값/엉뚱한 키를 줄 때 대비)
    traits = {}
    for ax in TRAIT_AXES:
        v = raw_traits.get(ax)
        if isinstance(v, (int, float)):
            traits[ax] = max(0.0, min(1.0, float(v)))
    # 임베딩에는 묘사 + 키워드 + '강한 인상 축'을 함께 넣어 미세 구분력을 높인다.
    # 예: 남성성 0.85 → "남성적인 인상" 을 문장에 더해 '남성적' 검색에 더 잘 걸리게.
    strong = [ax for ax in TRAIT_AXES if traits.get(ax, 0) >= 0.65]
    trait_phrase = ("강한 인상: " + ", ".join(strong)) if strong else ""
    parts = [desc]
    if keywords:
        parts.append(", ".join(keywords))
    if trait_phrase:
        parts.append(trait_phrase)
    embed_text = " / ".join(parts)
    emb = embedder.encode([embed_text])[0]

    # 분석에 쓴 사진(used)별 임베딩 — 검색어와 가장 닮은 사진을 프로필로 고르기 위함
    used_b64 = [base64.b64encode(p).decode() for p in used]
    per_photo = vis.get("per_photo") or []
    photo_embs = None
    if isinstance(per_photo, list) and len(per_photo) == len(used):
        photo_embs = embedder.encode([(t or desc) for t in per_photo])  # (사진수 × 차원)

    import uuid
    return {
        "actor": {
            "uid": uuid.uuid4().hex,
            "name": person_name[:20],
            "gender": vis.get("gender", "?"),
            "age": age_text if age_text is not None else vis.get("age", 0),
            "height_cm": height_text,                    # 글자에 없으면 None
            "voice": None,
            "specialty": specialty,                      # 특기(글자에 없으면 None)
            "arch": "",
            "traits": traits,
            "desc": desc,
            "keywords": keywords,
            "photo_count": total_photos,
            "used_count": len(used),
            "face_b64": used_b64[0],
            "all_faces_b64": used_b64,             # 분석에 쓴 사진들(프로필 후보)
            "photo_embs": photo_embs,              # 사진별 임베딩(없으면 None)
            "is_applicant": True,
            "face_found": any_face,
        },
        "emb": emb,
    }


CURRENT_YEAR = 2026   # 출생연도 → 나이 자동 계산 기준


def process_actor_self(name: str, gender: str, age, height, specialty,
                       image_files, extra: dict | None = None) -> dict | None:
    """배우 본인이 직접 올린 정보(성별·사진·특기 등)로 프로필을 만든다.
    - 얼굴 인상(desc/keywords/traits)은 사진을 AI가 분석해 뽑는다(감독 업로드와 동일한 자(ruler)).
    - 성별·나이·키·특기는 '본인이 입력한 값'을 우선한다(본인이 제일 정확하므로).
    - extra: 거주지·몸무게·연락처·언어·학력·경력·영상링크 등 추가 입력(있으면 합쳐 저장).
      ※ 경력은 '표시·약한 신호용'일 뿐 검색 임베딩(embed_text)에는 넣지 않는다(신인 공정성).
    감독이 검색하는 같은 지원자 풀(session_state.applicants)에 합류한다."""
    extra = extra or {}
    members = [(f.name, f.getvalue()) for f in image_files]
    face_pngs, any_face = [], False
    for fname, fbytes in members:
        images, _ = intake.extract_images_and_text(fbytes, fname)
        if not images:
            continue
        crop, src = intake.find_best_face(images)
        if crop is not None:
            any_face = True
        face_pngs.append(intake.to_png_bytes(crop if crop is not None else src))
    if not face_pngs:
        return {"error": "얼굴이 있는 사진을 한 장 이상 올려주세요", "name": name}

    total_photos = len(face_pngs)
    used = face_pngs
    if total_photos > MAX_FACES_PER_PERSON:
        idx = sorted(set(np.linspace(0, total_photos - 1, MAX_FACES_PER_PERSON).round().astype(int)))
        used = [face_pngs[i] for i in idx]

    # 같은 사진이면 저장된 분석 재사용(결과 고정 + 비용 절감)
    key = _cache_key(members)
    vis = _analysis_cache().get(key)
    if vis is None:
        vis = analyze_person_faces(used)
        _cache_put(key, vis)

    desc = vis.get("desc", "").strip()
    keywords = vis.get("keywords", []) or []
    raw_traits = vis.get("traits", {}) or {}
    traits = {}
    for ax in TRAIT_AXES:
        v = raw_traits.get(ax)
        if isinstance(v, (int, float)):
            traits[ax] = max(0.0, min(1.0, float(v)))
    strong = [ax for ax in TRAIT_AXES if traits.get(ax, 0) >= 0.65]
    trait_phrase = ("강한 인상: " + ", ".join(strong)) if strong else ""
    parts = [desc]
    if keywords:
        parts.append(", ".join(keywords))
    if trait_phrase:
        parts.append(trait_phrase)
    embed_text = " / ".join(parts)
    emb = embedder.encode([embed_text])[0]

    used_b64 = [base64.b64encode(p).decode() for p in used]
    per_photo = vis.get("per_photo") or []
    photo_embs = None
    if isinstance(per_photo, list) and len(per_photo) == len(used):
        photo_embs = embedder.encode([(t or desc) for t in per_photo])

    import uuid
    # 본인이 입력한 값을 우선(성별/나이/키/특기). 비워둔 건 AI 분석값/None으로.
    gender_val = gender if gender in ("남", "여") else vis.get("gender", "?")
    # 나이: 출생연도가 있으면 자동 계산, 없으면 직접 입력값, 그것도 없으면 AI 추정.
    birth_year = extra.get("birth_year")
    if birth_year:
        age_val = max(0, CURRENT_YEAR - int(birth_year))
    elif age:
        age_val = int(age)
    else:
        age_val = vis.get("age", 0)
    actor = {
        "uid": uuid.uuid4().hex,
        "name": (name or "이름미입력")[:20],
        "gender": gender_val,
        "age": age_val,
        "height_cm": int(height) if height else None,
        "voice": None,
        "specialty": (specialty or "").strip() or None,
        "arch": "",
        "traits": traits,
        "desc": desc,
        "keywords": keywords,
        "photo_count": total_photos,
        "used_count": len(used),
        "face_b64": used_b64[0],
        "all_faces_b64": used_b64,
        "photo_embs": photo_embs,
        "is_applicant": True,
        "self_registered": True,            # 배우 본인이 직접 등록한 프로필 표시
        "face_found": any_face,
        # 추가 입력값(거주지·몸무게·연락처·언어·학력·경력·영상링크 등) — 표시·필터·연락용
        "birth_year": birth_year,
        "region": extra.get("region"),
        "weight_kg": extra.get("weight_kg"),
        "phone": extra.get("phone"),
        "email": extra.get("email"),
        "languages": extra.get("languages"),
        "education": extra.get("education"),
        "career": extra.get("career"),       # 검색 임베딩엔 미반영(신인 공정성)
        "video_link": extra.get("video_link"),
    }
    return {"actor": actor, "emb": emb}


def render_search_controls(prefix: str):
    """검색창 + 필터 + top-k 슬라이더. 두 탭이 공유(prefix로 위젯 키 구분)."""
    query = st.text_input("원하는 분위기 (문장)",
                          placeholder="예: 시크하고 카리스마 있는 도시적인 사람 / 김우빈의 살목지에서의 분위기",
                          key=f"{prefix}_q")
    expand = st.toggle("🔎 검색어가 어떤 분위기인지 AI가 해석해서 검색",
                       value=True, key=f"{prefix}_expand",
                       help="‘한소희 같은 분위기’, ‘김우빈의 살목지에서의 분위기’ 같은 말을 "
                            "구체적 인상 묘사로 풀어 보여주고, 그 해석으로 top-k를 매깁니다. "
                            "끄면 입력한 문장 그대로 검색합니다.")
    with st.container(border=True):
        c1, c2 = st.columns(2)
        with c1:
            gender = st.segmented_control("성별", ["전체", "남", "여"], default="전체", key=f"{prefix}_g")
        with c2:
            age_mode = st.segmented_control("나이 방식", ["연령대", "직접 범위"],
                                            default="연령대", key=f"{prefix}_am")
        if age_mode == "직접 범위":
            age_min, age_max = st.slider("나이 범위", 15, 60, (20, 35), key=f"{prefix}_ar")
        else:
            band = st.segmented_control("연령대", ["전체", "10대", "20대", "30대", "40대+"],
                                        default="전체", selection_mode="multi", key=f"{prefix}_band")
            age_min, age_max = None, None
            if band and "전체" not in band:
                lows, highs = [], []
                for b in band:
                    if b == "10대": lows.append(10); highs.append(19)
                    elif b == "20대": lows.append(20); highs.append(29)
                    elif b == "30대": lows.append(30); highs.append(39)
                    elif b == "40대+": lows.append(40); highs.append(120)
                age_min, age_max = min(lows), max(highs)
        c3, c4 = st.columns(2)
        with c3:
            height_min, height_max = st.slider("키 범위 (cm)", 150, 195, (150, 195), key=f"{prefix}_h")
        with c4:
            voice = st.multiselect("목소리", VOICES,
                                   format_func=lambda v: f"{v} ({VOICE_HINT[v]})", key=f"{prefix}_v")
        topk = st.slider("최대 결과 수 (top-k) — 분위기 점수 상위 N명", 4, 50, 12, key=f"{prefix}_k")
    filters = {"gender": None if gender == "전체" else gender,
               "age_min": age_min, "age_max": age_max,
               "height_min": height_min if height_min > 150 else None,
               "height_max": height_max if height_max < 195 else None,
               "voice": voice or None}
    return query, filters, topk, expand


def resolve_query(query: str, expand: bool) -> tuple[str, bool]:
    """검색에 실제로 쓸 문장을 돌려준다.
    expand가 켜지고 검색어가 있으면 AI 해석문으로 바꾸고, 그 해석을 화면에 보여준다.
    반환: (검색에 쓸 문장, 해석을 화면에 보여줬는지 여부)."""
    if not (expand and query.strip()):
        return query, False
    with st.spinner("검색어를 AI가 해석하는 중…"):
        try:
            res = expand_query(query.strip())
        except Exception as e:
            st.warning(f"검색어 해석 실패({e}) — 원문 그대로 검색합니다.")
            return query, False
    # 검색 문장에서 계산된 인상 수치(traits)를 막대로 — 0보다 큰 축을 높은 순으로 표시
    tr = res.get("traits") or {}
    top = sorted((kv for kv in tr.items() if kv[1] > 0), key=lambda x: -x[1])[:6]
    bars = ""
    for t, v in top:
        strong = ' · 강한 인상' if v >= 0.65 else ''
        bars += (f'<div class="bar"><span class="bt">{html.escape(t)}</span>'
                 f'<span class="bk"><span class="bf" style="width:{v*100:.0f}%"></span></span>'
                 f'<span class="bv">{v:.2f}</span></div>')
    bars_block = (f'<div class="ptitle" style="margin-top:12px;margin-bottom:6px">'
                  f'이 문장의 인상 수치 (0~1, 검색에 실제로 반영됨)</div>'
                  f'<div class="bars">{bars}</div>') if bars else ""
    chips = "".join(f'<span class="pv">{html.escape(t)}</span>' for t in res["keywords"])
    st.markdown(f"""<div class="parse">
      <div class="ptitle">“{html.escape(query.strip())}” 를 이런 느낌으로 풀었어요</div>
      <div class="pdesc">{html.escape(res['desc'])}</div>
      <div style="margin-top:6px">{chips}</div>
      {bars_block}
    </div>""", unsafe_allow_html=True)
    return res["expanded"], True


def render_interp_feedback(search_id, query: str, prefix: str):
    """AI 해석이 원하는 느낌과 맞는지 👍/👎 받기. (검색 순위와 무관 — 해석 품질 신호로 기록)"""
    fb = st.session_state.interp_feedback.get(query)
    st.caption("이 해석이 원하던 느낌과 맞나요? (해석 품질을 기록해 두면 나중에 검색어 풀이를 개선해요)")
    b1, b2, _ = st.columns([1.2, 1.2, 5])
    with b1:
        if st.button("👍 잘 맞아요" + (" ✓" if fb == "up" else ""),
                     key=f"interp_up_{prefix}", use_container_width=True):
            st.session_state.interp_feedback[query] = "up"
            feedback_db.log_event(search_id, None, "(검색어 해석)", "interp_up")
            st.rerun()
    with b2:
        if st.button("👎 좀 달라요" + (" ✓" if fb == "down" else ""),
                     key=f"interp_down_{prefix}", use_container_width=True):
            st.session_state.interp_feedback[query] = "down"
            feedback_db.log_event(search_id, None, "(검색어 해석)", "interp_down")
            st.rerun()
    if fb:
        st.caption("✅ 해석 피드백이 기록됐어요. 아래는 이 해석으로 찾은 결과입니다.")


# ---------- 세밀 분석(상위 20명 정밀 재랭킹) ----------
def _embed_from_analysis(info: dict):
    """정밀 분석 결과(desc/keywords/traits)를 process_person과 똑같은 방식으로 임베딩한다.
    (검색어 임베딩과 같은 형식이라 코사인 비교가 대칭적으로 맞음)"""
    desc = (info.get("desc") or "").strip()
    keywords = info.get("keywords") or []
    traits = {}
    for ax in TRAIT_AXES:
        v = (info.get("traits") or {}).get(ax)
        if isinstance(v, (int, float)):
            traits[ax] = max(0.0, min(1.0, float(v)))
    strong = [ax for ax in TRAIT_AXES if traits.get(ax, 0) >= 0.65]
    parts = [desc]
    if keywords:
        parts.append(", ".join(keywords))
    if strong:
        parts.append("강한 인상: " + ", ".join(strong))
    return embedder.encode([" / ".join(parts)])[0], traits


def render_detail_rerank(apps, app_emb, filters, search_text, qvec, prefix, search_id):
    """기본 검색 상위 20명만 정밀 재분석해 재정렬한다. 비용은 그 20명에만 든다."""
    if qvec is None or not (search_text or "").strip():
        return
    flag = f"{prefix}::{search_text}"
    st.divider()
    if flag not in st.session_state.detail_active:
        st.markdown("##### 🔬 더 정확히 보고 싶다면")
        st.caption("아래 버튼을 누르면 **기본 검색 상위 20명만** 얼굴을 한 단계 더 정밀하게 다시 "
                   "분석(눈매·표정 등)해서 그 20명의 순위를 다시 매깁니다. **비용은 그 20명에만** 듭니다.")
        if st.button("🔬 상위 20명 세밀 분석 후 재정렬", key=f"detailgo_{prefix}",
                     type="primary"):
            st.session_state.detail_active.add(flag)
            st.rerun()
        return

    # 기본 점수 상위 20명 (슬라이더와 무관하게 항상 20명 기준)
    top20 = search_filtered(search_text, embedder, apps, app_emb, filters, k=20)
    if not top20:
        st.info("재정렬할 후보가 없어요.")
        return
    base_rank = {a["uid"]: idx + 1 for idx, (a, _) in enumerate(top20)}

    # 아직 정밀 분석 안 한 사람만 새로 분석(나머지는 저장분 재사용 → 비용·시간 절약)
    todo = [a for a, _ in top20 if a["uid"] not in st.session_state.detail_embs]
    if todo:
        prog = st.progress(0.0, text=f"상위 {len(top20)}명 중 {len(todo)}명 정밀 분석 중…")
        for n, a in enumerate(todo):
            pngs = [base64.b64decode(b) for b in (a.get("all_faces_b64") or [])]
            if not pngs and a.get("face_b64"):
                pngs = [base64.b64decode(a["face_b64"])]
            try:
                info = analyze_person_faces(pngs, detailed=True)
            except Exception as e:
                info = {"desc": a.get("desc", ""), "keywords": a.get("keywords", []),
                        "traits": a.get("traits", {}), "_error": str(e)}
            emb, traits = _embed_from_analysis(info)
            st.session_state.detail_embs[a["uid"]] = emb
            st.session_state.detail_info[a["uid"]] = {**info, "traits": traits}
            prog.progress((n + 1) / len(todo), text=f"정밀 분석 {n + 1}/{len(todo)}")
        prog.empty()

    # 정밀 임베딩으로 같은 검색어와의 유사도 재계산 → 재정렬
    rescored = []
    for a, _ in top20:
        demb = st.session_state.detail_embs[a["uid"]]
        rescored.append((a, float(np.asarray(demb) @ qvec)))
    rescored.sort(key=lambda x: -x[1])

    st.markdown("##### 🔬 세밀 재분석 결과 (상위 20명 재정렬)")
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        st.caption("같은 검색어로, 정밀 분석한 인상끼리 다시 비교해 순위를 매겼습니다. "
                   "카드의 ‘기본 N위 → 세밀 M위’로 순위 변화를 볼 수 있어요.")
    with cc2:
        if st.button("기본 결과로 닫기", key=f"detailclose_{prefix}"):
            st.session_state.detail_active.discard(flag)
            st.rerun()

    # 순위가 많이 오른 사람 요약(눈에 띄는 변화)
    movers = []
    for newi, (a, _) in enumerate(rescored, start=1):
        delta = base_rank[a["uid"]] - newi
        if delta >= 3:
            movers.append(f"**{html.escape(a['name'])}** 기본 {base_rank[a['uid']]}위→세밀 {newi}위 (▲{delta})")
    if movers:
        st.success("📈 크게 오른 후보: " + " · ".join(movers[:6]))

    # 정밀 분석된 묘사/수치로 카드 갱신 + 순위 변화 메모를 desc 앞에 표시
    detailed_results = []
    for newi, (a, s) in enumerate(rescored, start=1):
        info = st.session_state.detail_info.get(a["uid"], {})
        move = base_rank[a["uid"]] - newi
        arrow = (f"기본 {base_rank[a['uid']]}위 → 세밀 {newi}위 "
                 + ("▲" + str(move) if move > 0 else ("▼" + str(-move) if move < 0 else "─")))
        a2 = {**a,
              "desc": f"[{arrow}] " + (info.get("desc") or a.get("desc", "")),
              "keywords": info.get("keywords") or a.get("keywords", []),
              "traits": info.get("traits") or a.get("traits", {})}
        detailed_results.append((a2, s))
    render_cards(detailed_results, prefix=f"{prefix}_dtl", qvec=qvec, search_id=search_id)


def show_results(query: str, results, subject: str = "후보", prefix: str = "x",
                 qvec=None, search_id=None, total=None, k=None, pool=None):
    """점수 기준을 명확히 보여주며 결과 카드를 렌더한다.
    total = 필터 통과한 후보 수, k = 요청한 top-k, pool = 검색 대상 전체 수."""
    shown = len(results)
    # ── Top-K 근거를 한눈에: 전체 → 필터 통과 → 요청 top-k → 실제 표시 ──
    if k is not None and total is not None:
        pool_txt = f"전체 {pool}명 → " if pool is not None and pool != total else ""
        st.markdown(f"**{pool_txt}필터 통과 {total}명 → 상위 {min(k, total)}명 표시** "
                    f"(top-k 슬라이더 = {k})")
        if total <= k:
            st.caption(f"💡 필터 통과 후보가 {total}명뿐이라 **전원 표시**됩니다. "
                       f"슬라이더를 {total}보다 낮추면(예: {max(1, total-1)}) 상위 몇 명만 추려서 봐요. "
                       f"후보가 top-k보다 많아질 때부터 슬라이더로 잘라내는 효과가 보입니다.")
    if query.strip():
        st.caption("정렬 기준 = 검색 문장과 각 인물의 **종합 인상 묘사(생김새+분위기)** 사이 "
                   "**의미 유사도(코사인)**. 카드의 **매칭도 %**가 그 점수이고, 이 점수가 높은 "
                   "순서대로 위에서부터 나열됩니다. **top-k = 그 상위 몇 명까지 보여줄지**입니다.")
    else:
        st.warning("검색어가 비어 있어 **점수를 매길 수 없습니다.** "
                   "지금은 필터 통과 순서대로만 보여줍니다 — 문장을 입력하면 점수가 생기고 "
                   "top-k가 '점수 상위 N명'으로 동작합니다.")
    render_cards(results, prefix=prefix, qvec=qvec, search_id=search_id)
    if not results:
        st.info(f"조건에 맞는 {subject}가 없어요. 필터를 넓혀보세요.")


def render_feedback_panel():
    """쌓인 피드백(정답 신호)을 눈으로 확인하는 패널. 학습은 아직 끔 — 기록만."""
    s = feedback_db.stats()
    title = (f"📊 쌓인 피드백  ·  검색어 {s['distinct_queries']}종  ·  "
             f"신호 {s['events']}건")
    with st.expander(title, expanded=False):
        st.caption("지금은 **학습을 켜지 않고 기록만** 합니다(얼굴 인상 점수는 절대 안 바뀜). "
                   "신호가 충분히 쌓이면, 이 기록으로 '검색 결과 순서'만 똑똑하게 재조정합니다.")
        if s["by_type"]:
            order = ["contact", "up", "shortlist", "click", "down"]
            label = {"contact": "📞 컨택", "up": "👍 좋아요", "shortlist": "💛 찜",
                     "click": "📷 자세히(클릭)", "down": "👎 별로"}
            cols = st.columns(len(order))
            for col, t in zip(cols, order):
                col.metric(label[t], s["by_type"].get(t, 0))
        rows = feedback_db.recent_events(15)
        if rows:
            st.caption("최근 신호 (검색어 → 대상 · 신호 · 순위 · 점수)")
            st.dataframe(
                [{"시각": r[0], "검색어": r[1], "대상": r[2], "신호": r[3],
                  "순위": r[4], "점수": (round(r[5] * 100) if r[5] is not None else None)}
                 for r in rows],
                use_container_width=True, hide_index=True)
        else:
            st.info("아직 신호가 없어요. 결과 카드의 💛📞👍👎 또는 ‘자세히’를 눌러보세요.")


def process_uploads(files):
    """업로드 파일들을 사람 단위로 묶어 분석하고 session_state에 저장한다."""
    jobs = []  # (표시이름, 내용바이트) — zip은 펼쳐서 평탄화
    for f in files:
        data = f.getvalue()
        if f.name.lower().endswith(".zip"):
            inner = intake.expand_zip(data)
            if not inner:
                st.warning(f"{f.name}: 압축 안에 분석할 파일(PDF/이미지/DOCX)이 없어요")
            jobs.extend(inner)
        else:
            jobs.append((f.name, data))

    # 파일 이름 앞부분으로 동일 인물끼리 묶기 (예: 김민준_1~5 → 김민준)
    groups: dict[str, list] = {}
    for name, data in jobs:
        groups.setdefault(intake.person_key(name), []).append((name, data))

    preview = ", ".join(f"{p}({len(m)}장)" for p, m in list(groups.items())[:20])
    st.caption(f"📂 업로드 파일 {len(jobs)}개 → **인물 {len(groups)}명**으로 묶음 · {preview}"
               + (" …" if len(groups) > 20 else ""))

    for person, members in groups.items():
        sig = hashlib.md5(("|".join(sorted(hashlib.md5(d).hexdigest()
                                           for _, d in members))).encode()).hexdigest()
        if sig in st.session_state.app_seen:
            continue
        with st.spinner(f"분석 중: {person} (사진 {len(members)}장)"):
            try:
                out = process_person(person, members)
            except Exception as e:
                st.error(f"{person} 처리 실패: {e}")
                st.session_state.app_seen.add(sig)
                continue
        st.session_state.app_seen.add(sig)
        if out is None or "error" in out:
            st.warning(f"{person}: {out.get('error','분석 실패') if out else '분석 실패'}")
            continue
        st.session_state.applicants.append(out["actor"])
        st.session_state.app_embs.append(out["emb"])


# ============ 화면별 본문 ============

def screen_search():
    """① 배우 탐색 — 자연어 검색 + 필터 + 진짜 엔진 결과 카드."""
    render_brandbar("배우 탐색")
    apps = st.session_state.applicants
    if not apps:
        st.info("아직 분석된 지원자가 없습니다. 왼쪽 메뉴의 **📤 지원서 업로드**에서 "
                "지원서를 올리면, 여기에서 검색할 수 있어요.")
        return

    query, filters, topk, expand = render_search_controls("flow")
    st.markdown(f"###### 분석된 지원자 {len(apps)}명 중에서 검색합니다")

    if not (query or "").strip():
        # 검색어가 없으면 → 모든 지원자를 '최신순(나중에 올린 사람부터)'으로 둘러보기
        browse = [a for a in reversed(apps) if passes_filters(a, filters)]
        st.info("💡 검색창에 원하는 분위기를 문장으로 적으면 매칭도 순으로 정렬돼요. "
                "예) “청량하고 청순한 첫사랑 느낌”. 아래는 **전체 지원자(최신순)**입니다.")
        if not browse:
            st.warning("필터 조건에 맞는 지원자가 없습니다. 필터를 풀어보세요.")
            return
        st.markdown(f"###### 전체 지원자 {len(browse)}명 · 최신순")
        render_cards([(a, None) for a in browse], prefix="browse",
                     qvec=None, search_id=None, ranked=False)
        return

    # 1) 검색어를 AI가 어떤 느낌으로 해석했는지 먼저 보여주고 → 2) 그 해석 피드백 → 3) 결과
    search_text, shown = resolve_query(query, expand)
    sid = feedback_db.get_or_create_search(query or "", search_text)
    if shown:
        render_interp_feedback(sid, query.strip(), "flow")
    qvec = embedder.encode([search_text])[0] if search_text.strip() else None
    feedback_db.set_search_embedding(sid, qvec)   # 검색의 '의미 좌표'를 기록(피드백 묶기용)
    app_emb = np.array(st.session_state.app_embs)
    results = search_filtered(search_text or "", embedder, apps, app_emb, filters, k=topk)
    n_pass = sum(1 for a in apps if passes_filters(a, filters))
    show_results(query or "", results, "지원자", prefix="flow", qvec=qvec, search_id=sid,
                 total=n_pass, k=topk, pool=len(apps))
    render_detail_rerank(apps, app_emb, filters, search_text, qvec, "flow", sid)

    # 찜 목록 요약
    fav_uids = st.session_state.shortlist
    favs = [a for a in apps if a.get("uid") in fav_uids]
    if favs:
        with st.expander(f"💛 찜한 지원자 {len(favs)}명 보기", expanded=False):
            st.write(" · ".join(a["name"] for a in favs))

    # 피드백이 실제로 쌓이는지 '눈으로' 확인하는 패널
    render_feedback_panel()


def screen_upload():
    """③ 지원서 업로드 — 검색어·필터를 먼저 정해두고 → 올리면 → 그 조건으로 바로 결과."""
    render_brandbar("지원서 업로드")

    # ── 1단계: 올리기 전에 '미리' 검색어·필터를 설정 ──
    st.markdown("##### 1단계 · 어떤 배우를 찾는지 미리 설정")
    st.caption("검색어·필터를 먼저 정해두면, 아래에서 지원서를 올리는 즉시 이 조건으로 결과가 정렬돼요. "
               "(비워두면 분석만 하고, 나중에 검색해도 됩니다.)")
    query, filters, topk, expand = render_search_controls("up")

    # ── 2단계: 지원서 올리기 ──
    st.markdown("##### 2단계 · 지원서 올리기")
    st.caption("PDF · PNG · JPG · DOCX · ZIP 가능. **같은 사람은 파일 이름 앞부분으로 자동으로 묶입니다**"
               "(김민준_1~5 → 김민준 한 명, 여러 장 종합 분석). 나이·키는 지원서 글자에서 자동 추출"
               "(키는 글자에 있을 때만).")
    files = st.file_uploader("여기로 지원서 파일을 끌어다 놓으세요",
                             type=["pdf", "png", "jpg", "jpeg", "webp", "docx", "zip"],
                             accept_multiple_files=True, key="flow_up")
    if files:
        process_uploads(files)

    apps = st.session_state.applicants
    cc1, cc2 = st.columns([4, 1])
    with cc1:
        st.caption(f"분석 완료 · 지원자 {len(apps)}명")
    with cc2:
        if apps and st.button("전체 비우기"):
            st.session_state.applicants = []
            st.session_state.app_embs = []
            st.session_state.app_seen = set()
            st.rerun()

    if not apps:
        st.info("아직 분석된 지원자가 없습니다. 위에 파일을 올리면 자동으로 분석돼요.")
        return

    # ── 3단계: 미리 설정한 검색어·필터로 바로 결과 ──
    if not (query or "").strip():
        st.info("위 **1단계**에 검색어를 적어두면, 방금 올린 지원자 중에서 매칭도 순으로 바로 보여줘요. "
                "(또는 왼쪽 **🔎 배우 탐색** 화면에서 언제든 검색할 수 있어요.)")
        return

    st.divider()
    st.markdown("##### 3단계 · 검색 결과")
    search_text, shown = resolve_query(query, expand)
    sid = feedback_db.get_or_create_search(query or "", search_text)
    if shown:
        render_interp_feedback(sid, query.strip(), "up")
    qvec = embedder.encode([search_text])[0] if search_text.strip() else None
    feedback_db.set_search_embedding(sid, qvec)   # 검색의 '의미 좌표'를 기록(피드백 묶기용)
    app_emb = np.array(st.session_state.app_embs)
    results = search_filtered(search_text or "", embedder, apps, app_emb, filters, k=topk)
    n_pass = sum(1 for a in apps if passes_filters(a, filters))
    show_results(query or "", results, "지원자", prefix="up", qvec=qvec, search_id=sid,
                 total=n_pass, k=topk, pool=len(apps))
    render_detail_rerank(apps, app_emb, filters, search_text, qvec, "up", sid)


def screen_actor():
    """🎭 배우 등록 — 배우 '본인'이 직접 성별·얼굴사진·특기를 올려 프로필을 만든다.
    등록된 배우는 감독이 검색하는 같은 지원자 풀에 들어간다."""
    render_brandbar("배우 등록 · 본인 프로필")
    st.caption("배우 본인이 직접 프로필을 등록하는 화면이에요. 성별·얼굴 사진·특기를 올리면, "
               "AI가 얼굴에서 느껴지는 인상을 분석해 감독들의 검색에 노출됩니다.")

    with st.form("actor_register", clear_on_submit=False):
        st.markdown("##### 1) 기본 정보")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("이름(또는 활동명)", placeholder="예: 김도윤")
            gender = st.radio("성별", ["남", "여"], horizontal=True)
        with c2:
            age = st.number_input("나이 (선택)", min_value=0, max_value=100, value=0,
                                  help="0으로 두면 AI가 사진으로 추정한 값을 씁니다.")
            height = st.number_input("키 cm (선택)", min_value=0, max_value=230, value=0)
        specialty = st.text_input("특기 (선택)", placeholder="예: 승마, 검도, 수영")

        st.markdown("##### 2) 얼굴 사진")
        st.caption("얼굴이 잘 보이는 사진을 1장 이상 올려주세요. 여러 장 올리면 더 정확하게 분석돼요. "
                   "(JPG·PNG)")
        photos = st.file_uploader("얼굴 사진 올리기",
                                  type=["png", "jpg", "jpeg", "webp"],
                                  accept_multiple_files=True)
        submitted = st.form_submit_button("✅ 내 프로필 등록하기", type="primary")

    if submitted:
        if not (name or "").strip():
            st.warning("이름(또는 활동명)을 적어주세요.")
            return
        if not photos:
            st.warning("얼굴 사진을 한 장 이상 올려주세요.")
            return
        with st.spinner("AI가 얼굴 인상을 분석하는 중…"):
            try:
                out = process_actor_self(name, gender, age, height, specialty, photos)
            except Exception as e:
                st.error(f"등록 실패: {e}")
                return
        if out is None or "error" in out:
            st.warning(out.get("error", "분석 실패") if out else "분석 실패")
            return
        st.session_state.applicants.append(out["actor"])
        st.session_state.app_embs.append(out["emb"])
        st.success(f"🎉 등록 완료! '{out['actor']['name']}'님 프로필이 감독 검색에 추가됐어요.")
        a = out["actor"]
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.image(base64.b64decode(a["face_b64"]), use_container_width=True)
        with cc2:
            bits = [a["gender"]]
            if a["age"]:
                bits.append(f"{a['age']}세")
            if a["height_cm"]:
                bits.append(f"{a['height_cm']}cm")
            if a["specialty"]:
                bits.append(f"특기 {a['specialty']}")
            st.markdown(f"**{a['name']}** · " + " · ".join(bits))
            st.markdown(f"🪞 **AI 인상 분석**: {a['desc']}")
            if a["keywords"]:
                st.caption("키워드: " + ", ".join(a["keywords"]))

    n_self = sum(1 for a in st.session_state.applicants if a.get("self_registered"))
    st.divider()
    st.caption(f"지금까지 본인 등록 배우 {n_self}명 · 전체 지원자 {len(st.session_state.applicants)}명")


def _call_card_html(call: dict, mine: bool = False) -> str:
    """공고 한 건을 카드 HTML로 그린다(검색 결과 카드와 같은 디자인 톤).
    성별·나이 제한 없이 누구나 지원하는 형태 — 시놉시스와 원하는 배역 중심."""
    title = html.escape(call.get("title") or "")
    prod = html.escape(call.get("production") or "")
    role = html.escape(call.get("role_name") or "")
    synopsis = html.escape(call.get("synopsis") or "").replace("\n", "<br>")
    desc = html.escape(call.get("description") or "").replace("\n", "<br>")
    deadline = html.escape(call.get("deadline") or "")
    director = html.escape(call.get("director_name") or "감독")
    active = call.get("active", 1)
    status = ('<span class="tagnew">모집중 · 누구나 지원</span>' if active
              else '<span class="tagnew" style="background:var(--faint)">마감</span>')
    meta = " · ".join([b for b in [prod] if b]) or "작품 미정"

    body = ""
    if synopsis:
        body += (f'<div class="meta" style="margin-top:10px;font-weight:700;'
                 f'color:var(--navy)">시놉시스</div>'
                 f'<div class="desc">{synopsis}</div>')
    if role:
        body += (f'<div class="meta" style="margin-top:8px;font-weight:700;'
                 f'color:var(--navy)">찾는 배역</div>'
                 f'<div class="desc">{role}</div>')
    if desc:
        body += (f'<div class="meta" style="margin-top:8px;font-weight:700;'
                 f'color:var(--navy)">원하는 이미지</div>'
                 f'<div class="desc">{desc}</div>')

    chips = f'<span class="chip">마감 {deadline}</span>' if deadline else ""
    return (
        f'<div class="card">'
        f'{status}'
        f'<div class="nm">{title}</div>'
        f'<div class="meta">{meta}</div>'
        f'{body}'
        f'<div style="margin-top:8px">{chips}</div>'
        f'<div class="meta" style="margin-top:8px">올린 사람 · {director}</div>'
        f'</div>'
    )


def screen_calls_director():
    """📢 공고 올리기 — 감독이 캐스팅 공고를 올리고, 올린 공고를 관리한다."""
    render_brandbar("공고 올리기")
    st.caption("작품 **시놉시스**와 **찾는 배역**을 적어 올리면, 성별·나이 제한 없이 "
               "**누구나** 배우 화면에서 보고 지원할 수 있어요.")

    _u = current_user()
    _p = get_profile(_u["id"]) if _u else None
    director_uid = _u["id"] if _u else None
    director_name = (_p or {}).get("name") or "감독"

    with st.form("new_call", clear_on_submit=True):
        st.markdown("##### 새 공고 작성")
        title = st.text_input("공고 제목 *", placeholder="예: 청춘 멜로 영화 «여름의 끝» 출연 배우 모집")
        c1, c2 = st.columns(2)
        with c1:
            production = st.text_input("작품명 (선택)", placeholder="예: 단편영화 «여름의 끝»")
        with c2:
            deadline = st.text_input("마감일 (선택)", placeholder="예: 2026-07-15")
        synopsis = st.text_area(
            "시놉시스 · 작품 줄거리",
            placeholder="예: 바닷가 소도시에서 보낸 마지막 여름. 첫사랑과 재회한 두 사람이 "
                        "서로의 변화를 마주하며 진짜 자신을 찾아가는 청춘 멜로.",
            height=120)
        role_name = st.text_area(
            "찾는 배역",
            placeholder="예: 남자 주인공 '준호'(20대 초중반) — 서글서글하지만 속이 깊은 인물. "
                        "여자 주인공 '서연'(20대 초반) — 밝고 당찬 첫사랑.",
            height=90)
        description = st.text_area(
            "원하는 이미지 · 참고사항 (선택)",
            placeholder="예: 도시적이고 시크한 분위기보다는 풋풋하고 자연스러운 인상. "
                        "연기 경험 무관, 자유 연기 영상 환영.",
            height=80)
        submitted = st.form_submit_button("📢 이 내용으로 공고 등록", type="primary")

    if submitted:
        if not (title or "").strip():
            st.warning("공고 제목을 적어주세요.")
        elif not (synopsis or "").strip() and not (role_name or "").strip():
            st.warning("시놉시스나 찾는 배역 중 하나는 적어주세요.")
        else:
            feedback_db.create_casting_call(
                director_uid, director_name, title.strip(),
                production=production.strip(), synopsis=synopsis.strip(),
                role_name=role_name.strip(), deadline=deadline.strip(),
                description=description.strip())
            st.success("✅ 공고가 등록됐어요. 배우 화면에 바로 노출됩니다.")

    st.divider()
    st.markdown("##### 내가 올린 공고")
    my_calls = feedback_db.list_casting_calls(active_only=False, director_uid=director_uid)
    if not my_calls:
        st.info("아직 올린 공고가 없습니다. 위에서 첫 공고를 올려보세요.")
        return
    for call in my_calls:
        st.markdown(_call_card_html(call, mine=True), unsafe_allow_html=True)
        b1, b2, _sp = st.columns([1, 1, 4])
        with b1:
            if call.get("active"):
                if st.button("마감하기", key=f"close_{call['id']}"):
                    feedback_db.set_casting_call_active(call["id"], False)
                    st.rerun()
            else:
                if st.button("다시 모집", key=f"open_{call['id']}"):
                    feedback_db.set_casting_call_active(call["id"], True)
                    st.rerun()
        with b2:
            if st.button("삭제", key=f"del_{call['id']}"):
                feedback_db.delete_casting_call(call["id"], director_uid)
                st.rerun()


def screen_calls_actor():
    """📋 공고 보기 — 배우가 감독들이 올린 캐스팅 공고를 본다."""
    render_brandbar("공고 보기")
    st.caption("감독들이 올린 캐스팅 공고예요. 관심 있는 공고를 확인하고 프로필을 준비해두세요.")
    calls = feedback_db.list_casting_calls(active_only=True)
    if not calls:
        st.info("아직 올라온 공고가 없습니다. 공고가 올라오면 여기에 표시돼요.")
        return
    st.markdown(f"###### 현재 모집중인 공고 {len(calls)}건")
    for call in calls:
        st.markdown(_call_card_html(call), unsafe_allow_html=True)


# ============ 로그인 + 역할별 프로필 (게이트) ============
# 카카오 로그인(설정되면) 또는 데모 로그인(설정 전 테스트용) → 역할 선택 → 프로필 작성해야 사용 가능.

def _auth_configured() -> bool:
    """카카오(OIDC) 로그인이 secrets에 설정돼 있으면 True. 없으면 데모 로그인으로 대체."""
    try:
        return "auth" in st.secrets
    except Exception:
        return False


def current_user() -> dict | None:
    """지금 로그인한 사용자를 {id, email, name}으로 돌려준다(없으면 None).
    - 카카오 설정됨: 진짜 로그인(st.user) 사용
    - 미설정: 데모 로그인(session_state.demo_user) 사용"""
    if _auth_configured():
        try:
            if getattr(st.user, "is_logged_in", False):
                email = st.user.get("email") or st.user.get("sub") or "kakao_user"
                return {"id": email, "email": st.user.get("email"),
                        "name": st.user.get("name") or st.user.get("nickname") or "사용자"}
        except Exception:
            return None
        return None
    return st.session_state.get("demo_user")


def get_profile(uid: str) -> dict | None:
    return st.session_state.profiles.get(uid)


def save_profile(uid: str, prof: dict) -> None:
    st.session_state.profiles[uid] = prof


def _logo_or_text():
    lb64 = logo_b64()
    if lb64:
        st.markdown(f'<img src="data:image/png;base64,{lb64}" style="width:100%;'
                    f'max-width:160px;margin:4px auto 10px;display:block;">',
                    unsafe_allow_html=True)
    else:
        st.markdown("<div style='font-size:24px;font-weight:900;color:#1F3A5F;"
                    "margin:6px 0 10px'>CATING</div>", unsafe_allow_html=True)


def _go_login():
    st.session_state.show_login = True


def render_landing():
    """첫 화면 — 제품을 소개하는 랜딩페이지. '시작하기'를 누르면 로그인으로."""
    st.markdown("""
    <style>
    .hero { text-align:center; padding:54px 16px 30px; }
    .hero .brand { font-size:20px; font-weight:900; letter-spacing:.22em;
                   color:#4AA9A0; margin-bottom:18px; }
    .hero h1 { font-size:46px; line-height:1.18; font-weight:900; color:#1F3A5F;
               margin:0 0 18px; letter-spacing:-.01em; }
    .hero h1 .accent { color:#4AA9A0; }
    .hero p.sub { font-size:17px; color:#5C6677; max-width:620px; margin:0 auto;
                  line-height:1.7; }
    .feat { background:#FFFFFF; border:1px solid #E6DFD2; border-radius:18px;
            padding:24px 20px; height:100%; box-shadow:0 2px 12px rgba(31,58,95,.05); }
    .feat .ic { font-size:30px; }
    .feat h4 { font-size:17px; font-weight:800; color:#1F3A5F; margin:10px 0 6px; }
    .feat p { font-size:13.5px; color:#5C6677; line-height:1.65; margin:0; }
    .howto { background:#1F3A5F; border-radius:20px; padding:30px 26px; color:#EFE9DD;
             margin-top:8px; }
    .howto h3 { color:#FFFFFF; font-size:20px; font-weight:900; margin:0 0 16px; }
    .howto .step { font-size:14px; line-height:1.7; margin:6px 0; }
    .howto .step b { color:#7FD3C9; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div class="brand">CATING · CAST CATCHING</div>
      <h1>원하는 배우의 <span class="accent">이미지</span>를<br>문장으로 검색하세요</h1>
      <p class="sub">“청량하고 청순한 첫사랑 느낌”, “도시적인 시크함”—
      조건 필터로는 못 찾던 <b>분위기·인상</b>을 자연어로 검색해
      가장 잘 맞는 배우를 매칭도 순으로 찾아줍니다.</p>
    </div>
    """, unsafe_allow_html=True)

    _, cmid, _ = st.columns([1, 1.1, 1])
    with cmid:
        st.button("시작하기  →", type="primary", use_container_width=True,
                  on_click=_go_login)
    st.write("")

    f1, f2, f3 = st.columns(3)
    with f1:
        st.markdown("""<div class="feat"><div class="ic">🎬</div>
        <h4>감독 — 감각으로 캐스팅</h4>
        <p>수백 장 지원서를 일일이 넘기지 않아도 됩니다. 원하는 인상을 문장으로
        적으면 적합한 지원자가 바로 위로 올라옵니다.</p></div>""",
                    unsafe_allow_html=True)
    with f2:
        st.markdown("""<div class="feat"><div class="ic">🎭</div>
        <h4>배우 — 이미지로 발견되다</h4>
        <p>본인 사진과 프로필을 올리면 AI가 인상을 분석합니다. 노출 순서가 아니라
        ‘이미지 적합도’로 검토되어 신인도 공정하게 기회를 얻습니다.</p></div>""",
                    unsafe_allow_html=True)
    with f3:
        st.markdown("""<div class="feat"><div class="ic">✨</div>
        <h4>사전에 없는 표현도 OK</h4>
        <p>미리 정해둔 항목이 아니라 ‘의미’로 검색합니다. 어떤 표현을 적어도
        뜻이 가까운 배우를 찾아냅니다.</p></div>""",
                    unsafe_allow_html=True)

    st.write("")
    st.markdown("""
    <div class="howto">
      <h3>어떻게 작동하나요</h3>
      <div class="step"><b>1.</b> 감독이 지원서(PDF·이미지 등)를 올리면 얼굴을 자동 추출합니다.</div>
      <div class="step"><b>2.</b> AI가 얼굴 인상을 설명 텍스트로 만들어 ‘의미 좌표’로 저장합니다.</div>
      <div class="step"><b>3.</b> 검색 문장도 같은 좌표로 바꿔, 가장 가까운 배우를 매칭도 순으로 보여줍니다.</div>
    </div>
    """, unsafe_allow_html=True)
    st.write("")
    _, cmid2, _ = st.columns([1, 1.1, 1])
    with cmid2:
        st.button("지금 시작하기  →", type="primary", use_container_width=True,
                  on_click=_go_login, key="cta_bottom")


def render_login():
    """로그인 화면 — 카카오 버튼(설정 시) 또는 데모 로그인(설정 전)."""
    _, mid, _ = st.columns([1, 2, 1])
    with mid:
        if st.button("← 처음으로"):
            st.session_state.show_login = False
            st.rerun()
        _logo_or_text()
        st.markdown("<h3 style='text-align:center;color:#1F3A5F'>CATING 로그인</h3>",
                    unsafe_allow_html=True)
        st.caption("감독·배우 모두 로그인 후 프로필을 작성하면 이용할 수 있어요.")
        if _auth_configured():
            st.button("💬  카카오로 로그인", type="primary", use_container_width=True,
                      on_click=st.login, args=("kakao",))
        else:
            st.info("카카오 로그인은 설정이 끝나면 여기에 버튼으로 나타나요. "
                    "지금은 아래 **데모 로그인**으로 먼저 사용해 볼 수 있어요.")
            with st.form("demo_login"):
                em = st.text_input("이메일", placeholder="you@example.com")
                nm = st.text_input("이름", placeholder="홍길동")
                if st.form_submit_button("데모 로그인", type="primary",
                                         use_container_width=True):
                    if (em or "").strip():
                        st.session_state.demo_user = {
                            "id": em.strip(), "email": em.strip(),
                            "name": (nm or "사용자").strip()}
                        st.rerun()
                    else:
                        st.warning("이메일을 입력해 주세요.")


def _do_logout():
    if _auth_configured():
        try:
            st.logout()
            return
        except Exception:
            pass
    st.session_state.demo_user = None
    st.session_state.pending_role = None
    st.rerun()


GENRES = ["드라마", "멜로", "스릴러", "코미디", "액션", "공포", "다큐", "판타지", "사극"]
FIELDS = ["단편", "독립", "상업", "웹드라마", "숏폼", "광고", "뮤직비디오"]


def render_profile_setup(user: dict):
    """프로필 미작성 사용자 → 역할 선택 후 역할별 프로필 폼."""
    _, mid, _ = st.columns([1, 3, 1])
    with mid:
        _logo_or_text()
        st.markdown(f"##### 환영해요, {html.escape(user.get('name') or '사용자')}님 👋")
        role = st.session_state.pending_role
        if role is None:
            st.markdown("###### 어떤 분이신가요? 역할을 골라주세요.")
            c1, c2 = st.columns(2)
            if c1.button("🎬  감독 / 캐스팅", use_container_width=True):
                st.session_state.pending_role = "director"; st.rerun()
            if c2.button("🎭  배우 / 지원자", use_container_width=True):
                st.session_state.pending_role = "actor"; st.rerun()
            st.divider()
            st.button("로그아웃", on_click=_do_logout)
            return
        if role == "director":
            _director_profile_form(user)
        else:
            _actor_profile_form(user)
        st.divider()
        if st.button("← 역할 다시 선택"):
            st.session_state.pending_role = None; st.rerun()


def _director_profile_form(user: dict):
    st.markdown("###### 감독 프로필 작성")
    with st.form("director_profile"):
        st.markdown("**기본 정보**")
        name = st.text_input("이름 *", value=user.get("name") or "")
        c1, c2 = st.columns(2)
        with c1:
            org = st.text_input("소속 (선택)", placeholder="제작사·스튜디오·학교 / 프리랜서")
        with c2:
            title = st.text_input("직함 (선택)", placeholder="감독 / 조감독 / PD / 캐스팅 디렉터")
        email = st.text_input("이메일 *", value=user.get("email") or "")
        verified = st.checkbox("실명·신원 인증에 동의합니다 (가짜 공고 방지 · 배우 보호) *")
        st.markdown("**연출 정보 (선택)**")
        genres = st.multiselect("주 연출 장르", GENRES)
        fields = st.multiselect("활동 영역", FIELDS)
        career = st.text_area("작품 경력 / 연출작", placeholder="예) 2024 단편 '○○○' 연출")
        ok = st.form_submit_button("✅ 감독으로 시작하기", type="primary")
    if ok:
        if not name.strip() or not email.strip():
            st.warning("이름과 이메일은 필수예요."); return
        if not verified:
            st.warning("배우 보호를 위해 실명·신원 인증 동의가 필요해요."); return
        save_profile(user["id"], {
            "role": "director", "name": name.strip(), "org": org.strip(),
            "title": title.strip(), "email": email.strip(), "verified": verified,
            "genres": genres, "fields": fields, "career": career.strip(),
        })
        st.session_state.pending_role = None
        st.success("감독 프로필이 등록됐어요!"); st.rerun()


def _actor_profile_form(user: dict):
    st.markdown("###### 배우 프로필 작성  ·  정면 증명사진은 필수예요")
    with st.form("actor_profile"):
        st.markdown("**A. 기본 정보**")
        c1, c2 = st.columns(2)
        with c1:
            name = st.text_input("이름(또는 활동명) *", value=user.get("name") or "")
            birth_year = st.number_input("출생연도 * (예: 2000)", 1940, CURRENT_YEAR, 2000,
                                         help="나이는 출생연도로 자동 계산돼요.")
            gender = st.radio("성별 *", ["남", "여"], horizontal=True)
            phone = st.text_input("전화번호 *", placeholder="010-0000-0000")
        with c2:
            email = st.text_input("이메일 *", value=user.get("email") or "")
            region = st.text_input("거주지 (선택)", placeholder="예: 서울 / 부산 (촬영지 매칭용)")
            height = st.number_input("키 cm * (사진으로 분석 불가, 직접 입력)", 0, 230, 0)
            weight = st.number_input("몸무게 kg * ", 0, 200, 0)
        specialty = st.text_input("특기 (선택)", placeholder="예: 수영, 유도, 승마, 피아노")
        c3, c4 = st.columns(2)
        with c3:
            languages = st.text_input("가능 언어 (선택)", placeholder="예: 한국어, 영어, 경상도 사투리")
        with c4:
            education = st.text_input("학력 (선택)", placeholder="예: ○○대 연극영화과")

        st.markdown("**B. 사진** — 정면 증명사진은 필수, 추가 사진이 많을수록 인상 분석이 정확해져요")
        st.caption("여러 장이면 그날 표정에 휘둘리지 않고, 전신 사진이 있으면 체형·실루엣도 참고돼요. "
                   "본인이 올린 사진이라 초상권 안전하며, 원본은 저장하지 않고 분석 결과만 보관해요.")
        front = st.file_uploader("정면 증명사진 * (1장)", type=["png", "jpg", "jpeg", "webp"],
                                 accept_multiple_files=False)
        extras = st.file_uploader("추가 사진 (선택, 측면·전신·매력샷·출연 스틸 등 여러 장)",
                                  type=["png", "jpg", "jpeg", "webp"], accept_multiple_files=True)

        st.markdown("**C. 경력 / 필모그래피 (선택)**")
        st.caption("형식: 연도 / 방송사·제작 / 작품명 / 배역 / 비중.  예) 2026 tvN <○○○> ○○역 - 주연. "
                   "※ 경력은 표시·약한 참고용일 뿐, 검색 순위를 크게 올리지 않아요(신인 공정성).")
        career = st.text_area("경력", placeholder="2026 tvN <○○○> ○○역 - 주연\n2025 독립영화 <△△> □□역 - 조연")

        st.markdown("**D. 연기 영상 (선택)**")
        video_link = st.text_input("영상 링크", placeholder="유튜브/구글드라이브 등 (지금은 링크만 저장)")

        ok = st.form_submit_button("✅ 배우로 시작하기", type="primary")
    if ok:
        if not (name or "").strip():
            st.warning("이름(또는 활동명)을 적어주세요."); return
        if not (phone or "").strip() or not (email or "").strip():
            st.warning("연락처(전화번호·이메일)는 필수예요."); return
        if not height or not weight:
            st.warning("키와 몸무게는 직접 입력해 주세요(사진으로 알 수 없어요)."); return
        if not front:
            st.warning("정면 증명사진은 필수예요."); return
        photos = [front] + list(extras or [])
        extra = {"birth_year": int(birth_year), "region": (region or "").strip() or None,
                 "weight_kg": int(weight), "phone": phone.strip(), "email": email.strip(),
                 "languages": (languages or "").strip() or None,
                 "education": (education or "").strip() or None,
                 "career": (career or "").strip() or None,
                 "video_link": (video_link or "").strip() or None}
        with st.spinner("AI가 얼굴 인상을 분석하는 중…"):
            try:
                out = process_actor_self(name, gender, 0, height, specialty, photos, extra)
            except Exception as e:
                st.error(f"등록 실패: {e}"); return
        if out is None or "error" in out:
            st.warning(out.get("error", "분석 실패") if out else "분석 실패"); return
        st.session_state.applicants.append(out["actor"])
        st.session_state.app_embs.append(out["emb"])
        save_profile(user["id"], {"role": "actor", "name": out["actor"]["name"],
                                  "email": email.strip(), "actor_uid": out["actor"]["uid"]})
        st.session_state.pending_role = None
        st.success("배우 프로필이 등록됐어요! 감독 검색에 노출됩니다."); st.rerun()


def screen_profile():
    """내 프로필 보기(로그인 사용자의 역할별 프로필)."""
    user = current_user()
    prof = get_profile(user["id"]) if user else None
    if not prof:
        st.info("프로필이 없습니다."); return
    render_brandbar("내 프로필")
    if prof["role"] == "director":
        st.markdown(f"### 🎬 {prof['name']}")
        bits = []
        if prof.get("title"): bits.append(prof["title"])
        if prof.get("org"): bits.append(prof["org"])
        st.caption(" · ".join(bits) if bits else "감독")
        st.markdown(f"- 이메일: {prof.get('email','-')}")
        st.markdown(f"- 신원 인증 동의: {'✅' if prof.get('verified') else '❌'}")
        if prof.get("genres"): st.markdown("- 주 연출 장르: " + ", ".join(prof["genres"]))
        if prof.get("fields"): st.markdown("- 활동 영역: " + ", ".join(prof["fields"]))
        if prof.get("career"): st.markdown(f"- 경력:\n\n{prof['career']}")
    else:
        a = next((x for x in st.session_state.applicants
                  if x.get("uid") == prof.get("actor_uid")), None)
        st.markdown(f"### 🎭 {prof['name']}")
        if not a:
            st.info("등록된 배우 프로필을 찾을 수 없습니다. 다시 등록해 주세요.")
            return

        # ---- A 기본정보 + B 대표사진 ----
        cc1, cc2 = st.columns([1, 2])
        with cc1:
            st.image(base64.b64decode(a["face_b64"]), use_container_width=True,
                     caption="정면 증명사진")
        with cc2:
            top = [a["gender"]]
            if a.get("age"): top.append(f"{a['age']}세")
            if a.get("birth_year"): top.append(f"{a['birth_year']}년생")
            st.markdown("**" + " · ".join(top) + "**")
            info = []
            if a.get("height_cm"): info.append(f"키 {a['height_cm']}cm")
            if a.get("weight_kg"): info.append(f"몸무게 {a['weight_kg']}kg")
            if a.get("region"): info.append(f"거주지 {a['region']}")
            if a.get("specialty"): info.append(f"특기 {a['specialty']}")
            if a.get("languages"): info.append(f"언어 {a['languages']}")
            if a.get("education"): info.append(f"학력 {a['education']}")
            for line in info:
                st.markdown(f"- {line}")
            contact = []
            if a.get("phone"): contact.append(a["phone"])
            if a.get("email"): contact.append(a["email"])
            if contact:
                st.caption("연락처: " + " · ".join(contact))

        # ---- B 추가 사진(있으면) ----
        extras = a.get("all_faces_b64") or []
        if len(extras) > 1:
            st.markdown("**📸 추가 사진**")
            cols = st.columns(min(4, len(extras) - 1))
            for i, b64 in enumerate(extras[1:]):
                with cols[i % len(cols)]:
                    st.image(base64.b64decode(b64), use_container_width=True)

        # ---- E AI 인상 분석 (강점 중심·긍정적 표현) ----
        st.markdown("#### 🪞 AI 인상 분석")
        st.markdown(a["desc"])
        if a.get("keywords"):
            st.caption("키워드: " + ", ".join(a["keywords"]))
        traits = a.get("traits") or {}
        strong = [(ax, traits[ax]) for ax in TRAIT_AXES
                  if isinstance(traits.get(ax), (int, float)) and traits[ax] >= 0.55]
        strong.sort(key=lambda kv: kv[1], reverse=True)
        if strong:
            st.markdown("**돋보이는 매력 포인트**")
            for ax, v in strong[:5]:
                st.markdown(f"- {ax}  ·  {int(round(v * 100))}%")
            st.caption("※ 약점이 아닌 ‘강점’만 보여줍니다. 이 분석은 검색 매칭에 쓰이는 고정 기준입니다.")

        # ---- C 경력 / 필모그래피 (표시용) ----
        if a.get("career"):
            st.markdown("#### 🎬 경력 / 필모그래피")
            st.markdown(a["career"])
            st.caption("※ 경력은 참고 표시용입니다. 검색 순위에는 거의 영향을 주지 않아요(신인도 공정하게).")

        # ---- D 영상 (링크) ----
        if a.get("video_link"):
            st.markdown("#### 🎞 영상")
            st.markdown(f"[{a['video_link']}]({a['video_link']})")

        # ---- G 활동정보 (찜/조회수 등) ----
        st.markdown("#### 📊 활동 정보")
        try:
            act = feedback_db.actor_activity(a.get("uid"))
        except Exception:
            act = {}
        g1, g2, g3 = st.columns(3)
        g1.metric("찜(관심)", act.get("shortlist", 0))
        g2.metric("조회(클릭)", act.get("click", 0))
        g3.metric("연락 받음", act.get("contact", 0))
        st.caption("지원 현황(공고 지원 내역)은 공고 시스템이 열리면 함께 표시됩니다.")

        st.divider()
        st.caption("프로필을 다시 만들려면 로그아웃 후 재등록하거나, 왼쪽 메뉴를 사용하세요.")


# ============ 게이트: 랜딩 → 로그인 → 프로필 → 앱 ============
_user = current_user()
if _user is None:
    if not st.session_state.get("show_login"):
        render_landing()
        st.stop()
    render_login()
    st.stop()
_prof = get_profile(_user["id"])
if _prof is None:
    render_profile_setup(_user)
    st.stop()

# --- 로그인 + 프로필 완료: 역할별 사이드바 + 화면 ---
with st.sidebar:
    _logo_or_text()
    _role_kr = "감독" if _prof["role"] == "director" else "배우"
    st.markdown(f"**{html.escape(_prof['name'])}** · {_role_kr}")
    st.button("로그아웃", on_click=_do_logout, use_container_width=True)
    st.divider()
    if _prof["role"] == "director":
        NAV = {
            "🔎 배우 탐색": screen_search,
            "📤 지원서 업로드": screen_upload,
            "📢 공고 올리기": screen_calls_director,
            "🙍 내 프로필": screen_profile,
        }
        st.caption(f"분석된 지원자 {len(st.session_state.applicants)}명 · "
                   f"찜 {len(st.session_state.shortlist)}명")
    else:  # 배우
        NAV = {
            "📋 공고 보기": screen_calls_actor,
            "🙍 내 프로필": screen_profile,
        }

    # 메뉴 — 동그라미 라디오 대신 '항목 전체가 눌리는' 버튼 방식
    nav_keys = list(NAV.keys())
    if st.session_state.get("nav_choice") not in nav_keys:
        st.session_state.nav_choice = nav_keys[0]
    for _label in nav_keys:
        _active = (st.session_state.nav_choice == _label)
        if st.button(_label, key=f"nav_{_label}", use_container_width=True,
                     type="primary" if _active else "secondary"):
            st.session_state.nav_choice = _label
            st.rerun()
    choice = st.session_state.nav_choice

NAV[choice]()
