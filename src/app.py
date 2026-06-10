import streamlit as st
from src.graph import prepare
from src.generate import AnswerGenerator, REFUSAL_RESPONSE

st.set_page_config(page_title="FIFA World Cup Expert", page_icon="⚽", layout="centered")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  * { font-family: 'Inter', sans-serif; }

  /* ── Background ── */
  html, body {
      background-color: #060e1c !important;
  }
  .stApp {
      background:
          radial-gradient(ellipse at 20% 20%, rgba(255,215,0,0.06) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(74,158,218,0.06) 0%, transparent 50%),
          linear-gradient(160deg, #060e1c 0%, #0f1f38 40%, #071526 100%);
  }
  [data-testid="stAppViewContainer"],
  [data-testid="stAppViewBlockContainer"],
  [data-testid="stMain"],
  [data-testid="stMainBlockContainer"] {
      background: transparent !important;
  }
  header[data-testid="stHeader"] { background: transparent; }

  /* ── Hero ── */
  .hero {
      text-align: center;
      padding: 2.5rem 1rem 0.5rem;
  }
  .hero .ball {
      font-size: 4.5rem;
      display: inline-block;
      animation: spin 8s linear infinite;
      filter: drop-shadow(0 0 20px rgba(255,215,0,0.5));
  }
  @keyframes spin { to { transform: rotate(360deg); } }

  .hero h1 {
      font-size: 2.6rem;
      font-weight: 800;
      background: linear-gradient(135deg, #FFD700 0%, #FFA500 50%, #FFD700 100%);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      margin: 0.6rem 0 0.3rem;
      letter-spacing: -0.5px;
  }
  .hero .subtitle {
      color: #5a7a9a;
      font-size: 0.95rem;
      letter-spacing: 0.3px;
  }

  /* ── Divider ── */
  .gold-divider {
      height: 1px;
      background: linear-gradient(90deg, transparent, rgba(255,215,0,0.4), transparent);
      margin: 1.4rem 0;
      border: none;
  }

  /* ── Chat messages ── */
  [data-testid="stChatMessage"] {
      border-radius: 14px;
      margin-bottom: 0.8rem;
      backdrop-filter: blur(8px);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarUser"]) {
      background: rgba(255,215,0,0.06);
      border: 1px solid rgba(255,215,0,0.15);
  }
  [data-testid="stChatMessage"]:has([data-testid="stChatMessageAvatarAssistant"]) {
      background: rgba(255,255,255,0.03);
      border: 1px solid rgba(74,158,218,0.15);
  }

  /* ── Chat input ── */
  [data-testid="stChatInput"],
  [data-testid="stChatInput"] > div,
  [data-testid="stChatInput"] > div > div,
  [data-testid="stChatInput"] > div > div > div {
      background: #0a1628 !important;
      border-radius: 14px !important;
  }
  [data-testid="stChatInput"] {
      border: 1px solid rgba(255,215,0,0.25) !important;
  }
  [data-testid="stChatInput"] textarea {
      color: #e8f0f8 !important;
      background: #0a1628 !important;
      caret-color: #FFD700 !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #4a6a8a !important; }

  /* ── Bottom bar — full coverage ── */
  section[data-testid="stBottom"],
  section[data-testid="stBottom"] *,
  [data-testid="stBottomBlockContainer"],
  [data-testid="stBottomBlockContainer"] * {
      background-color: #060e1c !important;
  }

  /* Keep input + button visually intact — higher specificity than stBottom * */
  section[data-testid="stBottom"] [data-testid="stChatInput"],
  section[data-testid="stBottom"] [data-testid="stChatInput"] textarea,
  [data-testid="stChatInput"],
  [data-testid="stChatInput"] textarea {
      background-color: #0a1628 !important;
  }

  /* Pseudo-element backdrop to kill any leftover white strips */
  section[data-testid="stBottom"]::before {
      content: '';
      position: absolute;
      inset: 0;
      background: #060e1c;
      z-index: -1;
  }

  /* ── Expander (sources) ── */
  [data-testid="stExpander"] {
      background: rgba(255,255,255,0.02) !important;
      border: 1px solid rgba(255,215,0,0.12) !important;
      border-radius: 10px !important;
  }
  [data-testid="stExpander"] summary { color: #a0b8d0 !important; font-size: 0.85rem; }

  /* ── Sidebar ── */
  [data-testid="stSidebar"] {
      background: rgba(6, 14, 28, 0.97) !important;
      border-right: 1px solid rgba(255,215,0,0.08) !important;
  }
  .sidebar-title {
      font-size: 0.7rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #3a5a7a;
      padding: 0.2rem 0 0.6rem;
  }

  /* ── Suggestion buttons ── */
  .stButton > button {
      width: 100%;
      background: rgba(255,255,255,0.03) !important;
      border: 1px solid rgba(255,215,0,0.12) !important;
      border-radius: 10px !important;
      color: #8aa8c4 !important;
      text-align: left !important;
      padding: 0.55rem 0.9rem !important;
      font-size: 0.83rem !important;
      margin-bottom: 0.3rem;
      transition: all 0.18s ease;
      line-height: 1.4;
  }
  .stButton > button:hover {
      background: rgba(255,215,0,0.08) !important;
      border-color: rgba(255,215,0,0.35) !important;
      color: #FFD700 !important;
      transform: translateX(3px);
  }

  /* ── Clear button override ── */
  .clear-btn > button {
      background: rgba(220,50,50,0.07) !important;
      border-color: rgba(220,50,50,0.2) !important;
      color: #c07070 !important;
      text-align: center !important;
  }
  .clear-btn > button:hover {
      background: rgba(220,50,50,0.15) !important;
      border-color: rgba(220,50,50,0.4) !important;
      color: #ff9090 !important;
      transform: none !important;
  }

  /* ── Centered search (empty state) ── */
  .search-wrap {
      max-width: 520px;
      margin: 2.5rem auto 0;
  }
  .search-wrap [data-testid="stTextInput"] > div {
      background: rgba(255,255,255,0.05) !important;
      border: 1.5px solid rgba(255,215,0,0.3) !important;
      border-radius: 50px !important;
      padding: 0.1rem 0.5rem;
      transition: border-color 0.2s;
  }
  .search-wrap [data-testid="stTextInput"] > div:focus-within {
      border-color: #FFD700 !important;
      box-shadow: 0 0 0 3px rgba(255,215,0,0.1) !important;
  }
  .search-wrap [data-testid="stTextInput"] input {
      background: transparent !important;
      border: none !important;
      color: #e8f0f8 !important;
      font-size: 1rem !important;
      text-align: center;
  }
  .search-wrap [data-testid="stTextInput"] input::placeholder { color: #3a5a7a !important; }
  .search-wrap .stFormSubmitButton > button {
      background: linear-gradient(135deg, #FFD700, #e6960c) !important;
      color: #060e1c !important;
      font-weight: 700 !important;
      border: none !important;
      border-radius: 50px !important;
      padding: 0.55rem 2.8rem !important;
      font-size: 0.95rem !important;
      width: auto !important;
      display: block;
      margin: 0.8rem auto 0;
      letter-spacing: 0.3px;
      transition: opacity 0.2s, transform 0.2s;
  }
  .search-wrap .stFormSubmitButton > button:hover {
      opacity: 0.9;
      transform: scale(1.02);
  }

  /* ── Glowing orbs ── */
  .orb {
      position: fixed;
      border-radius: 50%;
      pointer-events: none;
      z-index: 0;
      filter: blur(80px);
  }
  .orb-gold { width: 320px; height: 320px; background: rgba(255,215,0,0.07); top: -80px; right: -80px; }
  .orb-blue { width: 280px; height: 280px; background: rgba(74,158,218,0.07); bottom: 60px; left: -60px; }

  /* ── Refusal box ── */
  .refusal-box {
      background: rgba(234,179,8,0.07);
      border: 1px solid rgba(234,179,8,0.25);
      border-left: 3px solid #facc15;
      border-radius: 10px;
      padding: 0.9rem 1.1rem;
      color: #c8b560;
      font-size: 0.9rem;
      line-height: 1.6;
      margin: 0.3rem 0;
  }

  /* ── Right info panel (Built with only) ── */
  .right-info-panel {
      position: fixed;
      top: 80px;
      right: 12px;
      z-index: 999;
      width: 170px;
      background: rgba(6,14,28,0.90);
      border: 1px solid rgba(255,215,0,0.12);
      border-radius: 12px;
      padding: 14px 16px;
      backdrop-filter: blur(10px);
  }
  .rp-section-title {
      font-size: 0.6rem;
      font-weight: 700;
      letter-spacing: 2px;
      text-transform: uppercase;
      color: #3a5a7a;
      margin-bottom: 8px;
  }
  .tech-row { display: flex; align-items: center; gap: 5px; margin-bottom: 4px; }
  .tech-badge {
      font-size: 0.68rem; font-weight: 600; padding: 2px 8px;
      border-radius: 20px; white-space: nowrap;
  }
  .badge-purple { background: rgba(168,85,247,0.15); color: #c084fc; border: 1px solid rgba(168,85,247,0.25); }
  .badge-green  { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.22); }
  .badge-blue   { background: rgba(74,158,218,0.13); color: #60a5fa; border: 1px solid rgba(74,158,218,0.25); }
  .badge-orange { background: rgba(251,146,60,0.13); color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
  .badge-yellow { background: rgba(250,204,21,0.12); color: #facc15; border: 1px solid rgba(250,204,21,0.22); }
  .badge-red    { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.22); }

  /* ── Metrics strip (Evaluation + Corpus) ── */
  /* ── Metrics caption (subtle, secondary) ── */
  .metrics-caption {
      text-align: center;
      margin: 0.2rem 0 1.2rem;
      display: flex;
      flex-direction: column;
      gap: 4px;
  }
  .metrics-caption .mc-line {
      font-size: 0.72rem;
      color: rgba(74,222,128,0.45);
      letter-spacing: 0.2px;
  }
  .metrics-caption .mc-line span {
      color: rgba(74,222,128,0.75);
      font-weight: 600;
  }
  .metrics-caption .mc-dot {
      color: rgba(74,222,128,0.2);
      margin: 0 5px;
  }


  /* ── Feedback buttons ── */
  .fb-msg {
      font-size: 0.72rem;
      margin-left: 4px;
  }
  .fb-msg.like    { color: #4ade80; }
  .fb-msg.dislike { color: #f87171; }

  /* Feedback buttons side by side */
  [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
      gap: 0 !important;
  }
  [data-testid="stChatMessage"] [data-testid="column"] {
      padding-left: 0 !important;
      padding-right: 0 !important;
      min-width: 0 !important;
  }

  /* Selected state — primary buttons in feedback row */
  button[data-testid="baseButton-primary"] {
      background: rgba(255, 215, 0, 0.18) !important;
      border: 1.5px solid rgba(255, 215, 0, 0.7) !important;
      color: #FFD700 !important;
      box-shadow: 0 0 8px rgba(255, 215, 0, 0.25) !important;
  }

  p, li { color: #c8dcef; }
  a { color: #4a9eda !important; }
  a:hover { color: #FFD700 !important; }
</style>

<div class="orb orb-gold"></div>
<div class="orb orb-blue"></div>

<div class="right-info-panel">
  <div class="rp-section-title">Built with</div>
  <div class="tech-row"><span class="tech-badge badge-purple">🤖 Claude Haiku 4.5</span></div>
  <div class="tech-row"><span class="tech-badge badge-green">🌲 Pinecone</span></div>
  <div class="tech-row"><span class="tech-badge badge-blue">🔗 LangGraph</span></div>
  <div class="tech-row"><span class="tech-badge badge-blue">⛓️ LangChain</span></div>
  <div class="tech-row"><span class="tech-badge badge-orange">🤗 HuggingFace</span></div>
  <div class="tech-row"><span class="tech-badge badge-yellow">📊 BM25</span></div>
  <div class="tech-row"><span class="tech-badge badge-red">🎈 Streamlit</span></div>
</div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="ball">⚽</div>
  <h1>FIFA World Cup Expert</h1>
  <p class="subtitle">Tournament history · Iconic players · Legendary moments &nbsp;·&nbsp; 1930 – 2022</p>
</div>
<hr class="gold-divider">

<div class="metrics-caption">
  <div class="mc-line">
    Faithfulness <span>0.92</span><span class="mc-dot">·</span>
    Relevancy <span>0.87</span><span class="mc-dot">·</span>
    Recall <span>0.86</span><span class="mc-dot">·</span>
    Golden Dataset <span>17/17</span>
  </div>
  <div class="mc-line">
    <span>1,466</span> docs<span class="mc-dot">·</span>
    <span>22</span> editions<span class="mc-dot">·</span>
    Hybrid retrieval<span class="mc-dot">·</span>
    <span>384‑dim</span> embeddings
  </div>
</div>
""", unsafe_allow_html=True)

SUGGESTIONS = [
    ("🏆", "Who won the 1970 World Cup?"),
    ("🤚", "What was the Hand of God?"),
    ("🐐", "How many goals did Messi score in 2022?"),
    ("🛡️", "Which team had the best defense in 2010?"),
    ("😱", "Tell me about the Maracanazo"),
    ("⚡", "Who is Kylian Mbappé?"),
]

# ── Sidebar ──
with st.sidebar:
    st.markdown('<p class="sidebar-title">Try asking</p>', unsafe_allow_html=True)
    for icon, text in SUGGESTIONS:
        if st.button(f"{icon}  {text}", key=f"btn_{text}"):
            st.session_state.pending_query = text

    st.markdown("<hr class='gold-divider'>", unsafe_allow_html=True)
    st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
    if st.button("Clear conversation"):
        st.session_state.history = []
        st.session_state.pop("pending_query", None)
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)



@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_generator():
    return AnswerGenerator()


def render_citations(citations):
    with st.expander("📚 Sources"):
        for c in citations:
            if c["source"] == "wikipedia":
                st.markdown(f"**[{c['index']}]** [{c.get('title')}]({c.get('url')})")
            elif c["source"] == "tournament_standings":
                st.markdown(f"**[{c['index']}]** Standings — {c.get('team')} ({c.get('year')})")
            elif c["source"] == "tournament_summary":
                st.markdown(f"**[{c['index']}]** Tournament summary — {c.get('year')}")
            else:
                st.markdown(f"**[{c['index']}]** {c.get('source', 'source')}")


if "history" not in st.session_state:
    st.session_state.history = []
if "feedback" not in st.session_state:
    st.session_state.feedback = {}


def render_feedback(idx: int):
    fb = st.session_state.feedback.get(idx)
    c1, c2, c3 = st.columns([1, 1, 8], gap="small")
    with c1:
        if st.button("👍", key=f"like_{idx}", type="primary" if fb == "like" else "secondary", use_container_width=True):
            st.session_state.feedback[idx] = None if fb == "like" else "like"
            st.rerun()
    with c2:
        if st.button("👎", key=f"dislike_{idx}", type="primary" if fb == "dislike" else "secondary", use_container_width=True):
            st.session_state.feedback[idx] = None if fb == "dislike" else "dislike"
            st.rerun()
    with c3:
        if fb == "like":
            st.markdown('<span class="fb-msg like">Thanks for the feedback!</span>', unsafe_allow_html=True)
        elif fb == "dislike":
            st.markdown('<span class="fb-msg dislike">Thanks — noted for improvement.</span>', unsafe_allow_html=True)


# ── Render history ──
for i, turn in enumerate(st.session_state.history):
    with st.chat_message("user"):
        st.write(turn["query"])
    with st.chat_message("assistant", avatar="⚽"):
        if turn.get("refused"):
            st.markdown(f'<div class="refusal-box">⚠️&nbsp; {turn["answer"]}</div>',
                        unsafe_allow_html=True)
        else:
            st.write(turn["answer"])
            if turn.get("citations"):
                render_citations(turn["citations"])
        render_feedback(i)


# ── Input ──
query = None
if not st.session_state.history:
    st.markdown('<div class="search-wrap">', unsafe_allow_html=True)
    with st.form("center_search", clear_on_submit=True):
        center_q = st.text_input(
            label="", placeholder="Ask a FIFA World Cup question…",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask ⚽")
        if submitted and center_q.strip():
            query = center_q.strip()
    st.markdown('</div>', unsafe_allow_html=True)
    if not query and "pending_query" in st.session_state:
        query = st.session_state.pop("pending_query")
else:
    chat_q = st.chat_input("Ask a FIFA World Cup question…")
    if not chat_q and "pending_query" in st.session_state:
        chat_q = st.session_state.pop("pending_query")
    query = chat_q


# ── Run pipeline ──
if query:
    with st.chat_message("user"):
        st.write(query)

    with st.chat_message("assistant", avatar="⚽"):
        # Phase 1: rewrite + retrieve (show spinner)
        with st.spinner("Searching the archives…"):
            state = prepare(query)

        refused   = state.get("refused", False)
        top_score = state.get("top_score", 0.0)

        if refused:
            st.markdown(f'<div class="refusal-box">⚠️&nbsp; {REFUSAL_RESPONSE}</div>',
                        unsafe_allow_html=True)
            answer, citations = REFUSAL_RESPONSE, []
        else:
            # Phase 2: stream the answer token-by-token
            generation_query = state.get("rewritten_query") or query
            gen, citations, _ = get_generator().stream_generate(generation_query, state)
            answer = st.write_stream(gen)
            if citations:
                render_citations(citations)

    st.session_state.history.append({
        "query":     query,
        "answer":    answer,
        "citations": citations,
        "refused":   refused,
        "top_score": top_score,
    })
    st.rerun()
