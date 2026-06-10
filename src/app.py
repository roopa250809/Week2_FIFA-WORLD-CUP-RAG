import streamlit as st
from src.graph import prepare
from src.generate import AnswerGenerator, REFUSAL_RESPONSE

st.set_page_config(page_title="FIFA World Cup Expert", page_icon="⚽", layout="centered")

st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&display=swap');

  * { font-family: 'Inter', sans-serif; }

  /* ── Background ── */
  .stApp {
      background:
          radial-gradient(ellipse at 20% 20%, rgba(255,215,0,0.06) 0%, transparent 50%),
          radial-gradient(ellipse at 80% 80%, rgba(74,158,218,0.06) 0%, transparent 50%),
          linear-gradient(160deg, #060e1c 0%, #0f1f38 40%, #071526 100%);
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
  [data-testid="stChatInput"] {
      background: rgba(255,255,255,0.04) !important;
      border: 1px solid rgba(255,215,0,0.25) !important;
      border-radius: 14px !important;
  }
  [data-testid="stChatInput"] textarea {
      color: #e8f0f8 !important;
      background: transparent !important;
  }
  [data-testid="stChatInput"] textarea::placeholder { color: #4a6a8a !important; }

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

  /* ── Confidence badge ── */
  .conf-badge {
      display: inline-flex;
      align-items: center;
      gap: 5px;
      padding: 3px 11px;
      border-radius: 20px;
      font-size: 0.73rem;
      font-weight: 600;
      letter-spacing: 0.3px;
      margin-top: 6px;
  }
  .conf-high { background: rgba(34,197,94,0.12);  color: #4ade80; border: 1px solid rgba(34,197,94,0.25); }
  .conf-med  { background: rgba(234,179,8,0.12);  color: #facc15; border: 1px solid rgba(234,179,8,0.25); }
  .conf-low  { background: rgba(239,68,68,0.12);  color: #f87171; border: 1px solid rgba(239,68,68,0.25); }

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

  p, li { color: #c8dcef; }
  a { color: #4a9eda !important; }
  a:hover { color: #FFD700 !important; }
</style>

<div class="orb orb-gold"></div>
<div class="orb orb-blue"></div>
""", unsafe_allow_html=True)

# ── Hero ──
st.markdown("""
<div class="hero">
  <div class="ball">⚽</div>
  <h1>FIFA World Cup Expert</h1>
  <p class="subtitle">Tournament history · Iconic players · Legendary moments &nbsp;·&nbsp; 1930 – 2022</p>
</div>
<hr class="gold-divider">
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

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("Claude · Pinecone · LangGraph")


@st.cache_resource(show_spinner="Loading RAG pipeline...")
def get_generator():
    return AnswerGenerator()


def confidence_badge(score: float) -> str:
    if score >= 0.65:
        cls, dot, label = "conf-high", "●", f"High confidence · {score:.2f}"
    elif score >= 0.50:
        cls, dot, label = "conf-med",  "●", f"Medium confidence · {score:.2f}"
    else:
        cls, dot, label = "conf-low",  "●", f"Low confidence · {score:.2f}"
    return f'<span class="conf-badge {cls}">{dot} {label}</span>'


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


# ── Render history ──
for turn in st.session_state.history:
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
            st.markdown(confidence_badge(turn.get("top_score", 0)), unsafe_allow_html=True)


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
            st.markdown(confidence_badge(top_score), unsafe_allow_html=True)

    st.session_state.history.append({
        "query":     query,
        "answer":    answer,
        "citations": citations,
        "refused":   refused,
        "top_score": top_score,
    })
