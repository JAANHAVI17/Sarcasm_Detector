import streamlit as st
import pickle
import time
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sarcasm Detector · AI",
    page_icon="◈",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Load model & tokenizer ────────────────────────────────────────────────────
@st.cache_resource
def load_resources():
    model = load_model("sarcasm_model.h5", compile=False)
    with open("tokenizer.pickle", "rb") as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

model, tokenizer = load_resources()
MAX_LEN = 100

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:ital,wght@0,400;0,700;1,400&family=Rajdhani:wght@400;500;600;700&display=swap');

/* ── Variables ── */
:root {
  --bg:      #080b14;
  --bg2:     #0d1120;
  --cyan:    #00f5ff;
  --magenta: #ff2d78;
  --green:   #00ff9d;
  --text:    #e2eaf8;
  --mid:     #7a8aaa;
  --dim:     #3d4a66;
  --border:  rgba(0,245,255,0.18);
}

/* ── Base ── */
html, body, [class*="css"] {
  font-family: 'Rajdhani', sans-serif !important;
  background-color: var(--bg) !important;
  color: var(--text) !important;
}

/* ── Animated grid background ── */
.stApp {
  background-color: var(--bg) !important;
  background-image:
    linear-gradient(rgba(0,245,255,0.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(0,245,255,0.03) 1px, transparent 1px) !important;
  background-size: 40px 40px !important;
}

/* ── Radial glows ── */
.stApp::before {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background:
    radial-gradient(ellipse 55% 45% at 15% 10%, rgba(0,245,255,0.07) 0%, transparent 65%),
    radial-gradient(ellipse 50% 40% at 85% 20%, rgba(255,45,120,0.07) 0%, transparent 60%),
    radial-gradient(ellipse 40% 35% at 50% 90%, rgba(0,255,157,0.05) 0%, transparent 55%);
}

/* ── Scanlines ── */
.stApp::after {
  content: "";
  position: fixed; inset: 0; pointer-events: none; z-index: 0;
  background: repeating-linear-gradient(
    0deg, transparent, transparent 2px,
    rgba(0,0,0,0.05) 2px, rgba(0,0,0,0.05) 4px
  );
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container {
  max-width: 760px !important;
  padding: 0 2rem 4rem !important;
}

/* ── Remove default padding on elements ── */
.element-container { margin-bottom: 0 !important; }

/* ── Collapse gap between stacked h1s ── */
.element-container + .element-container { margin-top: 0 !important; }
div[data-testid="stMarkdownContainer"] h1 { margin: 0 !important; padding: 0 !important; }
div[data-testid="stVerticalBlock"] > div { gap: 0 !important; }

/* ══════════════════════════════════════════════════════
   TEXTAREA — style Streamlit's native component
══════════════════════════════════════════════════════ */
.stTextArea > div > div {
  background: var(--bg2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  padding: 0 !important;
  position: relative !important;
  box-shadow:
    0 0 0 1px rgba(0,245,255,0.04),
    0 4px 24px rgba(0,0,0,0.4),
    inset 0 1px 0 rgba(0,245,255,0.06) !important;
}

/* Top neon glow line on the card */
.stTextArea > div > div::before {
  content: "" !important;
  position: absolute !important;
  top: 0; left: 20%; right: 20%; height: 1px !important;
  background: linear-gradient(90deg, transparent, var(--cyan), transparent) !important;
  filter: blur(1px) !important;
}

/* The "// INPUT" label tag */
.stTextArea > label {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.55rem !important;
  letter-spacing: 0.25em !important;
  text-transform: uppercase !important;
  color: var(--cyan) !important;
  background: var(--bg2) !important;
  padding: 0 0.5rem !important;
  margin-bottom: -1px !important;
  display: inline-block !important;
  position: relative !important;
  z-index: 1 !important;
}

.stTextArea textarea {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.85rem !important;
  color: var(--text) !important;
  background: var(--bg) !important;
  border: 1px solid rgba(0,245,255,0.1) !important;
  border-radius: 3px !important;
  padding: 1rem 1.1rem !important;
  line-height: 1.7 !important;
  caret-color: var(--cyan) !important;
  transition: border-color 0.25s, box-shadow 0.25s !important;
  resize: none !important;
  margin: 0.5rem !important;
  width: calc(100% - 1rem) !important;
}
.stTextArea textarea:focus {
  border-color: rgba(0,245,255,0.45) !important;
  box-shadow: 0 0 0 3px rgba(0,245,255,0.07), 0 0 14px rgba(0,245,255,0.07) !important;
  outline: none !important;
}
.stTextArea textarea::placeholder {
  color: var(--dim) !important;
  font-style: italic !important;
}

/* ══════════════════════════════════════════════════════
   BUTTON — style Streamlit's native component
══════════════════════════════════════════════════════ */
.stButton { margin-top: 0.75rem !important; }
.stButton > button {
  width: 100% !important;
  padding: 0.85rem 2rem !important;
  font-family: 'Rajdhani', sans-serif !important;
  font-size: 0.9rem !important;
  font-weight: 700 !important;
  letter-spacing: 0.22em !important;
  text-transform: uppercase !important;
  color: var(--bg) !important;
  background: var(--cyan) !important;
  border: none !important;
  border-radius: 3px !important;
  cursor: pointer !important;
  transition: box-shadow 0.2s, transform 0.15s !important;
  box-shadow: 0 0 18px rgba(0,245,255,0.35), 0 0 40px rgba(0,245,255,0.12) !important;
}
.stButton > button:hover {
  box-shadow: 0 0 28px rgba(0,245,255,0.6), 0 0 60px rgba(0,245,255,0.2) !important;
  transform: translateY(-1px) !important;
  opacity: 1 !important;
  color: var(--bg) !important;
  background: var(--cyan) !important;
  border: none !important;
}
.stButton > button:active { transform: translateY(0) !important; }
.stButton > button:focus {
  box-shadow: 0 0 18px rgba(0,245,255,0.35) !important;
  color: var(--bg) !important;
  background: var(--cyan) !important;
  border: none !important;
  outline: none !important;
}

/* ══════════════════════════════════════════════════════
   SPINNER
══════════════════════════════════════════════════════ */
.stSpinner > div {
  border-top-color: var(--cyan) !important;
}
.stSpinner p {
  font-family: 'Space Mono', monospace !important;
  font-size: 0.75rem !important;
  color: var(--mid) !important;
  letter-spacing: 0.08em !important;
}

/* ══════════════════════════════════════════════════════
   RESULT CARD
══════════════════════════════════════════════════════ */
.result-card {
  border-radius: 4px;
  padding: 2rem 2.25rem;
  margin-top: 0.5rem;
  position: relative; overflow: hidden;
  animation: cardReveal 0.5s cubic-bezier(0.34,1.56,0.64,1) forwards;
}
.result-card.sarcastic {
  background: var(--bg2);
  border: 1px solid rgba(255,45,120,0.4);
  box-shadow: 0 0 40px rgba(255,45,120,0.1), inset 0 0 50px rgba(255,45,120,0.04);
}
.result-card.genuine {
  background: var(--bg2);
  border: 1px solid rgba(0,255,157,0.35);
  box-shadow: 0 0 40px rgba(0,255,157,0.1), inset 0 0 50px rgba(0,255,157,0.04);
}
.result-card.sarcastic::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--magenta), transparent);
  filter: blur(1px);
}
.result-card.genuine::before {
  content: "";
  position: absolute; top: 0; left: 0; right: 0; height: 2px;
  background: linear-gradient(90deg, transparent, var(--green), transparent);
  filter: blur(1px);
}

.result-tag {
  font-family: 'Space Mono', monospace;
  font-size: 0.55rem; letter-spacing: 0.25em; text-transform: uppercase;
  margin-bottom: 1.2rem; display: block;
}
.result-tag.sarcastic { color: var(--magenta); }
.result-tag.genuine   { color: var(--green); }

.result-top { display: flex; align-items: center; gap: 1.2rem; margin-bottom: 1.4rem; }
.result-emoji {
  font-size: 2.8rem; line-height: 1; display: inline-block;
  animation: emojiBounce 0.5s cubic-bezier(0.34,1.56,0.64,1) 0.15s both;
}
.result-verdict {
  font-family: 'Rajdhani', sans-serif;
  font-size: 2.2rem; font-weight: 700;
  line-height: 1.1; letter-spacing: 0.04em; text-transform: uppercase;
}
.result-verdict.sarcastic { color: var(--magenta); text-shadow: 0 0 20px rgba(255,45,120,0.5); }
.result-verdict.genuine   { color: var(--green);   text-shadow: 0 0 20px rgba(0,255,157,0.4); }
.result-verdict-sub {
  display: block;
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.25em; text-transform: uppercase;
  color: var(--mid); margin-bottom: 0.15rem;
}

.conf-row {
  display: flex; justify-content: space-between;
  align-items: baseline; margin-bottom: 0.5rem;
}
.conf-label {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.2em; text-transform: uppercase; color: var(--mid);
}
.conf-pct { font-family: 'Space Mono', monospace; font-size: 1.15rem; font-weight: 700; }
.conf-pct.sarcastic { color: var(--magenta); }
.conf-pct.genuine   { color: var(--green); }

.bar-track {
  height: 6px; border-radius: 1px;
  background: rgba(255,255,255,0.05); overflow: hidden;
  border: 1px solid rgba(255,255,255,0.06);
}
.bar-fill {
  height: 100%; border-radius: 1px;
  animation: barGrow 1.2s cubic-bezier(0.4,0,0.2,1) 0.4s both;
  width: var(--bar-width);
}
.bar-fill.sarcastic {
  background: linear-gradient(90deg, rgba(255,45,120,0.4), var(--magenta));
  box-shadow: 0 0 10px rgba(255,45,120,0.7);
}
.bar-fill.genuine {
  background: linear-gradient(90deg, rgba(0,255,157,0.4), var(--green));
  box-shadow: 0 0 10px rgba(0,255,157,0.7);
}

.result-note {
  margin-top: 1.2rem;
  font-family: 'Space Mono', monospace;
  font-size: 0.72rem; line-height: 1.85; color: var(--mid);
  padding-top: 1rem;
  border-top: 1px solid rgba(255,255,255,0.06);
}
.result-note b { color: var(--cyan); }

/* ══════════════════════════════════════════════════════
   HOW IT WORKS
══════════════════════════════════════════════════════ */
.hiw { margin-top: 2.5rem; }
.hiw-title {
  font-family: 'Space Mono', monospace;
  font-size: 0.6rem; letter-spacing: 0.28em; text-transform: uppercase;
  color: var(--dim); margin-bottom: 1rem;
}
.steps-grid { display: grid; grid-template-columns: repeat(3,1fr); gap: 1rem; }
.step-card {
  background: var(--bg2); border: 1px solid rgba(0,245,255,0.1);
  border-radius: 4px; padding: 1.2rem 1rem; text-align: center;
  transition: border-color 0.25s, box-shadow 0.25s, transform 0.2s;
}
.step-card:hover {
  border-color: rgba(0,245,255,0.35); transform: translateY(-3px);
  box-shadow: 0 0 20px rgba(0,245,255,0.08);
}
.step-icon { font-size: 1.4rem; margin-bottom: 0.5rem; display: block; }
.step-name {
  font-family: 'Rajdhani', sans-serif; font-size: 0.8rem; font-weight: 700;
  letter-spacing: 0.14em; text-transform: uppercase; color: var(--cyan); margin-bottom: 0.3rem;
}
.step-desc {
  font-family: 'Space Mono', monospace; font-size: 0.62rem; color: var(--mid); line-height: 1.65;
}

/* ══════════════════════════════════════════════════════
   EXAMPLES
══════════════════════════════════════════════════════ */
.examples-section { margin-top: 2rem; }
.examples-title {
  font-family: 'Space Mono', monospace; font-size: 0.6rem;
  letter-spacing: 0.28em; text-transform: uppercase; color: var(--dim); margin-bottom: 0.75rem;
}
.example-pills { display: flex; flex-wrap: wrap; gap: 0.5rem; }
.example-pill {
  font-family: 'Space Mono', monospace; font-size: 0.65rem; color: var(--mid);
  background: var(--bg2); border: 1px solid rgba(0,245,255,0.12);
  border-radius: 2px; padding: 0.38rem 0.85rem;
  transition: border-color 0.2s, color 0.2s, box-shadow 0.2s;
}
.example-pill:hover {
  border-color: rgba(0,245,255,0.4); color: var(--cyan);
  box-shadow: 0 0 10px rgba(0,245,255,0.08);
}

/* ══════════════════════════════════════════════════════
   FOOTER
══════════════════════════════════════════════════════ */
.app-footer { text-align: center; padding: 2.5rem 0 1.5rem; }
.footer-rule {
  height: 1px;
  background: linear-gradient(90deg, transparent, rgba(0,245,255,0.2), transparent);
  margin-bottom: 1.5rem;
}
.footer-logo {
  font-family: 'Rajdhani', sans-serif; font-size: 1.1rem; font-weight: 700;
  letter-spacing: 0.15em; text-transform: uppercase;
  color: var(--cyan); text-shadow: 0 0 12px rgba(0,245,255,0.4); margin-bottom: 0.4rem;
}
.footer-meta {
  font-family: 'Space Mono', monospace; font-size: 0.62rem; color: var(--dim); line-height: 2;
}
.footer-meta a { color: var(--cyan); text-decoration: none; }

/* ══════════════════════════════════════════════════════
   KEYFRAMES
══════════════════════════════════════════════════════ */
@keyframes cardReveal {
  from { opacity: 0; transform: scale(0.96) translateY(10px); }
  to   { opacity: 1; transform: scale(1) translateY(0); }
}
@keyframes barGrow {
  from { width: 0; }
  to   { width: var(--bar-width); }
}
@keyframes emojiBounce {
  from { transform: scale(0.3) rotate(-20deg); opacity: 0; }
  to   { transform: scale(1) rotate(0deg); opacity: 1; }
}
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HEADER
# ══════════════════════════════════════════════════════
st.markdown("""
<div style="text-align:center; padding: 4rem 0 2.5rem; animation: fadeUp 0.8s ease forwards;">

  <p style="font-family:'Space Mono',monospace; font-size:0.65rem; letter-spacing:0.3em;
            text-transform:uppercase; color:#00f5ff; margin-bottom:1.5rem;">
    &gt; AI-Powered NLP &nbsp;·&nbsp; LSTM Model &nbsp;·&nbsp; v2.0
  </p>

  <h1 style="font-family:'Rajdhani',sans-serif; font-size:clamp(2.8rem,7vw,4.8rem);
             font-weight:700; line-height:1.05; letter-spacing:0.02em;
             text-transform:uppercase; color:#e2eaf8; margin:0 0 -0.15rem 0;">
    Detect The
  </h1>
  <h1 style="font-family:'Rajdhani',sans-serif; font-size:clamp(2.8rem,7vw,4.8rem);
             font-weight:700; line-height:1.05; letter-spacing:0.02em;
             text-transform:uppercase; margin:0;
             color:#00f5ff;
             text-shadow: 0 0 10px #00f5ff, 0 0 30px rgba(0,245,255,0.4), 0 0 60px rgba(0,245,255,0.15);">
    Sarcasm
  </h1>

  <p style="font-family:'Space Mono',monospace; font-size:0.75rem; color:#7a8aaa;
            line-height:1.8; max-width:400px; margin:1.5rem auto 0; letter-spacing:0.02em;">
    // Feed it a news headline. The LSTM neural network will decode if it's satire or straight journalism.
  </p>

  <div style="display:flex; justify-content:center; gap:0.55rem; flex-wrap:wrap; margin-top:1.6rem;">
    <span style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.12em;
                 text-transform:uppercase; padding:0.28rem 0.8rem; border-radius:2px;
                 background:rgba(0,245,255,0.06); color:#00f5ff; border:1px solid rgba(0,245,255,0.3);">
      🧠 LSTM · RNN
    </span>
    <span style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.12em;
                 text-transform:uppercase; padding:0.28rem 0.8rem; border-radius:2px;
                 background:rgba(0,245,255,0.06); color:#00f5ff; border:1px solid rgba(0,245,255,0.3);">
      📰 News Headlines
    </span>
    <span style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.12em;
                 text-transform:uppercase; padding:0.28rem 0.8rem; border-radius:2px;
                 background:rgba(0,245,255,0.06); color:#00f5ff; border:1px solid rgba(0,245,255,0.3);">
      ⚡ Real-Time
    </span>
    <span style="font-family:'Space Mono',monospace; font-size:0.6rem; letter-spacing:0.12em;
                 text-transform:uppercase; padding:0.28rem 0.8rem; border-radius:2px;
                 background:rgba(0,245,255,0.06); color:#00f5ff; border:1px solid rgba(0,245,255,0.3);">
      📊 Confidence Score
    </span>
  </div>

  <div style="display:flex; align-items:center; gap:0.75rem; margin-top:2rem;">
    <div style="flex:1; height:1px; background:linear-gradient(90deg,transparent,rgba(0,245,255,0.35),transparent);"></div>
    <span style="font-family:'Space Mono',monospace; font-size:0.65rem; color:#00f5ff;">◈</span>
    <div style="flex:1; height:1px; background:linear-gradient(90deg,transparent,rgba(0,245,255,0.35),transparent);"></div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  INPUT SECTION — using native Streamlit components
#  (label is styled via CSS to look like "// INPUT" tag)
# ══════════════════════════════════════════════════════
headline = st.text_area(
    label="// INPUT",
    placeholder="// paste your headline here...",
    height=120,
)

analyse = st.button("▶  Run Analysis")


# ══════════════════════════════════════════════════════
#  PREDICTION
# ══════════════════════════════════════════════════════
def predict_sarcasm(text: str):
    seq    = tokenizer.texts_to_sequences([text])
    padded = pad_sequences(seq, maxlen=MAX_LEN, padding="post", truncating="post")
    prob   = float(model.predict(padded, verbose=0)[0][0])
    return prob >= 0.5, prob


if analyse:
    if not headline.strip():
        st.markdown("""
        <div style="font-family:'Space Mono',monospace; font-size:0.72rem; color:#3d4a66;
                    text-align:center; margin-top:1rem; padding:1rem;
                    border:1px dashed rgba(0,245,255,0.15); border-radius:3px;">
          // no input detected — enter a headline above
        </div>""", unsafe_allow_html=True)
    else:
        with st.spinner("// processing input..."):
            time.sleep(0.4)
            is_sarcastic, confidence = predict_sarcasm(headline.strip())

        conf_pct = confidence * 100 if is_sarcastic else (1 - confidence) * 100
        card_cls = "sarcastic" if is_sarcastic else "genuine"
        emoji    = "🎭" if is_sarcastic else "📰"
        verdict  = "SARCASTIC" if is_sarcastic else "GENUINE"
        note     = (
            f"// model detected satirical patterns — irony, exaggeration or hyperbole flagged. confidence: <b>{conf_pct:.1f}%</b>"
            if is_sarcastic else
            f"// model returned low sarcasm signal — headline appears to be factual reporting. confidence: <b>{conf_pct:.1f}%</b>"
        )

        st.markdown(f"""
        <div class="result-card {card_cls}">
          <span class="result-tag {card_cls}">// ANALYSIS COMPLETE</span>
          <div class="result-top">
            <span class="result-emoji">{emoji}</span>
            <div>
              <span class="result-verdict-sub">Classification</span>
              <div class="result-verdict {card_cls}">{verdict}</div>
            </div>
          </div>
          <div class="conf-row">
            <span class="conf-label">Confidence Score</span>
            <span class="conf-pct {card_cls}">{conf_pct:.1f}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill {card_cls}" style="--bar-width:{conf_pct:.1f}%; width:{conf_pct:.1f}%;"></div>
          </div>
          <p class="result-note">{note}</p>
        </div>
        """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  HOW IT WORKS
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="hiw">
  <p class="hiw-title">// Pipeline</p>
  <div class="steps-grid">
    <div class="step-card">
      <span class="step-icon">✂️</span>
      <p class="step-name">Tokenise</p>
      <p class="step-desc">Headline split into word tokens via trained vocabulary.</p>
    </div>
    <div class="step-card">
      <span class="step-icon">🧩</span>
      <p class="step-name">Pad Seq</p>
      <p class="step-desc">Tokens padded to fixed length for model input.</p>
    </div>
    <div class="step-card">
      <span class="step-icon">🧠</span>
      <p class="step-name">LSTM</p>
      <p class="step-desc">Bi-directional LSTM reads context & outputs probability.</p>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  EXAMPLE HEADLINES
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="examples-section">
  <p class="examples-title">// Test Inputs</p>
  <div class="example-pills">
    <span class="example-pill">😏 Man Finally Discovers Coffee Actually Good</span>
    <span class="example-pill">📰 Senate Passes Budget Bill After Months of Debate</span>
    <span class="example-pill">😅 Nation's Experts Agree Watching TV All Day Fine</span>
    <span class="example-pill">📰 NASA Launches New Climate Satellite Into Orbit</span>
    <span class="example-pill">😏 Area Man Confident He Alone Has Read Constitution</span>
  </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════
#  FOOTER
# ══════════════════════════════════════════════════════
st.markdown("""
<div class="app-footer">
  <div class="footer-rule"></div>
  <p class="footer-logo">◈ Sarcasm Detector</p>
  <p class="footer-meta">
    Built with TensorFlow · Keras · Streamlit<br>
    Trained on <a href="https://www.kaggle.com/datasets/rmisra/news-headlines-dataset-for-sarcasm-detection" target="_blank">News Headlines Sarcasm Dataset</a>
  </p>
</div>
""", unsafe_allow_html=True)