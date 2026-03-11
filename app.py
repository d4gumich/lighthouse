# =========================
# Imports (Unsloth MUST be first)
# =========================
import unsloth
from unsloth import FastLanguageModel

import os
import faiss
import pandas as pd
import numpy as np
import fitz  # PyMuPDF
import streamlit as st
import time

from sentence_transformers import SentenceTransformer
from transformers import pipeline

# =========================
# Page config — MUST be first Streamlit call
# =========================
st.set_page_config(
    page_title="SkillMatch · AI Job Recommender",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# =========================
# Styles
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600&family=Libre+Baskerville:ital,wght@0,400;0,700;1,400&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Reset & base ── */
*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

:root {
    --bg:        #080808;
    --bg2:       #0E0E0E;
    --bg3:       #141414;
    --border:    #1C1C1C;
    --border2:   #252525;
    --amber:     #D4A853;
    --amber-dim: #8A6A2E;
    --amber-glow:rgba(212,168,83,0.08);
    --text:      #E8E0D0;
    --text2:     #7A7468;
    --text3:     #3D3A35;
    --green:     #4A7C59;
    --green-lt:  #6BAF80;
    --red:       #8B3A3A;
}

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: var(--bg) !important;
    color: var(--text);
}

.main { background: var(--bg) !important; }
.block-container {
    padding: 0 !important;
    max-width: 100% !important;
}

/* Hide Streamlit chrome */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton, [data-testid="stToolbar"] { display: none !important; }
section[data-testid="stSidebar"] { display: none; }

/* ── Layout shell ── */
.shell {
    min-height: 100vh;
    display: grid;
    grid-template-rows: auto 1fr;
    background: var(--bg);
}

/* ── Top bar ── */
.topbar {
    border-bottom: 1px solid var(--border);
    padding: 0 3rem;
    height: 56px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: var(--bg);
    position: sticky;
    top: 0;
    z-index: 100;
}
.topbar-logo {
    font-family: 'IBM Plex Mono', monospace;
    font-weight: 600;
    font-size: 1rem;
    color: var(--amber);
    letter-spacing: 0.05em;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.topbar-logo .sep { color: var(--text3); }
.topbar-tagline {
    font-size: 0.72rem;
    color: var(--text2);
    font-family: 'IBM Plex Mono', monospace;
    letter-spacing: 0.08em;
}
.topbar-status {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    color: var(--green-lt);
}
.status-dot {
    width: 6px; height: 6px;
    border-radius: 50%;
    background: var(--green-lt);
    box-shadow: 0 0 6px var(--green-lt);
    animation: pulse 2s infinite;
}
@keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
}

/* ── Main grid ── */
.main-grid {
    display: grid;
    grid-template-columns: 420px 1fr;
    min-height: calc(100vh - 56px);
}

/* ── Left panel ── */
.left-panel {
    border-right: 1px solid var(--border);
    padding: 2.5rem 2rem;
    background: var(--bg2);
    display: flex;
    flex-direction: column;
    gap: 1.5rem;
}

/* ── Right panel ── */
.right-panel {
    padding: 2.5rem 3rem;
    overflow-y: auto;
}

/* ── Section headers ── */
.section-head {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    font-weight: 500;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}
.section-head::after {
    content: '';
    flex: 1;
    height: 1px;
    background: var(--border);
}

/* ── Upload zone ── */
.upload-zone {
    border: 1px dashed var(--border2);
    border-radius: 8px;
    padding: 1.5rem;
    text-align: center;
    background: var(--bg3);
    transition: border-color 0.2s;
    cursor: pointer;
}
.upload-zone:hover { border-color: var(--amber-dim); }
.upload-icon {
    font-size: 1.8rem;
    margin-bottom: 0.5rem;
    opacity: 0.5;
}
.upload-label {
    font-size: 0.82rem;
    color: var(--text2);
    line-height: 1.5;
}

/* ── Text area ── */
.stTextArea textarea {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 6px !important;
    color: var(--text) !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.85rem !important;
    line-height: 1.6 !important;
    resize: vertical !important;
}
.stTextArea textarea:focus {
    border-color: var(--amber-dim) !important;
    box-shadow: 0 0 0 2px var(--amber-glow) !important;
    outline: none !important;
}
.stTextArea textarea::placeholder { color: var(--text3) !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: var(--bg3) !important;
    border: 1px dashed var(--border2) !important;
    border-radius: 8px !important;
    padding: 0.5rem !important;
}
[data-testid="stFileUploader"] label { color: var(--text2) !important; font-size: 0.82rem !important; }
[data-testid="stFileUploader"] button {
    background: transparent !important;
    border: 1px solid var(--border2) !important;
    color: var(--text2) !important;
    font-size: 0.78rem !important;
    border-radius: 4px !important;
}

/* ── Primary button ── */
.stButton > button {
    background: var(--amber) !important;
    color: #080808 !important;
    border: none !important;
    border-radius: 6px !important;
    font-family: 'IBM Plex Mono', monospace !important;
    font-weight: 600 !important;
    font-size: 0.82rem !important;
    letter-spacing: 0.06em !important;
    padding: 0.7rem 1.5rem !important;
    width: 100% !important;
    transition: all 0.15s !important;
    text-transform: uppercase !important;
}
.stButton > button:hover {
    background: #E8BC6A !important;
    box-shadow: 0 0 20px rgba(212,168,83,0.25) !important;
    transform: translateY(-1px) !important;
}
.stButton > button:disabled {
    background: var(--border2) !important;
    color: var(--text3) !important;
    cursor: not-allowed !important;
    transform: none !important;
    box-shadow: none !important;
}

/* ── Radio ── */
div[data-testid="stRadio"] > div {
    gap: 0.5rem !important;
    flex-direction: row !important;
}
div[data-testid="stRadio"] label {
    background: var(--bg3) !important;
    border: 1px solid var(--border2) !important;
    border-radius: 4px !important;
    padding: 0.4rem 1rem !important;
    font-size: 0.78rem !important;
    color: var(--text2) !important;
    cursor: pointer !important;
    transition: all 0.15s !important;
    font-family: 'IBM Plex Mono', monospace !important;
}
div[data-testid="stRadio"] label:has(input:checked) {
    border-color: var(--amber) !important;
    color: var(--amber) !important;
    background: var(--amber-glow) !important;
}

/* ── Slider ── */
.stSlider [data-baseweb="slider"] div[role="slider"] {
    background: var(--amber) !important;
    border-color: var(--amber) !important;
}
.stSlider [data-baseweb="slider"] div[data-testid="stSliderThumb"] {
    background: var(--amber) !important;
}

/* ── Progress bar ── */
.stProgress > div > div > div {
    background: var(--amber) !important;
}

/* ── Profile card ── */
.profile-card {
    background: var(--bg3);
    border: 1px solid var(--border2);
    border-radius: 8px;
    padding: 1.2rem 1.4rem;
    margin-bottom: 1.5rem;
}
.profile-name {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.3rem;
    color: var(--text);
    margin-bottom: 0.2rem;
}
.profile-meta {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text2);
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}
.profile-skills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
}
.skill-tag {
    background: var(--amber-glow);
    border: 1px solid var(--amber-dim);
    color: var(--amber);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.68rem;
    padding: 0.25rem 0.6rem;
    border-radius: 3px;
    letter-spacing: 0.04em;
}

/* ── Job cards ── */
.jobs-header {
    display: flex;
    align-items: baseline;
    justify-content: space-between;
    margin-bottom: 1.2rem;
}
.jobs-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.4rem;
    color: var(--text);
}
.jobs-count {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text2);
}

.job-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: 8px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 0.75rem;
    transition: border-color 0.15s, background 0.15s;
    position: relative;
    overflow: hidden;
}
.job-card::before {
    content: '';
    position: absolute;
    left: 0; top: 0; bottom: 0;
    width: 3px;
    background: var(--border);
    transition: background 0.15s;
}
.job-card:hover { border-color: var(--border2); background: var(--bg3); }
.job-card:hover::before { background: var(--amber); }

.job-card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 0.5rem;
}
.job-title-text {
    font-family: 'Libre Baskerville', serif;
    font-size: 1rem;
    color: var(--text);
    font-weight: 700;
}
.match-badge {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 0.2rem 0.55rem;
    border-radius: 3px;
    letter-spacing: 0.04em;
    flex-shrink: 0;
}
.match-high {
    background: rgba(74,124,89,0.15);
    border: 1px solid rgba(107,175,128,0.3);
    color: var(--green-lt);
}
.match-mid {
    background: rgba(212,168,83,0.1);
    border: 1px solid rgba(212,168,83,0.25);
    color: var(--amber);
}
.match-low {
    background: rgba(60,57,53,0.4);
    border: 1px solid var(--border2);
    color: var(--text2);
}
.job-company {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.72rem;
    color: var(--text2);
    letter-spacing: 0.05em;
    margin-bottom: 0.8rem;
}
.job-desc {
    font-size: 0.83rem;
    color: var(--text2);
    line-height: 1.6;
    margin-bottom: 0.9rem;
}
.job-skills {
    display: flex;
    flex-wrap: wrap;
    gap: 0.35rem;
}
.job-skill-tag {
    background: var(--bg);
    border: 1px solid var(--border2);
    color: var(--text2);
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    padding: 0.18rem 0.5rem;
    border-radius: 3px;
    letter-spacing: 0.03em;
}
.score-bar-bg {
    background: var(--border);
    border-radius: 2px;
    height: 2px;
    margin-top: 1rem;
    overflow: hidden;
}
.score-bar-fill {
    height: 2px;
    border-radius: 2px;
    background: linear-gradient(90deg, var(--amber-dim), var(--amber));
    transition: width 0.6s ease;
}

/* ── Recommendation block ── */
.rec-section {
    margin-top: 2rem;
    border-top: 1px solid var(--border);
    padding-top: 1.5rem;
}
.rec-label {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: var(--text3);
    margin-bottom: 1rem;
}
.rec-body {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-left: 3px solid var(--amber);
    border-radius: 0 6px 6px 0;
    padding: 1.4rem 1.6rem;
    font-size: 0.88rem;
    color: var(--text2);
    line-height: 1.8;
    white-space: pre-wrap;
    font-family: 'DM Sans', sans-serif;
}

/* ── Processing steps ── */
.step-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    padding: 0.5rem 0;
    font-family: 'IBM Plex Mono', monospace;
    font-size: 0.75rem;
    letter-spacing: 0.04em;
    transition: color 0.2s;
}
.step-row.pending { color: var(--text3); }
.step-row.active  { color: var(--text); }
.step-row.done    { color: var(--green-lt); }
.step-icon { width: 16px; text-align: center; flex-shrink: 0; }

/* ── Empty state ── */
.empty-state {
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 60vh;
    text-align: center;
    gap: 1rem;
}
.empty-glyph {
    font-family: 'IBM Plex Mono', monospace;
    font-size: 3rem;
    color: var(--text3);
    line-height: 1;
}
.empty-title {
    font-family: 'Libre Baskerville', serif;
    font-size: 1.2rem;
    color: var(--text3);
}
.empty-body {
    font-size: 0.82rem;
    color: var(--text3);
    max-width: 320px;
    line-height: 1.6;
}

/* ── Dividers ── */
hr { border: none; border-top: 1px solid var(--border) !important; margin: 1rem 0 !important; }

/* ── Selectbox ── */
div[data-baseweb="select"] > div {
    background: var(--bg3) !important;
    border-color: var(--border2) !important;
    color: var(--text) !important;
    font-size: 0.82rem !important;
    border-radius: 6px !important;
}

/* ── Labels ── */
label[data-testid="stWidgetLabel"] p {
    font-family: 'IBM Plex Mono', monospace !important;
    font-size: 0.68rem !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    color: var(--text3) !important;
}

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border2); border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Configuration
# =========================
BASE_MODEL  = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_PATH   = "."
DATA_PATH   = "job_skill_results.csv"
FAISS_DIR   = "data"
FAISS_PATH  = os.path.join(FAISS_DIR, "faiss.index")
TOP_K       = 5

os.makedirs(FAISS_DIR, exist_ok=True)

# =========================
# Session state
# =========================
for key, val in {
    "results": None,
    "candidate": None,
    "recommendation": None,
    "processing": False,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# =========================
# Cached model loading
# =========================
@st.cache_resource(show_spinner=False)
def load_models():
    """Load everything once — embedder, FAISS index, Llama + LoRA."""

    # ── Embedding model ──
    embedder = SentenceTransformer("BAAI/bge-large-en-v1.5", device="cpu")

    # ── Job data ──
    df = pd.read_csv(DATA_PATH)
    job_texts = (
        df["title"].fillna("") + " " +
        df["description"].fillna("") + " " +
        df["skill_names"].fillna("")
    ).tolist()

    # ── FAISS index ──
    if os.path.exists(FAISS_PATH):
        index = faiss.read_index(FAISS_PATH)
    else:
        embeddings = embedder.encode(job_texts, convert_to_numpy=True, show_progress_bar=False)
        faiss.normalize_L2(embeddings)
        index = faiss.IndexFlatIP(embeddings.shape[1])
        index.add(embeddings)
        faiss.write_index(index, FAISS_PATH)

    # ── Llama + LoRA ──
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        load_in_4bit=True,
        max_seq_length=4096,
    )
    model.load_adapter(LORA_PATH, adapter_name="job_lora")
    model.set_adapter("job_lora")
    FastLanguageModel.for_inference(model)

    llm = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=400,
        temperature=0.4,
        do_sample=False,
    )

    return embedder, index, df, llm


# =========================
# Core pipeline functions
# =========================
def extract_text_from_pdf(uploaded_file) -> str:
    """PyMuPDF — runs on the Space, no external calls needed."""
    data = uploaded_file.read()
    with fitz.open(stream=data, filetype="pdf") as doc:
        return "\n".join(page.get_text() for page in doc)


def extract_skills(llm, resume_text: str) -> str:
    prompt = (
        "Extract ONLY the skills explicitly mentioned in the resume. "
        "Return ONLY a comma-separated list. Do not add explanations.\n\n"
        f"{resume_text[:3000]}"
    )
    out = llm(prompt)[0]["generated_text"].strip()
    # Strip the prompt echo if model returns it
    if resume_text[:50] in out:
        out = out.split(resume_text[:50])[-1].strip()
    return out


def retrieve_jobs(embedder, index, df, candidate_skills: str, k: int = TOP_K):
    query_emb = embedder.encode([candidate_skills], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "title":       df.iloc[idx]["title"],
            "description": df.iloc[idx]["description"],
            "skills":      df.iloc[idx]["skill_names"],
            "score":       float(score),
        })
    return results


def recommend_jobs(llm, candidate_skills: str, jobs: list) -> str:
    prompt = f"Candidate skills: {candidate_skills}\n\n"
    prompt += (
        "Based on the following job matches, summarize each role and "
        "recommend additional job titles requiring similar skills:\n\n"
    )
    for job in jobs:
        prompt += (
            f"Title: {job['title']}\n"
            f"Description: {job['description']}\n"
            f"Skills: {job['skills']}\n\n"
        )
    out = llm(prompt)[0]["generated_text"]
    # Return only the generated part
    if prompt.strip() in out:
        out = out.replace(prompt.strip(), "").strip()
    return out


def score_to_pct(score: float) -> int:
    """Cosine similarity → 0-100 display score."""
    return min(100, max(0, int((score + 1) / 2 * 100)))


def match_class(pct: int) -> str:
    if pct >= 75: return "match-high"
    if pct >= 55: return "match-mid"
    return "match-low"


# =========================
# TOP BAR
# =========================
st.markdown("""
<div class="topbar">
    <div class="topbar-logo">
        ◈ SKILLMATCH
        <span class="sep">|</span>
        <span style="font-weight:400;color:#7A7468;font-size:0.82rem;">AI Job Intelligence</span>
    </div>
    <div style="display:flex;gap:2rem;align-items:center;">
        <span class="topbar-tagline">Meta-Llama 3.1 8B · LoRA · FAISS · T4 GPU</span>
        <div class="topbar-status">
            <div class="status-dot"></div>
            LIVE
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# =========================
# TWO-COLUMN LAYOUT
# =========================
left, right = st.columns([4, 6], gap="large")

# ══════════════════════════════════════════════
# LEFT PANEL — Input
# ══════════════════════════════════════════════
with left:
    st.markdown('<div style="padding: 2.5rem 2rem 0 2rem;">', unsafe_allow_html=True)

    # Input mode toggle
    st.markdown('<div class="section-head">01 — Input Method</div>', unsafe_allow_html=True)
    mode = st.radio("mode", ["Upload PDF", "Paste Text"], label_visibility="collapsed", horizontal=True)

    st.markdown('<div class="section-head" style="margin-top:1.5rem;">02 — Resume</div>', unsafe_allow_html=True)

    resume_text = ""

    if mode == "Upload PDF":
        uploaded = st.file_uploader(
            "Drop PDF resume",
            type=["pdf"],
            label_visibility="collapsed",
        )
        if uploaded is not None:
            with st.spinner(""):
                resume_text = extract_text_from_pdf(uploaded)
            char_count = len(resume_text)
            st.markdown(f"""
            <div style="margin-top:0.5rem;font-family:'IBM Plex Mono',monospace;
            font-size:0.7rem;color:#4A7C59;letter-spacing:0.06em;">
            ✓ {uploaded.name} — {char_count:,} chars extracted
            </div>
            """, unsafe_allow_html=True)
    else:
        resume_text = st.text_area(
            "Resume text",
            height=240,
            placeholder="Paste resume text here...\n\nWork experience, skills, education, projects.",
            label_visibility="collapsed",
        )

    # Search settings
    st.markdown('<div class="section-head" style="margin-top:1.5rem;">03 — Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Results to return", min_value=3, max_value=10, value=5, label_visibility="visible")

    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)

    # CTA button
    ready = bool(resume_text.strip())
    run_btn = st.button(
        "◈  ANALYSE RESUME",
        disabled=not ready,
    )

    if not ready:
        st.markdown("""
        <div style="text-align:center;font-family:'IBM Plex Mono',monospace;
        font-size:0.68rem;color:#3D3A35;letter-spacing:0.08em;margin-top:0.5rem;">
        ADD RESUME TO CONTINUE
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ══════════════════════════════════════════════
# RIGHT PANEL — Output
# ══════════════════════════════════════════════
with right:
    st.markdown('<div style="padding: 2.5rem 3rem 2.5rem 2rem;">', unsafe_allow_html=True)

    # ── Processing ──────────────────────────────
    if run_btn and ready:
        st.markdown('<div class="section-head">Processing</div>', unsafe_allow_html=True)

        steps = [
            "Loading models into memory",
            "Extracting skills from resume",
            "Running FAISS semantic retrieval",
            "Generating recommendations",
        ]
        placeholders = [st.empty() for _ in steps]
        prog = st.progress(0)

        def render_step(idx, state):
            icons = {"pending": "○", "active": "◌", "done": "◈"}
            icon = icons[state]
            placeholders[idx].markdown(
                f'<div class="step-row {state}"><span class="step-icon">{icon}</span>{steps[idx]}</div>',
                unsafe_allow_html=True,
            )

        for i in range(len(steps)):
            render_step(i, "pending")

        # Step 1 — load models
        render_step(0, "active")
        prog.progress(10)
        embedder, index, df, llm = load_models()
        render_step(0, "done")
        prog.progress(25)

        # Step 2 — skill extraction
        render_step(1, "active")
        prog.progress(35)
        skills_raw = extract_skills(llm, resume_text)
        skills_list = [s.strip() for s in skills_raw.split(",") if s.strip()]
        render_step(1, "done")
        prog.progress(55)

        # Step 3 — retrieval
        render_step(2, "active")
        prog.progress(65)
        job_results = retrieve_jobs(embedder, index, df, skills_raw, k=top_k)
        render_step(2, "done")
        prog.progress(80)

        # Step 4 — recommendation
        render_step(3, "active")
        prog.progress(90)
        recommendation = recommend_jobs(llm, skills_raw, job_results)
        render_step(3, "done")
        prog.progress(100)

        st.session_state.results = job_results
        st.session_state.candidate = {"skills": skills_list, "skills_raw": skills_raw}
        st.session_state.recommendation = recommendation
        time.sleep(0.3)
        st.rerun()

    # ── Results ─────────────────────────────────
    elif st.session_state.results is not None:
        candidate  = st.session_state.candidate
        job_results = st.session_state.results
        recommendation = st.session_state.recommendation
        skills_list = candidate.get("skills", [])

        # ── Extracted skills ──
        st.markdown('<div class="section-head">Extracted Skills</div>', unsafe_allow_html=True)
        chips = "".join(
            f'<span class="skill-tag">{s}</span>'
            for s in skills_list[:20]
        )
        st.markdown(
            f'<div style="display:flex;flex-wrap:wrap;gap:0.4rem;margin-bottom:1.5rem;">{chips}</div>',
            unsafe_allow_html=True,
        )

        # ── Job matches ──
        st.markdown(
            f'<div class="jobs-header">'
            f'<div class="jobs-title">Top Matches</div>'
            f'<div class="jobs-count">{len(job_results)} results · FAISS cosine</div>'
            f'</div>',
            unsafe_allow_html=True,
        )

        for i, job in enumerate(job_results):
            pct = score_to_pct(job["score"])
            mc  = match_class(pct)
            skill_tags = "".join(
                f'<span class="job-skill-tag">{s.strip()}</span>'
                for s in str(job["skills"]).split(",")[:6]
                if s.strip()
            )
            desc_preview = str(job["description"])[:160].strip()
            if len(str(job["description"])) > 160:
                desc_preview += "…"

            st.markdown(f"""
            <div class="job-card">
                <div class="job-card-top">
                    <div class="job-title-text">{job['title']}</div>
                    <div class="match-badge {mc}">{pct}% match</div>
                </div>
                <div class="job-company">RANK #{i+1} &nbsp;·&nbsp; SCORE {job['score']:.4f}</div>
                <div class="job-desc">{desc_preview}</div>
                <div class="job-skills">{skill_tags}</div>
                <div class="score-bar-bg">
                    <div class="score-bar-fill" style="width:{pct}%;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── LLM Recommendation ──
        st.markdown(f"""
        <div class="rec-section">
            <div class="rec-label">◈ &nbsp; AI Recommendation — Llama 3.1 8B + LoRA</div>
            <div class="rec-body">{recommendation}</div>
        </div>
        """, unsafe_allow_html=True)

        # Reset button
        st.markdown("<div style='margin-top:2rem;'>", unsafe_allow_html=True)
        if st.button("↺  NEW SEARCH"):
            st.session_state.results = None
            st.session_state.candidate = None
            st.session_state.recommendation = None
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    # ── Empty state ──────────────────────────────
    else:
        st.markdown("""
        <div class="empty-state">
            <div class="empty-glyph">◈</div>
            <div class="empty-title">Ready to match</div>
            <div class="empty-body">
                Upload a PDF resume or paste text on the left.<br>
                The model will extract your skills and find the most
                relevant roles from the job database.
            </div>
            <div style="font-family:'IBM Plex Mono',monospace;font-size:0.68rem;
            color:#3D3A35;letter-spacing:0.1em;margin-top:1rem;">
                LLAMA 3.1 8B · LORA · FAISS · T4 GPU
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)