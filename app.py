# =========================
# Imports (Unsloth MUST be first)
# =========================
import unsloth
from unsloth import FastLanguageModel

import os
import faiss
import pandas as pd
import numpy as np

from peft import PeftModel
from sentence_transformers import SentenceTransformer
from transformers import pipeline
import gradio as gr

# =========================
# Configuration
# =========================
# FIX 1: use 4-bit quantized version — loads on T4 GPU
BASE_MODEL = "unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit"

# FIX 2: load adapter from your HF model repo
LORA_REPO  = "Data4GoodCenter/skillmatch-llama3-8b-lora"
HF_TOKEN   = os.environ.get("HF_TOKEN")   # set in Space secrets

DATA_PATH  = "job_skill_results.csv"
FAISS_DIR  = "data"
FAISS_PATH = os.path.join(FAISS_DIR, "faiss.index")
TOP_K      = 3

# =========================
# Prepare directories
# =========================
os.makedirs(FAISS_DIR, exist_ok=True)
assert os.path.exists(DATA_PATH), f"CSV not found: {DATA_PATH}"

# =========================
# Load embedding model (CPU)
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cpu"
)

# =========================
# Load job data
# =========================
df = pd.read_csv(DATA_PATH)

# =========================
# Build / Load FAISS index
# =========================
job_texts = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["skill_names"].fillna("")
).tolist()

if os.path.exists(FAISS_PATH):
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)
else:
    print("Building FAISS index...")
    embeddings = embedder.encode(
        job_texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)

# =========================
# Load base model + LoRA
# FIX 3: use PeftModel.from_pretrained to load from HF Hub
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name     = BASE_MODEL,
    load_in_4bit   = True,
    max_seq_length = 4096,
)

# Load your LoRA adapter from HF model repo
model = PeftModel.from_pretrained(
    model,
    LORA_REPO,
    token = HF_TOKEN
)
FastLanguageModel.for_inference(model)

print("✓ Model + LoRA adapter loaded successfully")

# =========================
# Text generation pipeline
# =========================
llm = pipeline(
    "text-generation",
    model          = model,
    tokenizer      = tokenizer,
    max_new_tokens = 400,
    temperature    = 0.4,
    do_sample      = False,
)

# =========================
# Skill extraction
# =========================
def extract_skills(resume_text: str) -> str:
    prompt = (
        "Extract ONLY the skills explicitly mentioned in the resume. "
        "Return ONLY a comma-separated list. Do not add explanations.\n\n"
        f"{resume_text}"
    )
    return llm(prompt)[0]["generated_text"].strip()

# =========================
# RAG retrieval
# =========================
def retrieve_jobs(candidate_skills: str, k: int = TOP_K):
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

# =========================
# Recommendation generation
# =========================
def recommend_jobs(candidate_skills: str, jobs: list) -> str:
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
    return llm(prompt)[0]["generated_text"]

# =========================
# End-to-end pipeline
# =========================
def run_pipeline(resume_text: str):
    skills          = extract_skills(resume_text)
    jobs            = retrieve_jobs(skills)
    recommendations = recommend_jobs(skills, jobs)
    return {
        "extracted_skills": skills,
        "top_jobs":         jobs,
        "recommendations":  recommendations,
    }

# =========================
# Gradio UI — unchanged from original
# =========================
def gradio_pipeline(resume_text):
    result = run_pipeline(resume_text)
    return result['extracted_skills'], result['top_jobs'], result['recommendations']

demo = gr.Interface(
    fn          = gradio_pipeline,
    inputs      = gr.Textbox(lines=10, placeholder="Paste resume text here"),
    outputs     = [
        gr.Textbox(label="Extracted Skills"),
        gr.JSON(label="Top Jobs"),
        gr.Textbox(label="Recommendations")
    ],
    title       = "Resume Job Recommendation",
    description = "Extract skills from a resume and recommend jobs using FAISS + LLM"
)

demo.launch()