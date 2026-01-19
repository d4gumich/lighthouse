# =========================
# Imports (Unsloth MUST be first)
# =========================
import unsloth
from unsloth import FastLanguageModel

import os
import torch
import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================
# Configuration
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO = "./job_reco_lora"   # <-- change to your HF LoRA repo

DATA_PATH  = "job_skill_results.csv"
FAISS_DIR  = "data"
FAISS_PATH = os.path.join(FAISS_DIR, "faiss.index")

TOP_K = 3


# =========================
# Ensure writable dirs
# =========================
os.makedirs(FAISS_DIR, exist_ok=True)


# =========================
# Load Embedding Model (CPU)
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cpu"
)


# =========================
# Load Job Data
# =========================
assert os.path.exists(DATA_PATH), f"Missing file: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)


# =========================
# Build / Load FAISS Index
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
# Load Base Model + LoRA (GPU REQUIRED)
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=4096,
)

model.load_adapter(LORA_REPO, adapter_name="job_lora")
model.set_adapter("job_lora")

FastLanguageModel.for_inference(model)


# =========================
# Text Generation Pipeline
# =========================
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    max_new_tokens=400,
    temperature=0.4,
    do_sample=False,
)


# =========================
# Skill Extraction (LoRA)
# =========================
def extract_skills(resume_text: str) -> str:
    prompt = (
        "Extract ONLY the skills explicitly mentioned in the resume. "
        "Return ONLY a comma-separated list. Do not add explanations.\n\n"
        f"{resume_text}"
    )

    return llm(prompt)[0]["generated_text"].strip()


# =========================
# RAG Retrieval
# =========================
def retrieve_jobs(candidate_skills: str, k: int = TOP_K):
    query_emb = embedder.encode([candidate_skills], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)

    scores, indices = index.search(query_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "title": df.iloc[idx]["title"],
            "description": df.iloc[idx]["description"],
            "skills": df.iloc[idx]["skill_names"],
            "score": float(score),
        })

    return results


# =========================
# Recommendation Generation
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
# End-to-End Pipeline
# =========================
def run_pipeline(resume_text: str):
    skills = extract_skills(resume_text)
    jobs = retrieve_jobs(skills)
    recommendations = recommend_jobs(skills, jobs)

    return {
        "extracted_skills": skills,
        "top_jobs": jobs,
        "recommendations": recommendations,
    }


# =========================
# Local Test (HF ignores this)
# =========================
if __name__ == "__main__":
    test_resume = """
    Data Scientist with experience in NLP, LLMs, Python, SQL,
    RAG pipelines, FAISS, transformers, and machine learning.
    """

    output = run_pipeline(test_resume)
    print(output)
