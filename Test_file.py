# =========================
# GPU Sanity Check (MUST be first)
# =========================
import torch

assert torch.cuda.is_available(), "❌ GPU not detected! Make sure Space is NOT ZeroGPU."
print("✅ GPU detected:", torch.cuda.get_device_name(0))


# =========================
# Imports
# =========================
import unsloth
from unsloth import FastLanguageModel

import faiss
import pandas as pd
import numpy as np

from sentence_transformers import SentenceTransformer
from transformers import pipeline


# =========================
# Configuration
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO  = "your-username/job-reco-lora"  # <-- replace with your actual LoRA repo

DATA_PATH  = "data/job_skill_results.csv"
FAISS_PATH = "data/faiss.index"

TOP_K = 3
MAX_SEQ_LEN = 2048   # ✅ safer for T4


# =========================
# Load Embedding Model (CPU)
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-large-en-v1.5",
    device="cpu"
)


# =========================
# Load Job Data + FAISS Index
# =========================
df = pd.read_csv(DATA_PATH)

index = faiss.read_index(FAISS_PATH)

# Optional sanity check
print("FAISS index type:", type(index))


# =========================
# Load Base Model + LoRA (GPU)
# =========================
model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=BASE_MODEL,
    load_in_4bit=True,
    max_seq_length=MAX_SEQ_LEN,
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
    return_full_text=False,
    pad_token_id=tokenizer.eos_token_id,
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

    generated = llm(prompt)[0]["generated_text"]
    output = generated.replace(prompt, "").strip()

    return output


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

    generated = llm(prompt)[0]["generated_text"]
    output = generated.replace(prompt, "").strip()

    return output


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
    machine learning, RAG pipelines, and data analysis.
    """

    output = run_pipeline(test_resume)
    print(output)
