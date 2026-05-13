# =========================
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd
import re

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import gradio as gr

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# HF Token
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")

# =========================
# Configuration
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO = "Data4GoodCenter/careermatch-llama3-8b-lora"

DATA_PATH = "job_skill_results.csv"
FAISS_DIR = "data"
FAISS_PATH = os.path.join(FAISS_DIR, "faiss.index")
TOP_K = 3

# =========================
# Prepare directories
# =========================
os.makedirs(FAISS_DIR, exist_ok=True)
assert os.path.exists(DATA_PATH), f"CSV not found: {DATA_PATH}"

# =========================
# Embedding model
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
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
    print("Loading FAISS index from disk...")
    index = faiss.read_index(FAISS_PATH)
else:
    print("Building FAISS index...")
    embeddings = embedder.encode(job_texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)
    print("FAISS index saved to disk.")

# =========================
# 4-bit Quantization Config
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# =========================
# Load Base Model + LoRA
# =========================
print("Loading model with 4-bit quantization...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(base_model, LORA_REPO)
model.eval()
print("Model loaded on:", next(model.parameters()).device)

# =========================
# Helper: Token-safe truncation
# FIX: Use tokenizer-based truncation instead of character slicing
# =========================
def truncate_by_tokens(text, max_tokens=2048):
    tokens = tokenizer(
        text,
        max_length=max_tokens,
        truncation=True,
        return_tensors="pt"
    )
    return tokenizer.decode(tokens["input_ids"][0], skip_special_tokens=True)

# =========================
# Helper: Token-length-based prompt echo removal
# FIX: Slice by input token length instead of string replace()
# =========================
def generate_from_prompt(prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_length = inputs["input_ids"].shape[1]

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,          # FIX: consistent with temperature
            temperature=0.3,         # FIX: now actually used
            pad_token_id=tokenizer.eos_token_id,
            eos_token_id=tokenizer.eos_token_id,
            use_cache=True,
        )

    # FIX: Slice output by input token length — robust prompt echo removal
    response = tokenizer.decode(
        output_ids[0][input_length:],
        skip_special_tokens=True
    ).strip()
    return response

# =========================
# Stage 1: Skill Extraction
# FIX: Separate prompt for extraction only — not mixed with matching/recommendation
# =========================
def extract_skills(resume_text):
    resume_truncated = truncate_by_tokens(resume_text, max_tokens=2048)

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are an expert resume parser. Extract all hard and soft skills explicitly or implicitly mentioned in the resume.
Return ONLY a comma-separated list of skills. No explanations, no job titles, no repeated skills, no extra text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Resume:
{resume_truncated}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Skills:"""

    raw = generate_from_prompt(prompt, max_new_tokens=200)  # FIX: tight token budget for structured output

    # Clean up — take only first line in case model adds extra text
    skills_line = raw.split("\n")[0].strip()

    # Deduplicate
    skills_list = list(dict.fromkeys([s.strip() for s in skills_line.split(",") if s.strip()]))
    return ", ".join(skills_list)

# =========================
# Stage 2: Retrieval
# =========================
def retrieve_jobs(skills_text, k=TOP_K):
    query_emb = embedder.encode([skills_text], convert_to_numpy=True)
    faiss.normalize_L2(query_emb)
    scores, indices = index.search(query_emb, k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        results.append({
            "title": df.iloc[idx]["title"],
            "skills": df.iloc[idx]["skill_names"],
            "score": float(score),
        })
    return results

# =========================
# Stage 3: Recommendation Generation
# FIX: Separate prompt for recommendation only — focused single task
# =========================
def generate_recommendations(skills, jobs):
    job_context = ""
    for job in jobs:
        job_context += f"- {job['title']} (Skills: {job['skills']})\n"

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
You are a career advisor. Based on the candidate's skills and matching job roles, recommend 3 additional job titles they should consider.
Return ONLY a numbered list of job titles with a one-sentence explanation each. No extra text.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Candidate Skills: {skills}

Top Matching Jobs:
{job_context}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Recommendations:"""

    raw = generate_from_prompt(prompt, max_new_tokens=300)  # FIX: reasonable budget for recommendations

    # Take content after "Recommendations:" if echoed
    if "Recommendations:" in raw:
        raw = raw.split("Recommendations:")[-1].strip()

    return raw.strip()

# =========================
# Full Pipeline
# FIX: Three clean sequential stages instead of one overloaded prompt
# =========================
def run_pipeline(resume_text):
    print("Stage 1: Extracting skills...")
    skills = extract_skills(resume_text)
    print("Extracted Skills:", skills)

    print("Stage 2: Retrieving jobs...")
    jobs = retrieve_jobs(skills)
    print("Top Jobs:", jobs)

    print("Stage 3: Generating recommendations...")
    recommendations = generate_recommendations(skills, jobs)
    print("Recommendations:", recommendations)

    return {
        "extracted_skills": skills,
        "top_jobs": jobs,
        "recommendations": recommendations,
    }

# =========================
# Gradio UI
# =========================
def gradio_pipeline(resume_text):
    try:
        result = run_pipeline(resume_text)
        return (
            result["extracted_skills"],
            result["top_jobs"],
            result["recommendations"],
        )
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return str(e), [], str(e)

demo = gr.Interface(
    fn=gradio_pipeline,
    inputs=gr.Textbox(lines=10, placeholder="Paste resume text here..."),
    outputs=[
        gr.Textbox(label="Extracted Skills"),
        gr.JSON(label="Top Matched Jobs"),
        gr.Textbox(label="Recommended Roles"),
    ],
    title="CareerMatch — AI Job Recommender",
    description="Paste your resume to extract skills and get matched to BLS career paths."
)

demo.launch()