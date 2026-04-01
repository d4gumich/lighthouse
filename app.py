# ========================= 
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd
import gradio as gr
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

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
# Config
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO  = "Data4GoodCenter/careermatch-llama3-8b-lora"

DATA_PATH  = "job_skill_results.csv"
TOP_K      = 3

# =========================
# Load data
# =========================
assert os.path.exists(DATA_PATH), f"CSV not found: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)

df['combined_text'] = df['title'].fillna('') + " " + df['skill_names'].fillna('')

df = df[df['combined_text'].str.contains(
    "data|ai|ml|engineer|scientist|nlp|machine learning",
    case=False,
    na=False
)]

# =========================
# Embeddings (UNCHANGED)
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=device
)

# =========================
# FAISS (UNCHANGED)
# =========================
job_texts = df['combined_text'].tolist()
print(f"Encoding {len(job_texts)} job entries...")

embeddings = embedder.encode(
    job_texts,
    convert_to_numpy=True,
    batch_size=16,
    show_progress_bar=True
)

faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

print(f"FAISS index built with {index.ntotal} entries")

# =========================
# Quantization config
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"
)

# =========================
# Load Model + LoRA (FIXED)
# =========================
print("Loading LLaMA + LoRA...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN
)

# 🔴 important for llama
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.float16,
    token=HF_TOKEN
)

# Attach LoRA safely
model = PeftModel.from_pretrained(
    model,
    LORA_REPO,
    torch_dtype=torch.float16
)

model.eval()

# =========================
# 🔴 REPLACE pipeline with manual generate (more stable)
# =========================
def generate_text(prompt):
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        padding=True,
        truncation=True
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Retrieval (UNCHANGED)
# =========================
def retrieve_jobs(text, k=TOP_K):
    query_emb = embedder.encode([text], convert_to_numpy=True)
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
# LLM generation
# =========================
def generate_output(resume_text, jobs):
    job_context = ""
    for job in jobs:
        job_context += f"{job['title']} (Skills: {job['skills']})\n"

    prompt = f"""
You are a career assistant.

STRICT FORMAT:
Skills: <comma-separated>
Recommendations: <top 2 careers>

Resume:
{resume_text[:2000]}

Jobs:
{job_context}
"""

    response = generate_text(prompt)
    response = response.replace(prompt, "").strip()

    # Extract
    skills = ""
    recommendations = ""

    skills_match = re.search(r"Skills\s*[:\-]\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
    rec_match = re.search(r"Recommendations\s*[:\-]\s*(.*)", response, re.IGNORECASE)

    if skills_match:
        skills = skills_match.group(1).strip()

    if rec_match:
        recommendations = rec_match.group(1).strip()

    return skills, recommendations

# =========================
# Pipeline
# =========================
def run_pipeline(resume_text):
    skills, _ = generate_output(resume_text, [])
    jobs = retrieve_jobs(skills)
    skills, recommendations = generate_output(resume_text, jobs)

    return {
        "extracted_skills": skills,
        "top_jobs": jobs,
        "recommendations": recommendations,
    }

# =========================
# Gradio
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
    inputs=gr.Textbox(lines=10, placeholder="Paste resume text here"),
    outputs=[
        gr.Textbox(label="Extracted Skills"),
        gr.JSON(label="Top Jobs"),
        gr.Textbox(label="Top 2 Career Recommendations"),
    ],
    title="Fast Resume Job Recommender",
    description="Skill extraction + FAISS + LLaMA3 LoRA"
)

demo.launch()