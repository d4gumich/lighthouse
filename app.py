# =========================
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd
import gradio as gr
import re
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# HF Tokens (if needed)
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")

# =========================
# Configuration
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO  = "Data4GoodCenter/careermatch-llama3-8b-lora"

DATA_PATH  = "job_skill_results.csv"
TOP_K      = 3

# =========================
# Load job data
# =========================
assert os.path.exists(DATA_PATH), f"CSV not found: {DATA_PATH}"
df = pd.read_csv(DATA_PATH)

# =========================
# Build combined job text
# =========================
# Only include title + skills for better embeddings
df['combined_text'] = df['title'].fillna('') + " " + df['skill_names'].fillna('')

# Optional: filter only AI/Data jobs for better matches
df = df[df['combined_text'].str.contains(
    "data|ai|ml|engineer|scientist|nlp|machine learning",
    case=False,
    na=False
)]

# =========================
# Embedding model
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-small-en-v1.5",
    device=device  # Use GPU if available
)

# =========================
# Build FAISS index dynamically
# =========================
job_texts = df['combined_text'].tolist()
print(f"Encoding {len(job_texts)} job entries...")
embeddings = embedder.encode(job_texts, convert_to_numpy=True, batch_size=16, show_progress_bar=True)
faiss.normalize_L2(embeddings)

index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)
print(f"FAISS index built with {index.ntotal} entries, dim={embeddings.shape[1]}")

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
# Load LLaMA base + LoRA adapter
# =========================
print("Loading base model with 4-bit quantization + LoRA...")
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_auth_token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_config,
    torch_dtype=torch.float16,
    use_auth_token=HF_TOKEN
)

# Attach LoRA adapter
model = PeftModel.from_pretrained(model, LORA_REPO, torch_dtype=torch.float16)
model.eval()
print("Model loaded on:", next(model.parameters()).device)

# =========================
# LLM pipeline
# =========================
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_new_tokens=200,
    temperature=0.3,
    do_sample=False
)

# =========================
# Retrieval function
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
# LLM skill + recommendation generation
# =========================
def generate_output(resume_text, jobs):
    job_context = ""
    for job in jobs:
        job_context += f"{job['title']} (Skills: {job['skills']})\n"

    prompt = f"""
You are a career assistant.
1. Extract skills from the resume.
2. Match with jobs.
3. Recommend additional roles.
STRICT INSTRUCTIONS:
- You MUST follow the exact output format.
- Do NOT change labels.
- Do NOT add extra text.
Resume:
{resume_text[:2000]}
Jobs:
{job_context}
Output format:
Skills: <comma-separated>
Recommendations: <text>
"""

    response = llm(prompt)[0]["generated_text"]
    response = response.replace(prompt, "").strip()

    # Extract skills & recommendations
    skills = ""
    recommendations = response
    skills_match = re.search(r"Skills\s*[:\-]\s*(.*?)(?:\n|$)", response, re.IGNORECASE)
    rec_match = re.search(r"Recommendations\s*[:\-]\s*(.*)", response, re.IGNORECASE)

    if skills_match:
        skills = skills_match.group(1).strip()
    if rec_match:
        recommendations = rec_match.group(1).strip()

    return skills, recommendations

# =========================
# Full pipeline
# =========================
def run_pipeline(resume_text):
    # First extract skills
    skills, _ = generate_output(resume_text, [])
    # Then retrieve jobs using extracted skills
    jobs = retrieve_jobs(skills)
    skills, recommendations = generate_output(resume_text, jobs)
    return {
        "extracted_skills": skills,
        "top_jobs": jobs,
        "recommendations": recommendations,
    }

# =========================
# Gradio wrapper
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
        gr.Textbox(label="Recommendations"),
    ],
    title="Fast Resume Job Recommender",
    description="GPU-powered skill extraction + job recommendation"
)

demo.launch()