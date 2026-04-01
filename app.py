# =========================
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd
import gradio as gr

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer

# =========================
# Fix threads (IMPORTANT)
# =========================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# Config
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")

# Use PRE-QUANTIZED model (CRITICAL FIX)
BASE_MODEL = "unsloth/llama-3-8b-bnb-4bit"

# Your LoRA (kept same)
LORA_REPO = "Data4GoodCenter/careermatch-llama3-8b-lora"

DATA_PATH = "job_skill_results.csv"
FAISS_DIR = "data"
FAISS_PATH = os.path.join(FAISS_DIR, "faiss.index")
TOP_K = 3

os.makedirs(FAISS_DIR, exist_ok=True)

# =========================
# Embedding model (small)
# =========================
embedder = SentenceTransformer(
    "sentence-transformers/all-MiniLM-L6-v2",
    device="cpu"
)

# =========================
# Load data
# =========================
df = pd.read_csv(DATA_PATH)

job_texts = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["skill_names"].fillna("")
).tolist()

# =========================
# FAISS (SAFE BUILD)
# =========================
if os.path.exists(FAISS_PATH):
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)
else:
    print("Building FAISS index (first run only)...")

    # batch encode to avoid RAM spike
    batch_size = 32
    all_embeddings = []

    for i in range(0, len(job_texts), batch_size):
        batch = job_texts[i:i+batch_size]
        emb = embedder.encode(batch, convert_to_numpy=True)
        all_embeddings.append(emb)

    embeddings = torch.cat([torch.tensor(x) for x in all_embeddings]).numpy()

    faiss.normalize_L2(embeddings)

    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    faiss.write_index(index, FAISS_PATH)
    print("FAISS index saved!")

# =========================
# Load Model (QUANTIZED)
# =========================
print("Loading quantized model...")

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, token=HF_TOKEN)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True
)

# Attach LoRA
model = PeftModel.from_pretrained(model, LORA_REPO)

model.config.use_cache = False
model.eval()

print("Model loaded!")

# =========================
# Generation (NO pipeline)
# =========================
def generate_text(prompt):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=False
        )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# =========================
# Retrieval
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
# LLM Output
# =========================
def generate_output(resume_text, jobs):

    job_context = "\n".join(
        [f"{j['title']} (Skills: {j['skills']})" for j in jobs]
    )

    prompt = f"""
You are a career assistant.

Extract skills and recommend jobs.

STRICT FORMAT:
Skills: <comma-separated>
Recommendations: <text>

Resume:
{resume_text[:1500]}

Jobs:
{job_context}
"""

    response = generate_text(prompt)
    response = response.replace(prompt, "").strip()

    import re

    skills = ""
    recommendations = response

    skills_match = re.search(r"Skills\s*[:\-]\s*(.*?)(?:\n|$)", response, re.I)
    rec_match = re.search(r"Recommendations\s*[:\-]\s*(.*)", response, re.I)

    if skills_match:
        skills = skills_match.group(1).strip()

    if rec_match:
        recommendations = rec_match.group(1).strip()

    return skills, recommendations

# =========================
# Pipeline
# =========================
def run_pipeline(resume_text):
    jobs = retrieve_jobs(resume_text)
    skills, recommendations = generate_output(resume_text, jobs)
    return skills, jobs, recommendations

# =========================
# Gradio UI
# =========================
def gradio_pipeline(resume_text):
    try:
        return run_pipeline(resume_text)
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
    title="Resume Job Recommender",
)

demo.launch()