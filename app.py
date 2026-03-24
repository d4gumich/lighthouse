# =========================
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import gradio as gr

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

# =========================
# HF Token (IMPORTANT)
# =========================
HF_TOKEN = os.environ.get("HF_TOKEN")  # set in HF Spaces secrets

# =========================
# Configuration
# =========================
BASE_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
LORA_REPO  = "Data4GoodCenter/careermatch-llama3-8b-lora"

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
# Embedding model (FAST)
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
    print("Loading FAISS index...")
    index = faiss.read_index(FAISS_PATH)
else:
    print("Building FAISS index...")
    embeddings = embedder.encode(job_texts, convert_to_numpy=True)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    faiss.write_index(index, FAISS_PATH)

# =========================
# Load Model + LoRA (STABLE)
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    torch_dtype=torch.float16,
    device_map="auto",
    token=HF_TOKEN
)

model = PeftModel.from_pretrained(model, LORA_REPO)

model.eval()

print("Model loaded on:", next(model.parameters()).device)

# =========================
# Pipeline
# =========================
llm = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if device == "cuda" else -1,
    max_new_tokens=200,
    temperature=0.3,
    do_sample=False,
)

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
# Single-pass LLM
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

Resume:
{resume_text[:2000]}

Jobs:
{job_context}

Output format:
Skills: <comma-separated>
Recommendations: <text>
"""

    response = llm(prompt)[0]["generated_text"]

    skills = ""
    recommendations = response

    if "Skills:" in response:
        skills = response.split("Skills:")[-1].split("Recommendations:")[0].strip()

    if "Recommendations:" in response:
        recommendations = response.split("Recommendations:")[-1].strip()

    return skills, recommendations

# =========================
# Pipeline
# =========================
def run_pipeline(resume_text):
    jobs = retrieve_jobs(resume_text)
    skills, recommendations = generate_output(resume_text, jobs)

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
    inputs=gr.Textbox(lines=10, placeholder="Paste resume text here"),
    outputs=[
        gr.Textbox(label="Extracted Skills"),
        gr.JSON(label="Top Jobs"),
        gr.Textbox(label="Recommendations"),
    ],
    title="⚡ Fast Resume Job Recommender",
    description="GPU-powered skill extraction + job recommendation"
)

demo.launch()