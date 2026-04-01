# =========================
# Imports
# =========================
import os
import torch
import faiss
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline, BitsAndBytesConfig
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
# Embedding model
# =========================
embedder = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device="cpu"  # embeddings can stay on CPU
)

# =========================
# Load job data
# =========================
df = pd.read_csv(DATA_PATH)
job_texts = (
    df["title"].fillna("") + " " +
    df["description"].fillna("") + " " +
    df["skill_names"].fillna("")
).tolist()

# =========================
# Build / Load FAISS index
# =========================
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
# Quantization config for 4-bit
# =========================
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

# =========================
# Load Model + LoRA
# =========================
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(
    BASE_MODEL,
    use_auth_token=HF_TOKEN
)
tokenizer.pad_token = tokenizer.eos_token

# Load quantized base model
model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    use_auth_token=HF_TOKEN
)

# Load LoRA adapter
model = PeftModel.from_pretrained(model, LORA_REPO)
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
# Skills extraction & recommendation
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

    # Parse response reliably
    skills = ""
    recommendations = ""
    if "Skills:" in response and "Recommendations:" in response:
        skills = response.split("Skills:")[-1].split("Recommendations:")[0].strip()
        recommendations = response.split("Recommendations:")[-1].strip()
    else:
        recommendations = response.strip()

    return skills, recommendations

# =========================
# Full pipeline
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
# Gradio interface
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