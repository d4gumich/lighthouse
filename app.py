# =========================
# Imports
# =========================
import os
import torch
import pandas as pd
import re
import uuid
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, BitsAndBytesConfig
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import gradio as gr

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

# =========================
# Device
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)
print("****delete after successful deployment test")

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
####################################################################################
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ["QDRANT_API_KEY"]

COLLECTION_NAME = "career_rag"

client = QdrantClient(
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
###########################################################################################
#TOP_K = 3
TOP_K = int(os.getenv("TOP_K", 3))

# =========================
# Prepare directories
# =========================

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
# Build / Load Qdrant
# =========================
vector_size = embedder.get_sentence_embedding_dimension()
collections = client.get_collections().collections
if COLLECTION_NAME not in [c.name for c in collections]:
    print("Creating Qdrant collection...")

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config=VectorParams(
            size=vector_size,
            distance=Distance.COSINE,
        ),
    )

    job_texts = (
        df["title"].fillna("")
        + " "
        + df["description"].fillna("")
        + " "
        + df["skill_names"].fillna("")
    ).tolist()

    embeddings = embedder.encode(
        job_texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )

    points = []

    for idx, emb in enumerate(embeddings):

        payload = {
            "title": df.iloc[idx]["title"],
            "description": df.iloc[idx]["description"],
            "skills": df.iloc[idx]["skill_names"],
        }

        points.append(
            PointStruct(

                id=str(uuid.uuid4()),
                vector=emb.tolist(),
                payload=payload,
            )
        )

    client.upsert(
        collection_name=COLLECTION_NAME,
        points=points,
    )

    print("Uploaded", len(points), "jobs")

else:

    print("Existing collection found.")

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
            do_sample=False,          # FIX: consistent with temperature, was True before
           # temperature=0.3,         # FIX: now actually used
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
    resume_truncated = truncate_by_tokens(resume_text, max_tokens=2048)    #You are an expert resume parser. Extract all hard and soft skills explicitly or implicitly mentioned in the resume.
#Return ONLY a comma-separated list of skills. No explanations, no job titles, no repeated skills, no extra text.

    prompt = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>
                  Extract ONLY the skills explicitly mentioned in the resume.
                  Rules:
                  - Do NOT infer skills.
                  - Do NOT guess technologies.
                  - Do NOT repeat skills.
                  - If a skill is not written in the resume, do not include it.
                  Return ONLY a comma-separated list.
<|eot_id|><|start_header_id|>user<|end_header_id|>
Resume:
{resume_truncated}
<|eot_id|><|start_header_id|>assistant<|end_header_id|>
Skills:"""

    raw = generate_from_prompt(prompt, max_new_tokens=200)  # FIX: tight token budget for structured output
    print(raw)  #### CHECKING

    # Clean up — take only first line in case model adds extra text
    skills_line = raw.split("\n")[0].strip()

    # Deduplicate
    skills_list = list(dict.fromkeys([s.strip() for s in skills_line.split(",") if s.strip()]))
    return ", ".join(skills_list)

# =========================
# Stage 2: Retrieval
# =========================
# =========================
# Stage 2: Retrieval (Qdrant)
# =========================
def retrieve_jobs(skills_text, k=TOP_K):

    query_embedding = embedder.encode(
    [skills_text],
    convert_to_numpy=True,
    normalize_embeddings=True,
    )[0].tolist()

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_embedding,
        limit=k,
    ).points

    results = []

    for hit in hits:

        payload = hit.payload

        results.append({
            "title": payload["title"],
            "skills": payload["skills"],
            "description": payload["description"],
            "score": hit.score
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