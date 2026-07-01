import os
import io
import json
import logging
from contextlib import asynccontextmanager
from typing import List, Optional
from pathlib import Path

import torch
import fitz  # type: ignore # PyMuPDF
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
import asyncio

from app.rag.retriever import retriever
from app.personas.personas import lawyer_take, optimist_take, pessimist_take

# ─────────────────────────────────────────────────────────────────────────────
# Setup
# ─────────────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Model Loading
# ─────────────────────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent.parent.parent / "cuad-main" / "train_models" / "inlegalbert-cuad"

tokenizer = None
model = None
device = None
model_load_error = None

def load_model():
    global tokenizer, model, device, model_load_error
    try:
        if not MODEL_PATH.exists():
            logger.warning(f"Fine-tuned model not found at {MODEL_PATH}. Falling back to base InLegalBERT.")
            model_name = "law-ai/InLegalBERT"
        else:
            model_name = str(MODEL_PATH)
            logger.info(f"Loading fine-tuned model from {MODEL_PATH}")

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        model = AutoModelForQuestionAnswering.from_pretrained(model_name)
        model.to(device)
        model.eval()
        logger.info("Model loaded successfully!")
    except Exception as e:
        model_load_error = str(e)
        logger.error(f"FAILED to load model: {e}")
        logger.error("The server will start but /analyze will return 503 until the model is available.")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model
    logger.info("Starting up — loading AI model...")
    load_model()
    yield
    # Shutdown (nothing to do)
    logger.info("Shutting down.")


app = FastAPI(
    title="AI Devil's Advocate API",
    description="Analyzes legal contracts and flags risky clauses using a fine-tuned InLegalBERT model.",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────────────────────
# CUAD Clause Definitions (The 15 most important risk categories)
# ─────────────────────────────────────────────────────────────────────────────
RISK_CLAUSES = [
    {
        "id": "termination",
        "label": "Termination for Convenience",
        "question": "Does the contract allow termination for convenience without cause?",
        "severity": "high",
        "icon": "⚠️",
        "description": "One party can exit without any reason, leaving you exposed."
    },
    {
        "id": "liability_cap",
        "label": "Liability Cap",
        "question": "What is the cap on liability or damages?",
        "severity": "high",
        "icon": "🔒",
        "description": "Limits the maximum amount you can recover if things go wrong."
    },
    {
        "id": "ip_ownership",
        "label": "IP Ownership",
        "question": "Who owns the intellectual property created under this contract?",
        "severity": "high",
        "icon": "💡",
        "description": "Determines who controls work products and inventions."
    },
    {
        "id": "non_compete",
        "label": "Non-Compete",
        "question": "Is there a non-compete clause restricting business activities?",
        "severity": "high",
        "icon": "🚫",
        "description": "Restricts your ability to work with competitors after the contract ends."
    },
    {
        "id": "auto_renewal",
        "label": "Auto-Renewal",
        "question": "Does the contract automatically renew?",
        "severity": "medium",
        "icon": "🔄",
        "description": "Could lock you in for another full term without you noticing."
    },
    {
        "id": "governing_law",
        "label": "Governing Law & Jurisdiction",
        "question": "What is the governing law and jurisdiction for disputes?",
        "severity": "medium",
        "icon": "⚖️",
        "description": "Determines which country's or state's laws apply and where you'd have to litigate."
    },
    {
        "id": "indemnification",
        "label": "Indemnification",
        "question": "What are the indemnification obligations?",
        "severity": "high",
        "icon": "🛡️",
        "description": "You may be responsible for covering the other party's legal costs."
    },
    {
        "id": "warranty",
        "label": "Warranty",
        "question": "What warranties or guarantees are provided?",
        "severity": "medium",
        "icon": "✅",
        "description": "Defines the quality and performance guarantees you are responsible for."
    },
    {
        "id": "exclusivity",
        "label": "Exclusivity",
        "question": "Is there an exclusivity clause preventing dealing with other parties?",
        "severity": "high",
        "icon": "🔗",
        "description": "Prevents you from working with others, limiting your options."
    },
    {
        "id": "price_adjustment",
        "label": "Price Adjustment",
        "question": "Are there any price adjustment or escalation clauses?",
        "severity": "medium",
        "icon": "💰",
        "description": "Prices could rise automatically without renegotiation."
    },
    {
        "id": "liquidated_damages",
        "label": "Liquidated Damages",
        "question": "Are there liquidated damages or penalty clauses?",
        "severity": "high",
        "icon": "💸",
        "description": "Pre-set financial penalties you must pay if you fail to meet obligations."
    },
    {
        "id": "force_majeure",
        "label": "Force Majeure",
        "question": "What events are covered under force majeure?",
        "severity": "low",
        "icon": "🌪️",
        "description": "Defines what unforeseen events excuse parties from their obligations."
    },
    {
        "id": "audit_rights",
        "label": "Audit Rights",
        "question": "Does the other party have the right to audit your records?",
        "severity": "medium",
        "icon": "🔍",
        "description": "The other party can inspect your books, which may be intrusive."
    },
    {
        "id": "assignment",
        "label": "Assignment",
        "question": "Can the contract be assigned to another party without consent?",
        "severity": "medium",
        "icon": "📋",
        "description": "Your contract could be sold or transferred to someone you didn't choose."
    },
    {
        "id": "confidentiality",
        "label": "Confidentiality",
        "question": "What are the confidentiality obligations and duration?",
        "severity": "medium",
        "icon": "🤐",
        "description": "Defines what information must be kept secret and for how long."
    },
]

# ─────────────────────────────────────────────────────────────────────────────
# Core QA Logic
# ─────────────────────────────────────────────────────────────────────────────
def answer_question(question: str, context: str, max_length: int = 512) -> Optional[str]:
    """
    Given a question and a contract context, uses the fine-tuned model
    to find and return the relevant clause text, or None if not found.
    Uses token-decode approach (compatible with use_fast=False tokenizers).
    """
    if not context.strip():
        return None

    best_answer = None
    best_score = -float("inf")

    # Sliding window over the context to handle long documents
    # Encode just the question to know how many tokens it uses
    question_tokens = tokenizer.encode(question, add_special_tokens=False)
    # Reserve: [CLS] + question + [SEP] + context + [SEP] = len(q)+3 overhead
    context_max = max_length - len(question_tokens) - 3
    stride = 128

    # Tokenize context only (to split into chunks)
    context_encoding = tokenizer.encode(context, add_special_tokens=False)

    # Process in overlapping windows
    start = 0
    while start < max(1, len(context_encoding)):
        chunk_ids = context_encoding[start: start + context_max]
        chunk_text = tokenizer.decode(chunk_ids, skip_special_tokens=True)

        # Tokenize question + chunk together
        try:
            inputs = tokenizer(
                question,
                chunk_text,
                max_length=max_length,
                truncation="only_second",
                padding="max_length",
                return_tensors="pt",
            )
        except Exception:
            break

        input_ids = inputs["input_ids"][0]
        inputs_device = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs_device)

        start_logits = outputs.start_logits[0]
        end_logits = outputs.end_logits[0]

        # Find the best valid (start, end) pair within context tokens
        # Token 0 is [CLS]; we skip it for answer extraction
        start_idx = torch.argmax(start_logits).item()
        end_idx = torch.argmax(end_logits).item()
        score = (start_logits[start_idx] + end_logits[end_idx]).item()

        # Valid answer: end >= start, not [CLS] token (idx 0)
        if end_idx >= start_idx and start_idx > 0:
            # Decode the predicted token span back to text
            answer_ids = input_ids[start_idx: end_idx + 1].tolist()
            answer_text = tokenizer.decode(answer_ids, skip_special_tokens=True).strip()

            if answer_text and score > best_score and len(answer_text) > 5:
                best_score = score
                best_answer = answer_text

        # Move window forward
        if start + context_max >= len(context_encoding):
            break
        start += context_max - stride

    # Only return high-confidence answers
    if best_answer and len(best_answer) > 10 and best_score > 0:
        return best_answer

    return None



# ─────────────────────────────────────────────────────────────────────────────
# Text Extraction
# ─────────────────────────────────────────────────────────────────────────────
def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extract all text from a PDF file using PyMuPDF."""
    try:
        # Wrap in BytesIO — fitz works most reliably with a stream object
        stream = io.BytesIO(pdf_bytes)
        doc = fitz.open(stream=stream, filetype="pdf")
        full_text = ""
        for page in doc:
            full_text += page.get_text()
        doc.close()
        return full_text
    except Exception as e:
        raise ValueError(f"PyMuPDF could not open the file: {e}")


# ─────────────────────────────────────────────────────────────────────────────
# Pydantic Schemas
# ─────────────────────────────────────────────────────────────────────────────
class RiskFinding(BaseModel):
    id: str
    label: str
    question: str
    severity: str
    icon: str
    description: str
    found: bool
    answer: Optional[str] = None
    precedents: Optional[List[dict]] = []
    lawyer_take: Optional[dict] = None
    optimist_take: Optional[dict] = None
    pessimist_take: Optional[dict] = None

class AnalysisResult(BaseModel):
    filename: str
    word_count: int
    findings: List[RiskFinding]
    risk_score: int
    high_risks: int
    medium_risks: int
    low_risks: int


# ─────────────────────────────────────────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────────────────────────────────────────
@app.get("/health")
def health_check():
    return {
        "status": "ok" if model is not None else "model_not_loaded",
        "message": "AI Devil's Advocate API is running",
        "model_loaded": model is not None,
        "model_error": model_load_error,
        "device": str(device) if device else "unknown"
    }


@app.post("/analyze", response_model=AnalysisResult)
async def analyze_contract(file: UploadFile = File(...)):
    """
    Upload a PDF contract and receive a full risk analysis.
    The AI will scan for 15 categories of high-risk legal clauses.
    """
    # Guard: model not loaded
    if model is None:
        raise HTTPException(
            status_code=503,
            detail=f"AI model is not loaded yet. Error: {model_load_error or 'still loading, please retry in a moment.'}"
        )

    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    contents = await file.read()

    # Extract text
    try:
        contract_text = extract_text_from_pdf(contents)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract text from PDF: {str(e)}")

    if not contract_text.strip():
        raise HTTPException(status_code=400, detail="Could not extract any text from the PDF. Is it a scanned image?")

    word_count = len(contract_text.split())
    logger.info(f"Analyzing contract: {file.filename} ({word_count} words)")

    # Run the AI on each risk clause
    findings = []
    high_count = 0
    medium_count = 0
    low_count = 0

    for clause in RISK_CLAUSES:
        answer = answer_question(clause["question"], contract_text)
        found = answer is not None
        
        precedents = []
        lawyer = None
        optimist = None
        pessimist = None

        if found:
            if clause["severity"] == "high":
                high_count += 1
            elif clause["severity"] == "medium":
                medium_count += 1
            else:
                low_count += 1
                
            # 1. Retrieve Precedents
            precedents = retriever.retrieve_similar(answer, k=2)
            
            # 2. Get Persona Takes (Run sequentially to avoid overloading the GPU and causing timeouts)
            lawyer = await asyncio.to_thread(lawyer_take, answer, clause["severity"], precedents)
            optimist = await asyncio.to_thread(optimist_take, answer, clause["severity"], precedents)
            pessimist = await asyncio.to_thread(pessimist_take, answer, clause["severity"], precedents)

        findings.append(RiskFinding(
            id=clause["id"],
            label=clause["label"],
            question=clause["question"],
            severity=clause["severity"],
            icon=clause["icon"],
            description=clause["description"],
            found=found,
            answer=answer,
            precedents=precedents,
            lawyer_take=lawyer,
            optimist_take=optimist,
            pessimist_take=pessimist
        ))

    # Calculate an overall risk score (0 to 100)
    risk_score = min(100, (high_count * 15) + (medium_count * 7) + (low_count * 3))

    logger.info(f"Analysis complete. High: {high_count}, Medium: {medium_count}, Low: {low_count}, Score: {risk_score}")

    return AnalysisResult(
        filename=file.filename,
        word_count=word_count,
        findings=findings,
        risk_score=risk_score,
        high_risks=high_count,
        medium_risks=medium_count,
        low_risks=low_count
    )
