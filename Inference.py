

import os
import json
import time
from pathlib import Path
from typing import List, Dict, Any

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel



BASE_MODEL  = ""
ADAPTER_DIR = ""
EXTRACT_DIR = ""
OUTPUT_DIR  = ""

# Inference settings
MAX_INPUT_TOKENS        = 2048
MAX_NEW_TOKENS_SUMMARY  = 200
MAX_NEW_TOKENS_JSON     = 300
MAX_NEW_TOKENS_QNA      = 150
BATCH_SIZE              = 4  

LANG_HINTS = {
    "English": "English", "Hindi": "Hindi", "Gujarati": "Gujarati",
    "Bangla": "Bangla", "Assamese": "Assamese", "Kannada": "Kannada",
    "Marathi": "Marathi", "Tamil": "Tamil", "Telugu": "Telugu"
}



def tprint(msg: str):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def safe_read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def clip_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens:
        return text
    ids = ids[-max_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def chunk(lst, n):
    for i in range(0, len(lst), n):
        yield lst[i:i+n]

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")



JSON_TEMPLATE = {
  "patient_identifiers": None,
  "demographics": {"age": None, "sex": None},
  "visit": {"date_time": None, "type": None},
  "chief_complaint": None,
  "onset_duration": None,
  "symptom_description": None,
  "aggravating_factors": None,
  "relieving_factors": None,
  "associated_symptoms": [],
  "past_medical_history": None,
  "past_surgical_history": None,
  "family_history": None,
  "current_medications": [],
  "allergies": None,
  "social_history": [],
  "functional_status": None,
  "vital_signs": None,
  "examination_findings": None,
  "investigations": [],
  "assessment_primary_diagnosis": None,
  "differential_diagnoses": [],
  "management_plan": None,
  "tests_referrals_planned": [],
  "follow_up_plan": None,
  "chronology_response_to_treatment": None,
  "patient_concerns_preferences_consent": None,
  "safety_issues_red_flags": None,
  "coding_terms": None,
  "conversation_metadata": {"timestamps": [], "speaker_labels": []}
}

def build_summary_text_prompt(dialogue_text: str) -> str:
    return (
        "Summarize the following doctor–patient dialogue in English.\n\n"
        f"Dialogue:\n{dialogue_text}\n\nSummary:"
    )

def build_summary_json_prompt(dialogue_text: str) -> str:
    schema = json.dumps(JSON_TEMPLATE, indent=2)
    return (
        "From the following doctor–patient dialogue, extract structured clinical information.\n"
        "Return ONLY valid JSON matching this schema:\n"
        f"{schema}\n\nDialogue:\n{dialogue_text}\n\nJSON:"
    )

def build_qna_prompt(question: str, lang_hint: str) -> str:
    return (
        f"Answer the following patient question in {lang_hint}. "
        f"Be concise and factual.\n\nQuestion: {question}\nAnswer:"
    )

def try_extract_json(text: str):
    text = text.strip()
    if "```" in text:
        for block in text.split("```"):
            block = block.strip()
            if block.lower().startswith("json"):
                block = block[4:].strip()
            try:
                return json.loads(block)
            except Exception:
                pass
    try:
        return json.loads(text)
    except Exception:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end != -1 and end > start:
            try:
                return json.loads(text[start:end+1])
            except Exception:
                pass
    return None



def gen_batch_text(model, tokenizer, prompts, max_new_tokens):
    inputs = tokenizer(
        prompts, padding=True, truncation=True,
        max_length=MAX_INPUT_TOKENS, return_tensors="pt"
    ).to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    return tokenizer.batch_decode(out, skip_special_tokens=True)

def gen_batch_json(model, tokenizer, prompts, max_new_tokens):
    texts = gen_batch_text(model, tokenizer, prompts, max_new_tokens)
    objs = []
    for txt in texts:
        obj = try_extract_json(txt)
        if obj is None:
            txt2 = gen_batch_text(
                model, tokenizer,
                [txt + "\n\nReturn ONLY valid JSON."],
                max_new_tokens
            )[0]
            obj = try_extract_json(txt2)
        objs.append(obj or JSON_TEMPLATE)
    return objs

# ============================================================
# MODEL LOADING
# ============================================================

def find_latest_adapter(dir_path):
    """Find the newest adapter_step_* directory."""
    dirs = [d for d in Path(dir_path).iterdir()
            if d.is_dir() and d.name.startswith("adapter_step_")]
    if not dirs:
        return dir_path  # fallback to base adapter dir
    latest = sorted(dirs, key=lambda x: int(x.name.split("_")[-1]))[-1]
    tprint(f"Using latest adapter: {latest}")
    return latest


def run_inference():

    bnb_cfg = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    tprint("Loading tokenizer...")
    adapter_path = find_latest_adapter(ADAPTER_DIR)
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=False)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16,
    )

    model = PeftModel.from_pretrained(base, adapter_path)
    model.eval()

    langs = [p for p in Path(EXTRACT_DIR).iterdir() if p.is_dir()]
    tprint(f"Found {len(langs)} languages: {[p.name for p in langs]}")

    for lang_dir in langs:
        lang = lang_dir.name
        lang_hint = LANG_HINTS.get(lang, "the same language")
        out_lang = Path(OUTPUT_DIR) / lang
        dlg_dir = lang_dir / "Dialogues"
        qna_dir = lang_dir / "QnA"

        tprint(f"Language: {lang}")

        # ---- Summaries ----
        if dlg_dir.exists():
            files = sorted(dlg_dir.glob("*.jsonl"))
            tprint(f"  • Dialogues: {len(files)}")
            items = []
            for f in files:
                rows = safe_read_jsonl(f)
                dialogue = " ".join(
                    str(r.get("dialogue", "")) if isinstance(r, dict) else str(r)
                    for r in rows
                )
                dialogue = clip_tokens(tokenizer, dialogue, MAX_INPUT_TOKENS)
                items.append((f, dialogue))

            # TEXT
            for batch in chunk(items, BATCH_SIZE):
                prompts = [build_summary_text_prompt(d) for _, d in batch]
                outputs = gen_batch_text(model, tokenizer, prompts, MAX_NEW_TOKENS_SUMMARY)
                for (f, _), txt in zip(batch, outputs):
                    write_text(out_lang / "Summary_Text" / f"{f.stem}_summary.txt", txt)

            # JSON
            for batch in chunk(items, BATCH_SIZE):
                prompts = [build_summary_json_prompt(d) for _, d in batch]
                outputs = gen_batch_json(model, tokenizer, prompts, MAX_NEW_TOKENS_JSON)
                for (f, _), js in zip(batch, outputs):
                    write_json(out_lang / "Summary_Json" / f"{f.stem}_summary.json", js)

        # ---- QnA ----
        if qna_dir.exists():
            qfiles = sorted(qna_dir.glob("*.json"))
            tprint(f" QnA files: {len(qfiles)}")
            for qf in qfiles:
                try:
                    data = json.load(open(qf, encoding="utf-8"))
                except Exception:
                    tprint(f"    - Skipped malformed {qf.name}")
                    continue
                qs = [q.get("question", "").strip()
                      for q in data.get("questions", []) if isinstance(q, dict)]
                qs = [q for q in qs if q]

                answers = []
                for batch in chunk(qs, BATCH_SIZE):
                    prompts = [build_qna_prompt(q, lang_hint) for q in batch]
                    outs = gen_batch_text(model, tokenizer, prompts, MAX_NEW_TOKENS_QNA)
                    outs = [o.replace("Answer:", "").strip() for o in outs]
                    answers.extend(outs)

                out_payload = {"questions": [
                    {"question": q, "answer": a} for q, a in zip(qs, answers)
                ]}
                write_json(out_lang / "QnA" / f"{qf.stem}_answers.json", out_payload)

        torch.cuda.empty_cache()

    tprint(f"Inference complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    run_inference()
