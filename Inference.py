import os, json, time, re
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from langdetect import detect, DetectorFactory

DetectorFactory.seed = 42
torch.set_grad_enabled(False)


#SKIP_LANGS = ['Gujarati', 'Dogri', 'Hindi', 'Marathi', 'Kannada', 'English', 'Assamese', 'Tamil', 'Bangla']

# -----------------------
# PATHS
# -----------------------
BASE_MODEL   = ""  
ADAPTER_DIR  = ""
TEST_DIR     = ""
OUTPUT_DIR   = ""

# ============================================================
# RUNTIME SETTINGS
# ============================================================
MAX_INPUT_TOKENS       = 32000         
MAX_NEW_TOKENS_SUMMARY = 512
MAX_NEW_TOKENS_JSON    = 120
MAX_NEW_TOKENS_QNA     = 200
BATCH_SIZE             = 2

LANG_HINTS = {
    "English": "English", "Hindi": "Hindi", "Gujarati": "Gujarati",
    "Bangla": "Bangla", "Assamese": "Assamese", "Kannada": "Kannada",
    "Marathi": "Marathi", "Tamil": "Tamil", "Telugu": "Telugu", "Dogri": "Dogri"
}

# ============================================================
# PROMPTS
# ============================================================
SYSTEM_SUMMARY = (
    "You are a clinical summarization assistant. Write a fluent English summary "
    "focusing on diagnosis, symptoms, investigations, and management plan. "
    "Write 6–10 sentences. End your summary with the token <<END>>."
)

SYSTEM_JSON = (
    "You are a concise clinical information extraction assistant. "
    "Answer in English only. If the information is not present, answer exactly 'N/A'. "
    "Do not add explanations. Keep answers ≤ requested length."
)

SYSTEM_QNA = (
    "You are a multilingual clinical assistant. Answer in the SAME LANGUAGE as the user's question. "
    "Be concise, factual, and helpful."
)

JSON_FIELDS: Dict[str, tuple] = {
    "patient_identifiers": ("Identify the patient by name or ID if mentioned.", "str", 20, False),
    "demographics.age": ("What is the patient's age?", "str", 15, False),
    "demographics.sex": ("What is the patient's sex?", "str", 10, False),
    "visit.date_time": ("When did the visit occur? (date/time if mentioned)", "str", 20, False),
    "visit.type": ("What type of visit was it? (in-person/telemedicine etc.)", "str", 16, False),
    "chief_complaint": ("What is the chief complaint?", "str", 20, False),
    "onset_duration": ("How long have symptoms been present?", "str", 20, False),
    "symptom_description": ("Describe main symptoms briefly.", "str", 20, False),
    "aggravating_factors": ("What aggravates symptoms?", "str", 20, False),
    "relieving_factors": ("What relieves symptoms?", "str", 20, False),
    "associated_symptoms": ("List associated symptoms; semicolon-separated.", "list", 40, True),
    "past_medical_history": ("Summarize past medical history.", "str", 20, False),
    "past_surgical_history": ("Summarize surgical history.", "str", 16, False),
    "family_history": ("Summarize family history.", "str", 16, False),
    "current_medications": ("List current medications; semicolon-separated.", "list", 24, True),
    "allergies": ("List allergies (or N/A).", "str", 10, False),
    "social_history": ("Summarize social history (tobacco/alcohol).", "str", 18, False),
    "functional_status": ("Summarize functional status.", "str", 16, False),
    "vital_signs": ("List vital signs briefly.", "str", 18, False),
    "examination_findings": ("Summarize examination findings.", "str", 18, False),
    "investigations": ("List investigations performed/planned; semicolon-separated.", "list", 30, True),
    "assessment_primary_diagnosis": ("What is the primary diagnosis?", "str", 14, False),
    "differential_diagnoses": ("List 2–4 differentials; semicolon-separated.", "list", 16, True),
    "management_plan": ("Summarize management plan briefly.", "str", 22, False),
    "tests_referrals_planned": ("List tests/referrals planned; semicolon-separated.", "list", 30, True),
    "follow_up_plan": ("Summarize follow-up plan.", "str", 16, False),
    "chronology_response_to_treatment": ("Describe response/progress.", "str", 18, False),
    "patient_concerns_preferences_consent": ("Summarize patient concerns/consent.", "str", 18, False),
    "safety_issues_red_flags": ("List red flags briefly.", "str", 18, False),
    "coding_terms": ("Give likely coding terms or N/A.", "str", 16, False),
}

# ============================================================
# UTILS
# ============================================================
def tprint(msg): print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)

def safe_read_jsonl(path: Path):
    rows = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            s = line.strip()
            if s:
                try: rows.append(json.loads(s))
                except: continue
    return rows

def write_json(path: Path, obj: Any):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

def write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(text.strip() + "\n")

def detect_lang(text: str) -> str:
    try: return detect(text)
    except Exception: return "unknown"

def clip_tokens(tokenizer, text, max_tokens):
    ids = tokenizer.encode(text, add_special_tokens=False)
    if len(ids) <= max_tokens: return text
    ids = ids[-max_tokens:]
    return tokenizer.decode(ids, skip_special_tokens=True)

def shorten(text: str, max_words: int, max_chars: int = 180) -> str:
    text = re.sub(r"\s+", " ", text.strip())
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words])
    if len(text) > max_chars:
        text = text[:max_chars].rstrip(" .,;:-")
    return text

def parse_semicolon_list(s: str, max_items: int = 6):
    if s.strip().upper() == "N/A": return []
    parts = [p.strip(" ,;.-") for p in s.split(";")]
    return [p for p in parts if p][:max_items]

# ============================================================
# CHAT + GENERATION
# ============================================================
def build_messages(system_prompt, user_prompt):
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

def chat_generate(model, tokenizer, messages, max_new_tokens):
    text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=None,
            top_p=None,
            num_beams=1,
            pad_token_id=tokenizer.eos_token_id
        )
    gen = outputs[0][len(inputs.input_ids[0]):]
    return tokenizer.decode(gen, skip_special_tokens=True).strip()

# ============================================================
# MODEL LOADING
# ============================================================
def find_latest_adapter(path):
    p = Path(path)
    cands = [d for d in p.iterdir() if d.is_dir() and d.name.startswith("adapter_step_")]
    return sorted(cands, key=lambda x: int(x.name.split("_")[-1]))[-1] if cands else path

def load_model():
    tprint("Loading tokenizer...")
    tok = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16)
    tprint("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(BASE_MODEL, device_map="auto",
                                                quantization_config=bnb, torch_dtype=torch.float16)
    adapter = find_latest_adapter(ADAPTER_DIR)
    tprint(f"Using adapter: {adapter}")
    model = PeftModel.from_pretrained(base, adapter)
    model.eval()
    return model, tok

# ============================================================
# MAIN PIPELINE
# ============================================================
def main():
    model, tokenizer = load_model()
    langs = [p for p in Path(TEST_DIR).iterdir() if p.is_dir()]
    tprint(f"Found {len(langs)} languages: {[p.name for p in langs]}")

    for lang_dir in langs:
        lang = lang_dir.name
        if lang in SKIP_LANGS:
            tprint(f"Skipping {lang} (already completed or running elsewhere)")
            continue

        lang_hint = LANG_HINTS.get(lang, "the same language")
        out_lang = Path(OUTPUT_DIR) / lang
        dlg_dir = lang_dir / "Dialogues"
        qna_dir = lang_dir / "QnA"

        tprint(f"Language: {lang}")

        # ---- SUMMARIES + JSON ----
        if dlg_dir.exists():
            files = sorted(dlg_dir.glob("*.jsonl"))
            tprint(f"  • Dialogues: {len(files)}")
            for f in tqdm(files, desc=f"{lang} dialogues"):
                out_sum_txt  = out_lang / "Summary_Text" / f"{f.stem}_summary.txt"
                out_sum_json = out_lang / "Summary_Json" / f"{f.stem}_summary.json"
                if out_sum_txt.exists() and out_sum_json.exists(): continue
                rows = safe_read_jsonl(f)
                dialogue = " ".join(str(r.get("dialogue", "")) if isinstance(r, dict) else str(r) for r in rows)
                dialogue_clip = clip_tokens(tokenizer, dialogue, MAX_INPUT_TOKENS)
                messages = build_messages(SYSTEM_SUMMARY, f"Dialogue:\n{dialogue_clip}\n\nWrite the summary and end with <<END>>.")
                summary_raw = chat_generate(model, tokenizer, messages, MAX_NEW_TOKENS_SUMMARY)
                end_pos = summary_raw.find("<<END>>")
                summary = summary_raw[:end_pos].strip() if end_pos != -1 else summary_raw.strip()
                if detect_lang(summary) != "en":
                    messages = build_messages(SYSTEM_SUMMARY, f"Dialogue:\n{dialogue_clip}\n\nWrite ONLY in English. End with <<END>>.")
                    summary_raw = chat_generate(model, tokenizer, messages, MAX_NEW_TOKENS_SUMMARY)
                    end_pos = summary_raw.find("<<END>>")
                    summary = summary_raw[:end_pos].strip() if end_pos != -1 else summary_raw.strip()
                write_text(out_sum_txt, summary)

                flat = {}
                for field, (q, typ, max_words, is_list) in JSON_FIELDS.items():
                    user_q = (
                        f"Summary:\n{summary}\n\nDialogue:\n{dialogue_clip}\n\nQuestion:\n{q}\n\n"
                        f"Answer in English, ≤{max_words} words. "
                        f"{'Return a semicolon-separated list.' if is_list else 'Return short phrase.'} "
                        f"If unknown, answer exactly N/A."
                    )
                    msg = build_messages(SYSTEM_JSON, user_q)
                    ans = chat_generate(model, tokenizer, msg, MAX_NEW_TOKENS_JSON)
                    if detect_lang(ans) != "en":
                        msg = build_messages(SYSTEM_JSON, user_q + "\n\nAnswer strictly in English.")
                        ans = chat_generate(model, tokenizer, msg, MAX_NEW_TOKENS_JSON)
                    ans = shorten(ans, max_words=max_words, max_chars=180)
                    flat[field] = parse_semicolon_list(ans) if is_list else ans or "N/A"

                nested = {}
                for k, v in flat.items():
                    parts = k.split(".")
                    cur = nested
                    for p in parts[:-1]:
                        cur = cur.setdefault(p, {})
                    cur[parts[-1]] = v
                write_json(out_sum_json, nested)

        # ---- QnA ----
        if qna_dir.exists():
            qfiles = sorted(qna_dir.glob("*.json"))
            tprint(f"  • QnA files: {len(qfiles)}")
            for qf in tqdm(qfiles, desc=f"{lang} QnA"):
                try: data = json.load(open(qf, encoding="utf-8"))
                except Exception:
                    tprint(f"    - Skipped malformed {qf.name}")
                    continue
                qs = [q.get("question","").strip() for q in data.get("questions",[]) if q.get("question","").strip()]
                if not qs: continue
                answers = []
                for q in qs:
                    msg = build_messages(SYSTEM_QNA, f"Question ({lang_hint}): {q}")
                    ans = chat_generate(model, tokenizer, msg, MAX_NEW_TOKENS_QNA)
                    det = detect_lang(ans)
                    if lang_hint.lower() not in det.lower():
                        msg = build_messages(SYSTEM_QNA, f"Answer strictly in {lang_hint}:\nQuestion: {q}")
                        ans = chat_generate(model, tokenizer, msg, MAX_NEW_TOKENS_QNA)
                    answers.append(ans.strip())
                out_payload = {"questions":[{"question":q,"answer":a} for q,a in zip(qs,answers)]}
                write_json(out_lang / "QnA" / f"{qf.stem}_answers.json", out_payload)

        torch.cuda.empty_cache()

    tprint(f"Inference complete! Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
