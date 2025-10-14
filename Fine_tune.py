import os
import json
import torch
import sys, importlib, inspect, traceback

def accelerate_compat_shim():
    try:
        if "accelerate" in sys.modules:
            accelerate = sys.modules["accelerate"]
        else:
            import accelerate

        ver = getattr(accelerate, "__version__", "unknown")
        print(f"[shim] accelerate imported from: {getattr(accelerate, '__file__', 'unknown')}")
        print(f"[shim] accelerate.__version__ = {ver}")

        sig = None
        try:
            sig = inspect.signature(accelerate.Accelerator.unwrap_model)
            print(f"[shim] unwrap_model signature: {sig}")
        except Exception as e:
            print("[shim] Could not inspect unwrap_model:", e)

        needs_shim = True
        if sig is not None:
            if "keep_torch_compile" in list(sig.parameters.keys()):
                needs_shim = False

        if needs_shim:
            print("[shim] Applying unwrap_model compatibility patch...")
            _orig_unwrap = accelerate.Accelerator.unwrap_model

            def _unwrap_compat(self, model, *args, **kwargs):
                try:
                    return _orig_unwrap(self, model, *args, **kwargs)
                except TypeError:
                    kwargs.pop("keep_torch_compile", None)
                    return _orig_unwrap(self, model, *args, **kwargs)
                except Exception:
                    kwargs.pop("keep_torch_compile", None)
                    return _orig_unwrap(self, model)

            accelerate.Accelerator.unwrap_model = _unwrap_compat
            print("[shim] Applied unwrap_model shim.")
        else:
            print("[shim] No shim needed.")
    except Exception as e:
        print("[shim] ERROR applying accelerate shim:", e)
        traceback.print_exc()

# run shim before importing transformers/Trainer
accelerate_compat_shim()

# ============================================================
# IMPORTS
# ============================================================
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding
)
from peft import LoraConfig, get_peft_model, PeftModel
from transformers import BitsAndBytesConfig


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR = ""
BASE_MODEL = ""
OUTPUT_DIR = ""

MAX_LEN = 512
NUM_EPOCHS = 1
SAVE_STEPS = 1000
BATCH_SIZE = 1
GRAD_ACCUM = 16
LR = 2e-4



def read_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    return records


def make_examples(root):
    examples = []
    for lang in os.listdir(root):
        lang_path = os.path.join(root, lang)
        if not os.path.isdir(lang_path):
            continue

        dlg_dir = os.path.join(lang_path, "Dialogues")
        sum_dir = os.path.join(lang_path, "Summary_Text")
        qna_dir = os.path.join(lang_path, "QnA")

        # --- Summarization ---
        if os.path.isdir(dlg_dir) and os.path.isdir(sum_dir):
            for fn in os.listdir(dlg_dir):
                if not fn.endswith(".jsonl"):
                    continue
                dlg_path = os.path.join(dlg_dir, fn)

                dialogues = []
                for x in read_jsonl(dlg_path):
                    try:
                        val = x.get("dialogue", "") if isinstance(x, dict) else str(x)
                        if isinstance(val, list):
                            val = " ".join(map(str, val))
                        dialogues.append(str(val))
                    except Exception:
                        continue

                dialogue_text = "\n".join(dialogues).strip()
                if not dialogue_text:
                    continue

                sum_path = os.path.join(sum_dir, fn.replace(".jsonl", "_summary.txt"))
                if os.path.exists(sum_path):
                    try:
                        with open(sum_path, "r", encoding="utf-8") as f:
                            summary = f.read().strip()
                        if summary:
                            prompt = (
                                f"Summarize the following doctor–patient dialogue in English:\n"
                                f"{dialogue_text}\nSummary:"
                            )
                            examples.append({"text": prompt, "labels": summary})
                    except Exception:
                        continue

        # --- QnA ---
        if os.path.isdir(qna_dir):
            for fn in os.listdir(qna_dir):
                if not fn.endswith(".json"):
                    continue
                try:
                    with open(os.path.join(qna_dir, fn), "r", encoding="utf-8") as f:
                        data = json.load(f)
                except Exception:
                    continue
                if not isinstance(data, dict):
                    continue
                qs = data.get("questions", [])
                if not isinstance(qs, list):
                    continue
                for qa in qs:
                    if not isinstance(qa, dict):
                        continue
                    q = qa.get("question", "")
                    a = qa.get("answer", "")
                    if not q or not a:
                        continue
                    prompt = f"Answer the following question in the same language as the dialogue:\nQuestion: {q}\nAnswer:"
                    examples.append({"text": prompt, "labels": a})

    print(f"Loaded {len(examples)} examples from {root}")
    return examples


# ============================================================
# LOAD DATA
# ============================================================
train_data = make_examples(os.path.join(DATA_DIR, "train"))
dev_data = make_examples(os.path.join(DATA_DIR, "dev"))
print(f"Train: {len(train_data)}, Dev: {len(dev_data)}")


# ============================================================
# MODEL + QLORA
# ============================================================
bnb_cfg = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16
)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, use_fast=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    device_map="auto",
    quantization_config=bnb_cfg
)

lora_cfg = LoraConfig(
    r=8,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()


def tokenize_fn(examples):
    prompts = [str(x).strip() for x in examples["text"]]
    targets = [str(y).strip() for y in examples["labels"]]

    tokenized_prompts = tokenizer(prompts, truncation=True, max_length=256, padding=False)
    prompt_lens = [len(x) for x in tokenized_prompts["input_ids"]]

    full_texts = [p + " " + t for p, t in zip(prompts, targets)]
    tokenized_full = tokenizer(full_texts, truncation=True, max_length=MAX_LEN, padding=False)

    input_ids = tokenized_full["input_ids"]
    attention_mask = tokenized_full["attention_mask"]

    labels = []
    for i, ids in enumerate(input_ids):
        l = ids.copy()
        cutoff = min(prompt_lens[i], len(l))
        for j in range(cutoff):
            l[j] = -100
        labels.append(l)

    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


train_ds = Dataset.from_list(train_data).map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text", "labels"])
dev_ds = Dataset.from_list(dev_data).map(tokenize_fn, batched=True, num_proc=4, remove_columns=["text", "labels"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True, return_tensors="pt")


from transformers import Trainer, TrainingArguments, TrainerCallback

class SavePeftAdapterCallback(TrainerCallback):
    def __init__(self, out_dir, save_steps=1000):
        self.out_dir = out_dir
        self.save_steps = save_steps
        os.makedirs(out_dir, exist_ok=True)

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.save_steps == 0 and state.global_step > 0:
            model = kwargs.get("model")
            if hasattr(model, "save_pretrained"):
                ckpt = os.path.join(self.out_dir, f"adapter_step_{state.global_step}")
                model.save_pretrained(ckpt)
                print(f"[callback] Saved adapter checkpoint at step {state.global_step}")


args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LR,
    fp16=True,
    eval_strategy="epoch",
    save_strategy="steps",
    save_steps=SAVE_STEPS,
    save_total_limit=3,
    logging_steps=50,
    report_to="none",
    dataloader_num_workers=4,
)



def find_latest_checkpoint(output_dir):
    ckpts = [
        os.path.join(output_dir, d)
        for d in os.listdir(output_dir)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(output_dir, d)) and not d.endswith("-safe")
    ]
    if not ckpts:
        return None
    ckpts = sorted(ckpts, key=lambda x: int(x.split("-")[-1]))
    return ckpts[-1]


def convert_checkpoint_to_safetensors(model, src_path, dst_path):
    os.makedirs(dst_path, exist_ok=True)
    print(f"Converting checkpoint from {src_path} to {dst_path} (safetensors format)...")
    model = PeftModel.from_pretrained(model, src_path)
    model.save_pretrained(dst_path, safe_serialization=True)
    print(f"Safetensors checkpoint saved at: {dst_path}")
    return dst_path


trainer = Trainer(
    model=model,
    args=args,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    data_collator=data_collator,
    callbacks=[SavePeftAdapterCallback(OUTPUT_DIR, SAVE_STEPS)]
)

latest_ckpt = find_latest_checkpoint(OUTPUT_DIR)
safe_ckpt = None

if latest_ckpt:
    print(f"Found existing checkpoint: {latest_ckpt}")
    torch_ver = tuple(map(int, torch.__version__.split(".")[:2]))
    if torch_ver < (2, 6):
        safe_ckpt = latest_ckpt + "-safe"
        if not os.path.exists(safe_ckpt):
            safe_ckpt = convert_checkpoint_to_safetensors(model, latest_ckpt, safe_ckpt)
        else:
            print(f"Using existing safetensors checkpoint: {safe_ckpt}")
    else:
        safe_ckpt = latest_ckpt
else:
    print("No existing checkpoint found. Starting fresh training run.")
    safe_ckpt = None



if safe_ckpt:
    print(f"Found checkpoint folder: {safe_ckpt}")
    trainer_state_path = os.path.join(safe_ckpt, "trainer_state.json")

    if os.path.exists(trainer_state_path):
        # Full Trainer checkpoint (rare in this setup)
        print("Full checkpoint detected — resuming complete training state...")
        trainer.train(resume_from_checkpoint=safe_ckpt)
    else:
        # Only adapter weights exist
        print(" No trainer_state.json found — loading adapter weights only.")
        model = PeftModel.from_pretrained(model, safe_ckpt)


        for name, param in model.named_parameters():
            if "lora" in name:
                param.requires_grad = True
        
                
        model.train()
        trainer.model = model
        model.print_trainable_parameters()
        print("Continuing training from adapter weights...")
        trainer.train()
else:
    print("Starting fresh training...")
    trainer.train()

trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print("Training complete! Model saved to:", OUTPUT_DIR)
