import os, json, torch, inspect, traceback
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    BitsAndBytesConfig,
    TrainerCallback
)
from peft import LoraConfig, get_peft_model


# ============================================================
# ACCELERATE SHIM (unwrap_model fix for older versions)
# ============================================================
def accelerate_compat_shim():
    try:
        import accelerate
        sig = inspect.signature(accelerate.Accelerator.unwrap_model)
        if "keep_torch_compile" not in sig.parameters:
            _orig = accelerate.Accelerator.unwrap_model
            def _shim(self, model, *a, **kw):
                kw.pop("keep_torch_compile", None)
                return _orig(self, model, *a, **kw)
            accelerate.Accelerator.unwrap_model = _shim
            print("[shim] unwrap_model patched for Accelerate.")
        else:
            print("[shim] No patch needed.")
    except Exception as e:
        print("[shim] Accelerate shim failed:", e)
        traceback.print_exc()

accelerate_compat_shim()


# ============================================================
# CONFIGURATION
# ============================================================
DATA_DIR   = ""
BASE_MODEL = ""
OUTPUT_DIR = ""

SEED = 42
CTX_MAX = 2048
PROMPT_MAX = 768
TARGET_MAX = CTX_MAX - PROMPT_MAX
WINDOW_OVERLAP = 256
BATCH_SIZE = 1
GRAD_ACCUM = 8
NUM_EPOCHS = 1
LR = 2e-4
SAVE_STEPS = 1000
FP16 = True


# ============================================================
# HELPERS
# ============================================================
def tprint(msg): 
    print(f"[{msg}]", flush=True)

def safe_read_jsonl(path):
    rows=[]
    with open(path,"r",encoding="utf-8",errors="replace") as f:
        for line in f:
            s=line.strip()
            if s:
                try: rows.append(json.loads(s))
                except: continue
    return rows

def join_dialogue(rows):
    parts=[]
    for r in rows:
        if isinstance(r,dict):
            val=r.get("dialogue","")
            if isinstance(val,list): val=" ".join(map(str,val))
            parts.append(str(val))
        else:
            parts.append(str(r))
    return "\n".join(parts).strip()


# ============================================================
# LOAD TRAINING DATA
# ============================================================
def make_examples(split):
    examples=[]
    for lang in os.listdir(split):
        ldir=os.path.join(split,lang)
        if not os.path.isdir(ldir): 
            continue
        dlg=os.path.join(ldir,"Dialogues")
        summ=os.path.join(ldir,"Summary_Text")
        qna=os.path.join(ldir,"QnA")

        # --- Summaries ---
        if os.path.isdir(dlg) and os.path.isdir(summ):
            for fn in os.listdir(dlg):
                if not fn.endswith(".jsonl"): continue
                rows=safe_read_jsonl(os.path.join(dlg,fn))
                dialogue=join_dialogue(rows)
                if not dialogue: continue
                sfile=os.path.join(summ,fn.replace(".jsonl","_summary.txt"))
                if not os.path.exists(sfile): continue
                text=open(sfile,"r",encoding="utf-8").read().strip()
                if not text: continue
                prompt=f"Summarize the following doctor–patient dialogue in English:\n{dialogue}\n\nSummary:"
                examples.append({"prompt":prompt,"target":text})

        # --- QnA ---
        if os.path.isdir(qna):
            for fn in os.listdir(qna):
                if not fn.endswith(".json"): continue
                try: data=json.load(open(os.path.join(qna,fn),encoding="utf-8"))
                except: continue
                for qa in data.get("questions",[]):
                    q=qa.get("question","").strip()
                    a=qa.get("answer","").strip()
                    if not q or not a: continue
                    prompt=f"Answer the following question in the same language:\nQuestion: {q}\nAnswer:"
                    examples.append({"prompt":prompt,"target":a})

    tprint(f"Loaded {len(examples)} examples from {split}")
    return examples


# ============================================================
# TOKENIZATION + CHUNKING
# ============================================================
def build_tokenizer(path):
    tok=AutoTokenizer.from_pretrained(path, use_fast=True)
    if tok.pad_token is None: 
        tok.pad_token=tok.eos_token
    return tok

def sliding_chunks(ids,size,overlap):
    if len(ids)<=size: return [ids]
    step=size-overlap
    return [ids[i:i+size] for i in range(0,len(ids),step)]

def map_fn(tokenizer):
    def _map(batch):
        out={"input_ids":[],"attention_mask":[],"labels":[]}
        for p,t in zip(batch["prompt"],batch["target"]):
            p_ids=tokenizer.encode(p,add_special_tokens=False)[-PROMPT_MAX:]
            t_ids=tokenizer.encode(" "+t,add_special_tokens=False)[:TARGET_MAX]
            full=p_ids+t_ids
            for ch in sliding_chunks(full,CTX_MAX,WINDOW_OVERLAP):
                att=[1]*len(ch)
                plen=min(len(ch),len(p_ids))
                labels=[-100]*plen+ch[plen:]
                out["input_ids"].append(ch)
                out["attention_mask"].append(att)
                out["labels"].append(labels)
        return out
    return _map


# ============================================================
# COLLATOR
# ============================================================
class Collator:
    def __init__(self,tok): self.tok=tok
    def __call__(self,batch):
        m=max(len(x["input_ids"]) for x in batch)
        def pad(lst,val): return lst+[val]*(m-len(lst))
        ids=[pad(x["input_ids"],self.tok.pad_token_id) for x in batch]
        att=[pad(x["attention_mask"],0) for x in batch]
        lbl=[pad(x["labels"],-100) for x in batch]
        return {
            "input_ids":torch.tensor(ids),
            "attention_mask":torch.tensor(att),
            "labels":torch.tensor(lbl)
        }


# ============================================================
# MODEL + QLoRA
# ============================================================
def build_model():
    bnb_cfg=BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16
    )
    model=AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        device_map="auto",
        quantization_config=bnb_cfg,
        torch_dtype=torch.float16
    )
    model.gradient_checkpointing_enable()
    model.config.use_cache=False
    if hasattr(model,"enable_input_require_grads"):
        model.enable_input_require_grads()

    lora=LoraConfig(
        r=8,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","v_proj"],
        task_type="CAUSAL_LM"
    )
    model=get_peft_model(model,lora)
    model.print_trainable_parameters()
    return model


# ============================================================
# CALLBACK: Save adapter checkpoints
# ============================================================
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
                model.save_pretrained(ckpt, safe_serialization=True)
                print(f"[callback] Saved adapter-only checkpoint at step {state.global_step}")


# ============================================================
# TRAINING LOOP (safe resume for Torch < 2.6)
# ============================================================
def main():
    torch.manual_seed(SEED)
    tok=build_tokenizer(BASE_MODEL)
    train_raw=make_examples(os.path.join(DATA_DIR,"train"))
    dev_raw=make_examples(os.path.join(DATA_DIR,"dev"))

    train=Dataset.from_list(train_raw).map(map_fn(tok),batched=True,num_proc=4,remove_columns=["prompt","target"])
    dev=Dataset.from_list(dev_raw).map(map_fn(tok),batched=True,num_proc=4,remove_columns=["prompt","target"])

    model=build_model()
    args=TrainingArguments(
        output_dir=OUTPUT_DIR,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        num_train_epochs=NUM_EPOCHS,
        learning_rate=LR,
        fp16=FP16,
        logging_steps=50,
        save_strategy="steps",
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        eval_strategy="epoch",
        report_to="none",
        save_safetensors=True,
        max_steps=28000
    )

    trainer=Trainer(
        model=model,
        args=args,
        train_dataset=train,
        eval_dataset=dev,
        data_collator=Collator(tok),
        callbacks=[SavePeftAdapterCallback(OUTPUT_DIR, SAVE_STEPS)]
    )

    # ---- Resume from latest checkpoint
    ckpt=None
    if os.path.isdir(OUTPUT_DIR):
        ckpts=[d for d in os.listdir(OUTPUT_DIR) if d.startswith(("checkpoint-","adapter_step_"))]
        if ckpts:
            ckpt=os.path.join(OUTPUT_DIR,sorted(ckpts,key=lambda x:int(''.join(filter(str.isdigit,x))) or 0)[-1])

    if ckpt:
        tprint(f"Resuming from {ckpt}")
        try:
            trainer.train(resume_from_checkpoint=ckpt)
        except ValueError as e:
            if "vulnerability issue in `torch.load`" in str(e):
                print("[safe-resume] Torch < 2.6 detected; resuming weights only (optimizer reset).")
                trainer.train(resume_from_checkpoint=None)
            else:
                raise
    else:
        tprint("Starting fresh training")
        trainer.train()

    tprint("Saving final model…")
    trainer.save_model(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    tprint("Training complete!")


if __name__=="__main__":
    main()
