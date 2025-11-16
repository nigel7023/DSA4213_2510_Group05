# RESPONSE-BASED KNOWLEDGE DISTILLATION (RBKD)

import os, gc, psutil, json, optuna, torch, numpy as np, pandas as pd, evaluate
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Trainer
)
from optuna.exceptions import TrialPruned

#Clean memory
def clear_memory(tag=""):
    """Free CUDA & CPU memory."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        torch.cuda.synchronize()
    rss = psutil.Process(os.getpid()).memory_info().rss / 1024 ** 2
    print(f"🧹 Cleared memory {tag} | CPU RSS {rss:.2f} MB\n")


# DATA PREPROCESSING
data = pd.read_csv("sarcasm_KD_final.csv").fillna("")
train_df, val_df = train_test_split(data, test_size=0.2, random_state=4213)

# Prompts
PROMPT_A = (
    "In exactly 1-2 sentences, identify the specific words or phrases that make the text sarcastic "
    "and explain how they create the sarcastic effect. "
    "Focus only on observable linguistic elements without adding interpretation beyond what's directly evident in the text.\n\n"
)

PROMPT_B = (
    "In exactly 1-2 sentences, explain what the speaker actually means by removing the sarcasm "
    "and stating their true intended message directly. "
    "Focus on the genuine sentiment or opinion being expressed beneath the sarcastic language.\n\n"
)

# Build label text
def build_target(row, col):
    exp = str(row[col]).strip()
    return f"Explanation: {exp}"

train_df["target_text_A"] = train_df.apply(lambda r: build_target(r, "part_sarcastic"), axis=1)
val_df["target_text_A"]   = val_df.apply(lambda r: build_target(r, "part_sarcastic"), axis=1)
train_df["target_text_B"] = train_df.apply(lambda r: build_target(r, "sarcasm_explanation"), axis=1)
val_df["target_text_B"]   = val_df.apply(lambda r: build_target(r, "sarcasm_explanation"), axis=1)

# Convert to HF datasets
def to_hfds(df, tgt_col):
    return Dataset.from_pandas(df[["text", tgt_col]].rename(columns={tgt_col: "target_text"}))

taskA_train_ds = to_hfds(train_df, "target_text_A")
taskA_val_ds   = to_hfds(val_df, "target_text_A")
taskB_train_ds = to_hfds(train_df, "target_text_B")
taskB_val_ds   = to_hfds(val_df, "target_text_B")

# Tokenizer setup
base_model = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(base_model)
MAX_SRC, MAX_TGT = 128, 64

def make_preprocess(prompt):
    """Attach prompt and tokenize inputs + labels."""
    def _fn(examples):
        inputs = [prompt + "Text: " + t for t in examples["text"]]
        model_inputs = tokenizer(inputs, max_length=MAX_SRC, truncation=True)
        labels = tokenizer(examples["target_text"], max_length=MAX_TGT, truncation=True)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    return _fn

# Tokenize
taskA_train_tok = taskA_train_ds.map(make_preprocess(PROMPT_A), batched=True, remove_columns=taskA_train_ds.column_names)
taskA_val_tok   = taskA_val_ds.map(make_preprocess(PROMPT_A),   batched=True, remove_columns=taskA_val_ds.column_names)
taskB_train_tok = taskB_train_ds.map(make_preprocess(PROMPT_B), batched=True, remove_columns=taskB_train_ds.column_names)
taskB_val_tok   = taskB_val_ds.map(make_preprocess(PROMPT_B),   batched=True, remove_columns=taskB_val_ds.column_names)


device = "cuda" if torch.cuda.is_available() else "cpu"
rouge = evaluate.load("rouge")

def compute_metrics(eval_preds):
    """Compute ROUGE-L metric."""
    preds, labels = eval_preds
    preds = preds[0] if isinstance(preds, tuple) else preds
    if preds.dtype in [np.float32, np.float64]:
        preds = np.argmax(preds, axis=-1)

    preds = np.where(preds != -100, preds, tokenizer.pad_token_id)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    pred_texts = tokenizer.batch_decode(preds, skip_special_tokens=True)
    label_texts = tokenizer.batch_decode(labels, skip_special_tokens=True)

    r = rouge.compute(predictions=pred_texts, references=label_texts)
    return {"rougeL": round(r["rougeL"], 4)}


class KDTrainer(Trainer):
    """Custom Trainer for Response-Based KD."""
    def __init__(self, teacher_model, alpha=0.5, temperature=2, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher_model.eval()
        self.alpha = alpha
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # Student forward
        student_outputs = model(**inputs)
        loss_ce = student_outputs.loss

        # Teacher forward on its own device
        teacher_inputs = {k: v.to(self.teacher.device) for k, v in inputs.items() if torch.is_tensor(v)}
        with torch.no_grad():
            teacher_outputs = self.teacher(**teacher_inputs)

        # Align logits and compute KD loss
        t_logits = teacher_outputs.logits.to(student_outputs.logits.device)
        s_logits = student_outputs.logits / self.temperature
        t_logits = t_logits / self.temperature

        loss_kd = F.kl_div(
            F.log_softmax(s_logits, dim=-1),
            F.softmax(t_logits, dim=-1),
            reduction="batchmean",
        ) * (self.temperature ** 2)

        total_loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        return (total_loss, student_outputs) if return_outputs else total_loss


# OPTUNA HYPERPARAMETER TUNING

def subset_dataset(ds, fraction=0.2, seed=4213):
    total = len(ds)
    subset_size = int(total * fraction)
    idx = np.random.default_rng(seed).choice(total, subset_size, replace=False)
    return ds.select(idx.tolist())

data_collator = DataCollatorForSeq2Seq(tokenizer)
student_base = "google/flan-t5-small"

def make_objective(task_name, teacher_path, train_ds, val_ds):
    def objective(trial):
        lr = trial.suggest_categorical("learning_rate", [1e-4, 3e-4, 5e-4])
        bs = trial.suggest_categorical("batch_size", [4, 8, 16])
        alpha = trial.suggest_categorical("alpha", [0.3, 0.5, 0.7])
        temp = trial.suggest_categorical("temperature", [1.0, 2.0, 3.0])

        teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to(device)
        student = AutoModelForSeq2SeqLM.from_pretrained(student_base).to(device)

        args = Seq2SeqTrainingArguments(
            num_train_epochs=1,
            learning_rate=lr,
            per_device_train_batch_size=bs,
            per_device_eval_batch_size=bs,
            eval_strategy="epoch",
            save_strategy="no",
            logging_strategy="epoch",
            predict_with_generate=True,
            generation_max_length=64,
            report_to="none"
        )

        trainer = KDTrainer(
            model=student,
            teacher_model=teacher,
            alpha=alpha,
            temperature=temp,
            args=args,
            train_dataset=subset_dataset(train_ds),
            eval_dataset=subset_dataset(val_ds),
            data_collator=data_collator,
            tokenizer=tokenizer,
            compute_metrics=compute_metrics,
        )

        trainer.train()
        result = trainer.evaluate().get("eval_rougeL", 0.0)
        del teacher, student, trainer
        clear_memory(f"{task_name}_trial")
        return result
    return objective


# Run Optuna
teacherA_path = "./model_final_taskA"
teacherB_path = "./model_final_taskB"

studyA = optuna.create_study(direction="maximize")
studyA.optimize(make_objective("TaskA", teacherA_path, taskA_train_tok, taskA_val_tok), n_trials=5)

studyB = optuna.create_study(direction="maximize")
studyB.optimize(make_objective("TaskB", teacherB_path, taskB_train_tok, taskB_val_tok), n_trials=5)

bestA, bestB = studyA.best_params, studyB.best_params
os.makedirs("./optuna_results", exist_ok=True)

with open("./optuna_results/best_taskA_RBKD_params.json", "w") as f:
    json.dump(bestA, f, indent=4)
with open("./optuna_results/best_taskB_RBKD_params.json", "w") as f:
    json.dump(bestB, f, indent=4)

print("Saved best Optuna params for both tasks.")
print("TaskA:", bestA)
print("TaskB:", bestB)


with open("./optuna_results/best_taskA_RBKD_params.json", "r") as f:
    bestA = json.load(f)
with open("./optuna_results/best_taskB_RBKD_params.json", "r") as f:
    bestB = json.load(f)

def train_with_best(task_name, teacher_path, train_ds, val_ds, params):
    lr, bs, alpha, temp = params["learning_rate"], params["batch_size"], params["alpha"], params["temperature"]
    print(f"\nFull KD Training for {task_name} | α={alpha}, T={temp}, lr={lr}, bs={bs}")

    teacher = AutoModelForSeq2SeqLM.from_pretrained(teacher_path).to("cpu")
    student = AutoModelForSeq2SeqLM.from_pretrained(student_base).to("cpu")

    collator = DataCollatorForSeq2Seq(tokenizer, model=student)
    args = Seq2SeqTrainingArguments(
        num_train_epochs=3,
        learning_rate=lr,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=64,
        report_to="none"
    )

    trainer = KDTrainer(
        model=student,
        teacher_model=teacher,
        alpha=alpha,
        temperature=temp,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=collator,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    print(f"Train size: {len(train_ds)} | Eval size: {len(val_ds)}")
    trainer.train()
    trainer.save_model(f"./studentRBKD_{task_name}")
    del teacher, student, trainer
    clear_memory(f"{task_name}_final")


train_with_best("TaskA", teacherA_path, taskA_train_tok, taskA_val_tok, bestA)
train_with_best("TaskB", teacherB_path, taskB_train_tok, taskB_val_tok, bestB)


