import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset
from transformers import AutoTokenizer
import torch, random, numpy as np

SEED = 4213
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

df = pd.read_csv("combine_data_clean.csv")
df = df[["text", "is_sarcastic"]].dropna()
df["is_sarcastic"] = df["is_sarcastic"].astype(int)
df = df.rename(columns={"is_sarcastic": "label"})

pool_df, _ = train_test_split(
    df,
    test_size=0.8, # keep 20%
    random_state=SEED,
    stratify=df["label"]
)

train_pool_df, temp_pool_df = train_test_split(
    pool_df,
    test_size=0.2, # 80/20 within the 20% pool
    random_state=SEED,
    stratify=pool_df["label"]
)

val_df, test_df = train_test_split(
    temp_pool_df,
    test_size=0.5, # 50/50 of the 20% pool remainder
    random_state=SEED,
    stratify=temp_pool_df["label"]
)

print(f"Pool size (20%): {len(pool_df)}  | Train: {len(train_pool_df)}  Val: {len(val_df)}  Test: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(batch):
    return tokenizer(
        batch["text"],
        padding="max_length",
        truncation=True,
        max_length=128
    )

train_ds = Dataset.from_pandas(train_pool_df.reset_index(drop=True))
val_ds   = Dataset.from_pandas(val_df.reset_index(drop=True))
test_ds  = Dataset.from_pandas(test_df.reset_index(drop=True))

train_ds = train_ds.map(tokenize_function, batched=True)
val_ds   = val_ds.map(tokenize_function, batched=True)
test_ds  = test_ds.map(tokenize_function, batched=True)

import optuna
from transformers import (
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
import torch

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    return {"accuracy": acc, "f1": f1}

def objective(trial):
    model_name = "roberta-base"
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2)

    #Suggest hyperparameters
    learning_rate = trial.suggest_float("lr", 1e-6, 5e-5, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    weight_decay = trial.suggest_float("weight_decay", 0.0, 0.1)
    num_epochs = trial.suggest_int("epochs", 2, 4)

    #Training arguments
    args = TrainingArguments(
        output_dir=f"results_roberta/trial_{trial.number}",
        eval_strategy="epoch",
        save_strategy="epoch",
        learning_rate=learning_rate,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=num_epochs,
        weight_decay=weight_decay,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        logging_dir="logs",
        logging_steps=20,
        seed=4213,
        disable_tqdm=False,
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
    )

    trainer.train()

    #Evaluate on validation set
    metrics = trainer.evaluate(eval_dataset=val_ds)
    return metrics["eval_f1"]  # Optimize for F1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=6)
print("Best trial:")
print(f"  Value (F1): {study.best_trial.value}")
print("  Params: ")
for key, value in study.best_trial.params.items():
    print(f"    {key}: {value}")

import time
from xml.parsers.expat import model
import torch
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
import optuna

SEED = 4213
np.random.seed(SEED)
torch.manual_seed(SEED)

train_df, temp_df = train_test_split(df, test_size=0.2, random_state=SEED, stratify=df["label"])
val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=SEED, stratify=temp_df["label"])

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

tokenizer = AutoTokenizer.from_pretrained("roberta-base")

def tokenize_function(batch):
    return tokenizer(batch["text"], padding="max_length", truncation=True, max_length=128)

train_ds = Dataset.from_pandas(train_df).map(tokenize_function, batched=True)
val_ds   = Dataset.from_pandas(val_df).map(tokenize_function, batched=True)
test_ds  = Dataset.from_pandas(test_df).map(tokenize_function, batched=True)

cols = ["input_ids", "attention_mask", "label"]
train_ds.set_format(type="torch", columns=cols)
val_ds.set_format(type="torch", columns=cols)
test_ds.set_format(type="torch", columns=cols)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    f1 = f1_score(labels, preds)
    try:
        roc_auc = roc_auc_score(labels, torch.softmax(torch.tensor(logits), dim=1)[:, 1].numpy())
    except:
        roc_auc = float("nan")
    return {"accuracy": acc, "f1": f1, "roc_auc": roc_auc}

# Load study
best_params = study.best_trial.params
best_model = AutoModelForSequenceClassification.from_pretrained("roberta-base", num_labels=2)

training_args = TrainingArguments(
    output_dir="full_finetuned_roberta",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=best_params.get("lr", 3e-5),
    per_device_train_batch_size=best_params.get("batch_size", 16),
    per_device_eval_batch_size=best_params.get("batch_size", 16),
    num_train_epochs=best_params.get("epochs", 3),
    weight_decay=best_params.get("weight_decay", 0.01),
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    logging_dir="logs_final",
    seed=SEED,
)

trainer = Trainer(
    model=best_model,
    args=training_args,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
)

print("Starting final full fine-tuning...")
start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"Training completed in {train_time/60:.2f} minutes")

val_metrics = trainer.evaluate(eval_dataset=val_ds)
test_preds = trainer.predict(test_ds)
test_metrics = compute_metrics((test_preds.predictions, test_preds.label_ids))

print("\n===== VALIDATION METRICS =====")
for k, v in val_metrics.items():
    if k.startswith("eval_"):
        print(f"{k}: {v:.4f}")

print("\n===== TEST METRICS =====")
for k, v in test_metrics.items():
    print(f"{k}: {v:.4f}")

import os
from sklearn.metrics import roc_curve, auc

os.makedirs("results_roberta", exist_ok=True)
y_true = test_preds.label_ids
y_pred = np.argmax(test_preds.predictions, axis=-1)
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(6, 6))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Not Sarcastic", "Sarcastic"])
disp.plot(cmap="Blues", ax=ax, colorbar=False)
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("results_roberta/confusion_matrix_test.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("Confusion matrix saved to results_roberta/confusion_matrix_test.png")

# get predicted probabilities for the positive class (sarcastic)
probs = torch.softmax(torch.tensor(test_preds.predictions), dim=1)[:, 1].numpy()
fpr, tpr, thresholds = roc_curve(y_true, probs)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(6, 6))
ax.plot(fpr, tpr, color="blue", lw=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
ax.plot([0, 1], [0, 1], color="gray", linestyle="--")
ax.set_xlabel("False Positive Rate")
ax.set_ylabel("True Positive Rate")
ax.set_title("ROC Curve - Test Set")
ax.legend(loc="lower right")
plt.tight_layout()
plt.savefig("results_roberta/roc_curve_test.png", bbox_inches="tight", dpi=300)
plt.close(fig)
print("ROC curve saved to results_roberta/roc_curve_test.png")

model_dir = "results_roberta/best_full_finetuned_roberta"
trainer.save_model(model_dir)

# Also save as .pt checkpoint
pt_path = os.path.join("results_roberta", "best_full_finetuned_roberta.pt")
torch.save(best_model.state_dict(), pt_path)
print(f"Model weights saved as {pt_path}")

summary = {
    "Best Parameters": best_params,
    "Training Time (min)": round(train_time / 60, 2),
    "Validation Metrics": {k: round(v, 4) for k, v in val_metrics.items() if k.startswith("eval_")},
    "Test Metrics": {k: round(v, 4) for k, v in test_metrics.items()},
    "ROC AUC": round(roc_auc, 4)
}

print("\n===== SUMMARY =====")
for k, v in summary.items():
    print(f"{k}: {v}")