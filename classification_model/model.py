import optuna
import shutil
import json
from copy import deepcopy
import os
import numpy as np
import pandas as pd
from transformers import (
  Trainer, TrainingArguments, AutoTokenizer, AutoModel,
  AutoModelForSequenceClassification, EarlyStoppingCallback,
  TrainerCallback
)
from tensorflow.keras.utils import to_categorical
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, r2_score, confusion_matrix, root_mean_squared_error, mean_absolute_error
from sklearn.model_selection import ParameterGrid
from collections import Counter
from peft import LoraConfig, get_peft_model

# Global constants
MAX_TIME = 0
NUM_BINS = 0
TRANSFORMERS_MODEL = "bert-base-uncased" 
TASK_TYPE = "classification"
MAX_TEXT_LENGTH = 512
TOKENIZER = None

# -------------- Utility functions --------------

def get_dynamic_max_length(texts, tokenizer, percentile=95):
  lengths = [len(tokenizer(text)["input_ids"]) for text in texts]
  return int(np.percentile(lengths, percentile))

def set_global_variables(audio_df_path, transcript_df_path, task_type):
  global MAX_TIME, NUM_BINS, TASK_TYPE, MAX_TEXT_LENGTH, TOKENIZER

  TASK_TYPE = task_type
  TOKENIZER = AutoTokenizer.from_pretrained(TRANSFORMERS_MODEL)

  train_texts = retrieve_csv_file(f"{transcript_df_path}/train_transcripts.csv")["Text"].tolist()
  val_texts = retrieve_csv_file(f"{transcript_df_path}/val_transcripts.csv")["Text"].tolist()
  test_texts = retrieve_csv_file(f"{transcript_df_path}/test_transcripts.csv")["Text"].tolist()
  all_texts = train_texts + val_texts + test_texts
  # MAX_TEXT_LENGTH = get_dynamic_max_length(all_texts, TOKENIZER)
  MAX_TEXT_LENGTH = 512 

  max_time, num_bins = 0, 0
  for prefix in ["train", "val", "test"]:
    X = retrieve_npz_file(f"{audio_df_path}/{prefix}_samples.npz")
    print(f"Processing {prefix} set with {len(X)} samples")
    max_time = max(max_time, max(x.shape[0] for x in X))
    num_bins = max(num_bins, X[0].shape[1])
  print(f"Max time: {max_time}, Num bins: {num_bins}")
  MAX_TIME = max_time
  NUM_BINS = num_bins

def retrieve_npz_file(file_path, column_name="arr_0"):
  with np.load(file_path, allow_pickle=True) as data:
    return data[column_name]

def retrieve_csv_file(file_path):
  return pd.read_csv(file_path)

# -------------- Dataset functions --------------

def preprocess_audio(X):
  X_proc = np.zeros((len(X), MAX_TIME, NUM_BINS), dtype=np.float32)
  for idx, x in enumerate(X):
    len_x = min(x.shape[0], MAX_TIME)
    X_proc[idx, :len_x, :] = x[:len_x, :]
  return X_proc

def get_text_dataset(set_name, transcripts_df_path):
  labels_column = {
    "classification": "Targets",
    "regression": "Scores",
    "ordinal": "Ordinal"
  }.get(TASK_TYPE)

  df = retrieve_csv_file(f"{transcripts_df_path}/{set_name}_transcripts.csv")
  texts = df["Text"].tolist()
  labels = df[labels_column].tolist()
  tokenizer = TOKENIZER

  return TextDataset(texts, labels, tokenizer)

class TextDataset(Dataset):
  def __init__(self, texts, labels, tokenizer):
    self.texts = texts
    self.labels = labels
    self.tokenizer = tokenizer
    self.max_length = MAX_TEXT_LENGTH

  def __len__(self):
    return len(self.texts)

  def __getitem__(self, idx):
    encoding = self.tokenizer(
      self.texts[idx],
      truncation=True,
      padding="max_length",
      max_length=self.max_length,
      return_tensors="pt"
    )
    item = {key: val.squeeze(0) for key, val in encoding.items()}
    dtype = torch.long if TASK_TYPE in ["classification", "ordinal"] else torch.float
    item["labels"] = torch.tensor(self.labels[idx], dtype=dtype)
    return item

class MultimodalDataset(Dataset):
  def __init__(self, set_name, df_dir):
    labels = retrieve_npz_file(f"{df_dir}/{labels}_labels.npz")
    labels = labels.astype(np.int32)
    self.y = to_categorical(labels, num_classes=2)
    self.num_outputs = 2


    self.X_audios = retrieve_npz_file(f"{audios_df_path}/{set_name}_samples.npz")
    self.X_audios = preprocess_audio(self.X_audios)

    self.text_dataset = get_text_dataset(set_name, transcripts_df_path)

  def __len__(self):
    return len(self.X_audios)

  def __getitem__(self, idx):
    audio_tensor = torch.tensor(self.X_audios[idx], dtype=torch.float32)
    text_data = self.text_dataset[idx]
    label_tensor = torch.tensor(self.y[idx])
    return {
      "audio": audio_tensor,
      **text_data,
      "labels": label_tensor
    }

def print_dataset_info(dataset, set_name):
  print()
  print(f"Dataset: {set_name}")
  print(f"Number of samples: {len(dataset)}")
  print(f"Sample audio shape: {dataset[0]['audio'].shape}")
  print(f"Sample text input IDs shape: {dataset[0]['input_ids'].shape}")
  print(f"Sample text attention mask shape: {dataset[0]['attention_mask'].shape}")
  print(f"Sample labels shape: {dataset[0]['labels'].shape}")

  labels = dataset.y
  print(f"Unique labels: {np.unique(labels)}")
  label_counts = Counter(np.argmax(labels, axis=1)) if len(labels.shape) > 1 else Counter(labels)
  print("\nLabel distribution:")
  for label, count in sorted(label_counts.items()):
    print(f"  Label {label}: {count} samples ({count / len(labels) * 100:.2f}%)")

  print()

# -------------- Model functions --------------

def get_activation_layer(name):
  if name == "relu":
    return nn.ReLU()
  elif name == "leaky_relu":
    return nn.LeakyReLU(0.01)
  elif name == "elu":
    return nn.ELU(1.0)
  elif name == "swish":
    return nn.SiLU()
  elif name == "gelu":
    return nn.GELU()
  else:
    raise ValueError(f"Unknown activation: {name}")

class AudioCNNFeaturizer(nn.Module):
  def __init__(self, num_bins, max_time, dense_size=64, dropout_rate=0.5):
    super().__init__()
    self.conv1 = nn.Conv1d(num_bins, 256, kernel_size=64, padding='same')
    self.bn1 = nn.BatchNorm1d(256)
    self.pool1 = nn.MaxPool1d(kernel_size=2)
    
    self.conv2 = nn.Conv1d(256, 256, kernel_size=64, padding='same')
    self.bn2 = nn.BatchNorm1d(256)
    self.pool2 = nn.MaxPool1d(kernel_size=2)

    self.conv3 = nn.Conv1d(256, 256, kernel_size=64, padding='same')
    self.bn3 = nn.BatchNorm1d(256)

    self.dropout = nn.Dropout(dropout_rate)

    self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
    self.global_max_pool = nn.AdaptiveMaxPool1d(1)

    self.dense = nn.Sequential(
      nn.Linear(256 * 2, dense_size),
      nn.ReLU(),
      nn.Dropout(dropout_rate)
    )

  def forward(self, x):
    x = x.permute(0, 2, 1)

    x = self.pool1(F.relu(self.bn1(self.conv1(x))))
    x = self.pool2(F.relu(self.bn2(self.conv2(x))))
    x = F.relu(self.bn3(self.conv3(x)))
    x = self.dropout(x)

    avg_pool = self.global_avg_pool(x).squeeze(-1)
    max_pool = self.global_max_pool(x).squeeze(-1)
    x = torch.cat([avg_pool, max_pool], dim=1)

    return self.dense(x)
  
class TextFeaturizer(nn.Module):
  def __init__(self, model_name, dropout=0.5, dense_size=256, use_lora=False, lora_r=8, lora_alpha=32, lora_dropout=0.1):
    super().__init__()

    # Load base encoder 
    self.encoder = AutoModel.from_pretrained(model_name)
    hidden_size = self.encoder.config.hidden_size

    # Projection head
    self.projection = nn.Sequential(
      nn.Linear(hidden_size, dense_size),
      nn.ReLU(),
      nn.Dropout(dropout)
    )

    # Add LoRA adapters to the encoder and freeze base weights
    self.use_lora = use_lora
    if self.use_lora:
      print("Using LoRA adapters for TextFeaturizer.")

      # Apply to attention query/value matrices
      lora_config = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        target_modules=["query", "value"],
        lora_dropout=lora_dropout,
        bias="none",
        task_type="FEATURE_EXTRACTION"
      )

      # Wrap the model with PEFT
      self.encoder = get_peft_model(self.encoder, lora_config)

      # Freeze base model parameters
      for name, param in self.encoder.named_parameters():
        if "lora" not in name:
          param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0]
    return self.projection(cls_token)

class MultimodalModel(nn.Module):
  def __init__(self, num_bins, max_time,
                num_dense_layers=3, dense_size=128, dropout=0.5, activation="relu", use_lora=False):
    super().__init__()
    text_featurizer_dense_size = 256
    audio_featurizer_dense_size = 64
    self.text_model = TextFeaturizer(TRANSFORMERS_MODEL, dropout=dropout, dense_size=text_featurizer_dense_size, use_lora=use_lora)
    self.audio_model = AudioCNNFeaturizer(num_bins, max_time, dense_size=audio_featurizer_dense_size)

    combined_dim = text_featurizer_dense_size + audio_featurizer_dense_size

    layers = []
    input_dim = combined_dim
    for _ in range(num_dense_layers):
      layers.append(nn.Linear(input_dim, dense_size))
      layers.append(nn.BatchNorm1d(dense_size))
      layers.append(get_activation_layer(activation))
      layers.append(nn.Dropout(dropout))
      input_dim = dense_size

    self.fusion_mlp = nn.Sequential(*layers)

    if TASK_TYPE == "classification":
      self.out = nn.Linear(dense_size, 2)
    elif TASK_TYPE == "ordinal":
      self.out = nn.Linear(dense_size, 5)
    else:  # regression
      self.out = nn.Sequential(
        nn.Linear(dense_size, 1),
        nn.ReLU()
      )

  def forward(self, input_ids=None, attention_mask=None, audio=None, labels=None):
    # Text features
    text_feat = self.text_model(input_ids, attention_mask)

    # Audio features
    audio_feat = self.audio_model(audio)

    # Combine and predict
    combined = torch.cat([text_feat, audio_feat], dim=1)
    fused = self.fusion_mlp(combined)
    logits = self.out(fused)

    if labels is not None:
      if TASK_TYPE == "classification":
        loss_fn = nn.CrossEntropyLoss()
        loss = loss_fn(logits, torch.argmax(labels, dim=1))
      elif TASK_TYPE == "ordinal":
        loss_fn = nn.BCEWithLogitsLoss()
        loss = loss_fn(logits, labels.float())
      else:
        loss_fn = nn.MSELoss()
        loss = loss_fn(logits.squeeze(), labels.float())
      return {"loss": loss, "logits": logits}

    return {"logits": logits}

# -------------- Training and evaluation functions --------------

def evaluate_model(trainer, test_dataset, use_tta=False):
  print("\nEvaluating on test set...")

  labels = test_dataset.y
  eval_results = trainer.evaluate(test_dataset)
  predictions = trainer.predict(test_dataset)
  logits = predictions.predictions


  if use_tta and test_dataset.tta_mapping is not None:
    aggregated_logits = []
    aggregated_labels = []

    for i, (pid, start_idx, end_idx) in enumerate(test_dataset.tta_mapping):
      tta_logits = logits[start_idx:end_idx]
      avg_logits = np.mean(tta_logits, axis=0)
      aggregated_logits.append(avg_logits)
      aggregated_labels.append(labels[start_idx])

    logits = np.array(aggregated_logits)
    labels = np.array(aggregated_labels)

  print("\nEvaluation results:")
  for key, value in eval_results.items():
    print(f"{key}: {value:.4f}")

  print("\nSample predictions and labels:")
  for i in range(logits.shape[0]):
    if TASK_TYPE == "classification":
      pred_label = np.argmax(logits[i])
      true_label = np.argmax(labels[i])
    elif TASK_TYPE == "ordinal":
      pred_label = np.argmax(logits[i])
      true_label = np.argmax(labels[i])
    else:  # regression
      pred_label = logits[i].item()
      true_label = labels[i].item()
    
    print(f"Sample {i+1}: Predicted: {pred_label}, True: {true_label}")
  print("\n")  

  if TASK_TYPE == "classification":
    preds = np.argmax(logits, axis=1)
    true = np.argmax(labels, axis=1)
    precision = precision_score(true, preds, average="weighted")
    recall = recall_score(true, preds, average="weighted")
    f1 = f1_score(true, preds, average="weighted")
    acc = accuracy_score(true, preds)
    print(f"\nClassification metrics:")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    cm = confusion_matrix(true, preds)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm))
    return f1

  elif TASK_TYPE == "ordinal":
    true = np.argmax(labels, axis=1)
    preds = np.argmax(logits, axis=1)
    mse = mean_squared_error(true, preds)
    r2 = r2_score(true, preds)
    print(f"\nOrdinal regression metrics (using predicted class index):")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"R² Score: {r2:.4f}")

    cm = confusion_matrix(true, preds)
    print("\nConfusion Matrix:")
    print(pd.DataFrame(cm))
    return r2

  else:
    preds = logits.squeeze()
    true = labels.squeeze()
    mse = mean_squared_error(true, preds)
    rmse = root_mean_squared_error(true, preds)
    mae = mean_absolute_error(true, preds)
    r2 = r2_score(true, preds)
    print(f"\nRegression metrics:")
    print(f"Mean Squared Error: {mse:.4f}")
    print(f"Root Mean Squared Error: {rmse:.4f}")
    print(f"Mean Absolute Error: {mae:.4f}")
    print(f"R² Score: {r2:.4f}")
    return mse

def compute_metrics(eval_pred):
  logits, labels = eval_pred
  preds = logits.argmax(axis=1)
  
  if labels.ndim > 1:
    labels = labels.argmax(axis=1)

  if TASK_TYPE == "classification":
    acc = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    return {
      "accuracy": acc,
      "precision": precision,
      "recall": recall,
      "f1": f1
    }
  else:
    mse = mean_squared_error(labels, preds)
    rmse = root_mean_squared_error(labels, preds)
    mae = mean_absolute_error(labels, preds)
    r2 = r2_score(labels, preds)
    return {
      "rmse": rmse,
      "mae": mae,
      "mse": mse,
      "r2": r2
    }
  
class ReduceLROnPlateauCallback(TrainerCallback):
  def __init__(self, scheduler, metric_name="eval_loss"):
    super().__init__()
    self.scheduler = scheduler
    self.metric_name = metric_name

  def on_evaluate(self, args, state, control, metrics, **kwargs):
    metric = metrics.get(self.metric_name)
    if metric is not None:
      self.scheduler.step(metric)
      current_lr = self.scheduler.optimizer.param_groups[0]['lr']
      print(f"Reducing learning rate to {current_lr:.6f} based on {self.metric_name}: {metric:.4f}")

  def on_log(self, args, state, control, logs=None, **kwargs):
    current_lr = self.scheduler.optimizer.param_groups[0]["lr"]
    print(f"[Trainer log] Current learning rate: {current_lr:.6f}")

def train_model(audio_df_path, transcript_df_path, model_dir, 
                num_dense_layers=3, dense_size=128, dropout=0.5, activation="relu", 
                use_lora=False, use_tta=False):
  # Load datasets
  train_dataset = MultimodalDataset("train", audio_df_path, transcript_df_path)
  val_dataset = MultimodalDataset("val", audio_df_path, transcript_df_path)
  test_dataset = MultimodalDataset("test", audio_df_path, transcript_df_path, use_tta=use_tta)

  # Build model
  model = MultimodalModel(NUM_BINS, MAX_TIME,
                          num_dense_layers, dense_size, dropout, activation, use_lora=use_lora)

  # Training arguments
  training_args = TrainingArguments(
    output_dir=model_dir,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    eval_strategy="epoch",
    save_strategy="epoch",
    logging_strategy="epoch",
    save_total_limit=2,
    num_train_epochs=100,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    logging_dir=os.path.join(model_dir, "logs"),
    report_to="none",
  )

  # Optimizer
  trainable_params = [p for p in model.parameters() if p.requires_grad]
  optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

  # Scheduler
  lr_scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

  # Callbacks
  callbacks = [
    EarlyStoppingCallback(early_stopping_patience=10),
    ReduceLROnPlateauCallback(lr_scheduler, metric_name="eval_loss")
  ]

  # Print model summary
  num_trainable = sum(p.numel() for p in trainable_params)
  num_total = sum(p.numel() for p in model.parameters())
  print(f"Trainable parameters: {num_trainable:,} / {num_total:,} ({100 * num_trainable / num_total:.2f}%)")

  print("\nTrainable layers:")
  for name, param in model.named_parameters():
    if param.requires_grad:
      print(f"    {name}")

  # Trainer setup
  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=TOKENIZER,
    optimizers=(optimizer, None),
    callbacks=callbacks,
    compute_metrics=compute_metrics
  )

  print("\n=== Running pre-training evaluation (sanity check) ===")
  try:
    evaluate_model(trainer, test_dataset, use_tta=use_tta)
  except Exception as e:
    print(f"[Warning] Pre-training evaluation failed: {e}")

  # Train
  trainer.train()
  return trainer

def run_hyperparameter_optimization(audio_df_path, transcript_df_path, model_dir, task_type, 
                                    use_lora=False, use_tta=False, n_trials=20):
  set_global_variables(audio_df_path, transcript_df_path, task_type)

  # Load datasets
  train_dataset = MultimodalDataset("train", audio_df_path, transcript_df_path)
  val_dataset = MultimodalDataset("val", audio_df_path, transcript_df_path)
  test_dataset = MultimodalDataset("test", audio_df_path, transcript_df_path, use_tta=use_tta)

  print_dataset_info(train_dataset, "Train")
  print_dataset_info(val_dataset, "Validation")
  print_dataset_info(test_dataset, "Test")

  results = []
  best_score = float("inf") if task_type == "regression" else float("-inf")
  best_params = None
  best_model_path = None

  def objective(trial):
    # Define search space
    params = {
      "num_dense_layers": trial.suggest_categorical("num_dense_layers", [3, 5]),
      "dense_size": trial.suggest_categorical("dense_size", [64, 128]),
      "dropout": trial.suggest_float("dropout", 0.3, 0.5),
      "activation": trial.suggest_categorical("activation", ["swish", "relu"])
    }

    # Unique run directory
    run_dir = os.path.join(model_dir, f"trial_{trial.number}")
    os.makedirs(run_dir, exist_ok=True)

    print(f"\n=== Trial {trial.number} with parameters: {params} ===")

    trainer = train_model(audio_df_path, transcript_df_path, run_dir, **params, use_tta=use_tta, use_lora=use_lora)
    score = evaluate_model(trainer, test_dataset, use_tta=use_tta)

    # Save trial results
    results.append({**params, "score": score})

    # Check if it's the best model so far
    nonlocal best_score, best_params, best_model_path
    is_better = (
        (task_type == "regression" and score < best_score)
        or (task_type != "regression" and score > best_score)
    )

    if is_better:
      print(f"⭐ New best model found! Score: {score:.4f} | Params: {params}")

      best_score = score
      best_params = deepcopy(params)
      best_model_path = run_dir

      # --- Clear old models ---
      for item in os.listdir(model_dir):
        item_path = os.path.join(model_dir, item)
        if item_path != run_dir:  # don't delete current best
          if os.path.isdir(item_path):
            shutil.rmtree(item_path)
          else:
            os.remove(item_path)

      # --- Move best model to a clean subfolder ---
      final_best_dir = os.path.join(model_dir, "best_model")
      if os.path.exists(final_best_dir):
        shutil.rmtree(final_best_dir)
      shutil.copytree(run_dir, final_best_dir)
      print(f"✅ Saved new best model to: {final_best_dir}")

    # Tell Optuna how to minimize/maximize
    return score if task_type == "regression" else -score

  # Choose optimization direction
  direction = "minimize" if task_type == "regression" else "maximize"
  study = optuna.create_study(direction=direction)
  study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

  # Save all results
  results_df = pd.DataFrame(results)
  results_path = os.path.join(model_dir, "optuna_results.csv")
  results_df.to_csv(results_path, index=False)

  print("\nOptuna search complete!")
  print(f"Best parameters: {best_params}")
  print(f"Best score: {best_score:.4f}")
  print(f"Best model path: {os.path.join(model_dir, 'best_model')}")
  print(f"Optuna best trial: {study.best_trial.number}")

  return study, best_params, best_score, best_model_path