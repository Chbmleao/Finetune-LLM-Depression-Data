import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import Trainer, TrainingArguments, AutoTokenizer, AutoModel
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from peft import LoraConfig, get_peft_model

MODEL_NAME = "allenai/longformer-base-4096"

class TranscriptsDataset(Dataset):
  def __init__(self, dataframe, tokenizer, max_length=4096):
    self.data = dataframe
    self.tokenizer = tokenizer
    self.max_length = max_length

  def __len__(self):
    return len(self.data)

  def __getitem__(self, idx):
    text = str(self.data.iloc[idx]["text"])
    label = int(self.data.iloc[idx]["depression_label"])

    encoding = self.tokenizer(
      text,
      truncation=True,
      padding="max_length",
      max_length=self.max_length,
      return_tensors="pt"
    )

    return {
      "input_ids": encoding["input_ids"].squeeze(),
      "attention_mask": encoding["attention_mask"].squeeze(),
      "labels": torch.tensor(label, dtype=torch.long),
    }
  
class TextFeaturizer(nn.Module):
  def __init__(self, model_name, dropout=0.5, dense_size=256,
               lora_r=8, lora_alpha=16, lora_dropout=0.1):
    super().__init__() 

    # Load Longformer encoder 
    self.encoder = AutoModel.from_pretrained(model_name)
    hidden_size = self.encoder.config.hidden_size

    self.projection = nn.Sequential(
      nn.Linear(hidden_size, dense_size),
      nn.ReLU(),
      nn.Dropout(dropout)
    )

    lora_config = LoraConfig(
      r=lora_r,
      lora_alpha=lora_alpha,
      target_modules=["query", "key", "value"],
      lora_dropout=lora_dropout,
      bias="none",
      task_type="FEATURE_EXTRACTION"
    )
    self.encoder = get_peft_model(self.encoder, lora_config)

    for name, param in self.encoder.named_parameters():
      if 'lora' not in name:
        param.requires_grad = False

  def forward(self, input_ids, attention_mask):
    outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
    cls_token = outputs.last_hidden_state[:, 0]
    return self.projection(cls_token)

class TextClassifier(nn.Module):
  def __init__(self, model_name, num_labels=2):
    super().__init__()
    self.featurizer = TextFeaturizer(model_name)
    self.classifier = nn.Linear(256, num_labels)

  def forward(self, input_ids, attention_mask, labels=None):
    features = self.featurizer(input_ids, attention_mask)
    logits = self.classifier(features)

    if labels is not None:
      loss_fn = nn.CrossEntropyLoss()
      loss = loss_fn(logits, labels)
      return {"loss": loss, "logits": logits}
    return {"logits": logits}

def evaluate_model(trainer, test_dataset):
  predictions = trainer.predict(test_dataset)
  preds = np.argmax(predictions.predictions, axis=1)
  labels = predictions.label_ids

  for label, pred in zip(labels, preds):
    print(f"True: {label}, Predicted: {pred}")

  accuracy = accuracy_score(labels, preds)
  precision = precision_score(labels, preds)
  recall = recall_score(labels, preds)
  f1 = f1_score(labels, preds)

  print(f"Test Accuracy: {accuracy:.4f}")
  print(f"Test Precision: {precision:.4f}")
  print(f"Test Recall: {recall:.4f}")
  print(f"Test F1 Score: {f1:.4f}")

def train_model(df):
  tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

  train_df = df[df['split'] == 'train'].reset_index(drop=True)
  val_df = df[df['split'] == 'validation'].reset_index(drop=True)
  test_df = df[df['split'] == 'test'].reset_index(drop=True)

  train_dataset = TranscriptsDataset(train_df, tokenizer)
  val_dataset = TranscriptsDataset(val_df, tokenizer)
  test_dataset = TranscriptsDataset(test_df, tokenizer)
  
  model = TextClassifier(MODEL_NAME, num_labels=2)

  training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=1,
    per_device_eval_batch_size=1,
    num_train_epochs=10,
    gradient_accumulation_steps=4,
    fp16=True,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
  )
  trainer.train()

  evaluate_model(trainer, test_dataset)
