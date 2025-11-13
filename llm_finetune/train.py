from data_loader import load_daic_data
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments, Trainer, DataCollatorForLanguageModeling


def get_tokenizer_and_early_model(model_name="TinyLlama/TinyLlama-1.1B-Chat-v1.0"):
  tokenizer = AutoTokenizer.from_pretrained(model_name)
  model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_8bit=False,
    device_map="auto"
  )
  return tokenizer, model, model_name

def get_lora_model(model):
  lora_config = LoraConfig(
    r=8, # rank
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"], # depends on model architecture
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
  )

  model = get_peft_model(model, lora_config)
  model.print_trainable_parameters()
  return model

def fine_tune_model(
  model, 
  tokenizer, 
  tokenized_datasets,
  output_dir="./tiny_llama_instruction_tuned",
  ):
  training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    warmup_steps=10,
    max_steps=1000,
    learning_rate=2e-4,
    fp16=True,
    logging_steps=10,
    save_steps=50
  )

  data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets,
    data_collator=data_collator,
  )
  trainer.train()
  model.save_pretrained(output_dir)
  tokenizer.save_pretrained(output_dir)

def use_tokenizer(tokenizer, text):
  return tokenizer(text, truncation=True, padding='max_length', max_length=512)

if __name__ == "__main__":
  print('Loading tokenizer and model...')
  tokenizer, model, model_name = get_tokenizer_and_early_model()

  print('Loading dataset...')
  tokenized_dataset = load_daic_data(tokenizer, should_create_csv=False)

  print('Getting LoRA model...')
  model = get_lora_model(model)

  print('Fine-tuning model...')
  fine_tune_model(model, tokenizer, tokenized_dataset)