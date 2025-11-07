import pandas as pd
import glob, os
import torch
from torch.utils.data import Dataset

def get_questions_answers_df(transcripts_dir):
  transcripts_files = glob.glob(os.path.join(transcripts_dir, "*.csv"))

  # Load and concatenate all transcript files
  df = pd.concat(
    (
      pd.read_csv(file, sep="\t", encoding="utf-8-sig").assign(source=os.path.basename(file))
      for file in transcripts_files
    ),
    ignore_index=True
  )

  # Create block_id to identify contiguous speaker segments
  df['block_id'] = (df['speaker'] != df['speaker'].shift(1)).cumsum()

  # Aggregate by source and block_id to merge contiguous segments by the same speaker
  df = df.groupby(['source', 'block_id']).agg(
    speaker=('speaker', 'first'),
    start_time=('start_time', 'min'),
    stop_time=('stop_time', 'max'),
    value=('value', lambda x: ' '.join(x.astype(str)))
  )

  # Sort by participant and time
  df = df.sort_values(by=['source', 'start_time']).reset_index()

  # Add previous speaker and value columns only if the previous source is the same
  df['prev_speaker'] = df.groupby('source')['speaker'].shift(1)
  df['prev_value'] = df.groupby('source')['value'].shift(1)

  is_answer = (
    (df['speaker'] == 'Participant') &
    (df['prev_speaker'] == 'Ellie') &
    (df['source'] == df['source'].shift(1))
  )

  df = df[is_answer].copy()
  df = df.rename(columns={
    'prev_value': 'question', # The previous Ellie utterance is the question
    'value': 'answer',            # The current Participant utterance is the answer
  })

  df['participant_id'] = df['source'].str.split("_").str[0].astype(int)
  df = df[['participant_id', 'question', 'answer', 'start_time']]

  return df

def add_labels_to_df(qa_df, labels_dir):
  splits = ['train', 'dev', 'test']

  all_labels_df = pd.DataFrame()
  for split in splits:
    split_labels_df = pd.read_csv(os.path.join(labels_dir, f"{split}.csv"))
    split_labels_df = split_labels_df.rename(columns={
      "Participant_ID": "participant_id",
      "PHQ8_Binary": "depression_label",
      "PHQ8_Score": "depression_severity",
      "PHQ_Binary": "depression_label",
      "PHQ_Score": "depression_severity",
    })
    split_labels_df = split_labels_df[["participant_id", "depression_label", "depression_severity"]]
    split_labels_df["split"] = split
    all_labels_df = pd.concat([all_labels_df, split_labels_df], ignore_index=True)

  merged_df = pd.merge(qa_df, all_labels_df, on="participant_id", how="left")
  return merged_df

def format_input(df, row, n_context=3):
  past_pairs = df[
    (df['participant_id'] == row['participant_id']) &
    (df.index < row.name)
  ].tail(n_context)

  # Build context from past Q&A pairs
  context_lines = []
  for _, past_row in past_pairs.iterrows():
    q = str(past_row.get("question", "")).strip()
    a = str(past_row.get("answer", "")).strip()
    context_lines.append(f"Q: {q}\nA: {a}")
  context = "[START]\n" + "\n".join(context_lines) if context_lines else "[START]\n"

  # Build instruction with PHQ score information
  instruction = (
    "### Instruction:\n"
    "You are analyzing a therapeutic interview between a virtual interviewer (Ellie) and a participant.\n"
    "The participant has a PHQ-8 score ranging from 0 (no depression) to 24 (severe depression). "
    f"This participant’s score is {row['depression_severity']}. "
    "Scores of 10 or higher are typically considered indicative of depression.\n"
    "Given the participant’s previous responses and their PHQ score, "
    "predict how they might answer the next question in a coherent and realistic way."
    "Use natural, casual language. Avoid overly formal styles. "
    "Tolerate some irregularities (omissions, repetitions, filler words) given the conversational context.\n\n"
  )

  # Build input text with context and current question
  question = str(row.get("question", "")).strip()
  input_text = f"### Input:\n{context}\nQ: {question}\nA:"

  return instruction + input_text

class InstructionDataset(Dataset):
  def __init__(self, dataframe):
    dataframe = dataframe.sort_values(by=['participant_id', 'start_time']).reset_index(drop=True)
    self.dataframe = dataframe

    self.texts = []
    for _, row in self.dataframe.iterrows():
      instruction_plus_input = format_input(dataframe, row)
      response_text = f"\n\n### Response:\n{str(row.get('answer', '')).strip()} [END]"
      full_text = instruction_plus_input + response_text
      self.texts.append(full_text)

  def __getitem__(self, idx):
    return self.texts[idx]

  def __len__(self):
    return len(self.texts)

def load_daic_data(data_dir="./daic_data/", should_create_csv=False):
  transcripts_dir = os.path.join(data_dir, "transcripts")
  labels_dir = os.path.join(data_dir, "labels")

  qa_df = get_questions_answers_df(transcripts_dir)
  qa_df = add_labels_to_df(qa_df, labels_dir)

  if should_create_csv:
    qa_df.to_csv("questions_and_answers.csv", index=False, encoding="utf-8-sig")

  instruction_dataset = InstructionDataset(qa_df)
  print(f"Loaded DAIC dataset with {len(instruction_dataset)} samples.")
  print("Example sample:")
  print(instruction_dataset[99])

  return instruction_dataset
  

if __name__ == "__main__":
  load_daic_data(should_create_csv=False)