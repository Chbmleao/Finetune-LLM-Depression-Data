import pandas as pd
import glob, os

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
  df = df[['participant_id', 'question', 'answer']]

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

def load_daic_data(data_dir="./daic_data/", should_create_csv=False):
  transcripts_dir = os.path.join(data_dir, "transcripts")
  labels_dir = os.path.join(data_dir, "labels")

  qa_df = get_questions_answers_df(transcripts_dir)
  qa_df = add_labels_to_df(qa_df, labels_dir)

  if should_create_csv:
    qa_df.to_csv("questions_and_answers.csv", index=False, encoding="utf-8-sig")

  return qa_df
  

if __name__ == "__main__":
  load_daic_data(should_create_csv=True)