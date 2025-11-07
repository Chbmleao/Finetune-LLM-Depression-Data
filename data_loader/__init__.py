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

def load_daic_data(data_dir="./daic_data/"):
  transcripts_dir = os.path.join(data_dir, "transcripts")
  labels_dir = os.path.join(data_dir, "labels")

  qa_df = get_questions_answers_df(transcripts_dir)
  

if __name__ == "__main__":
  load_daic_data()