import pandas as pd
import os

def process_daic_data(data_dir):
  transcripts_dir = os.path.join(data_dir, "transcripts")
  labels_dir = os.path.join(data_dir, "labels")

  df = pd.DataFrame()

  for file in os.listdir(labels_dir):
    if not file.endswith(".csv"):
      continue

    split_name = file.replace(".csv", "")
    split_df = pd.read_csv(os.path.join(labels_dir, file))
    split_df = split_df.rename(columns={
      "PHQ_Binary": "depression_label",
      "PHQ_Score": "depression_severity",
      "PHQ8_Binary": "depression_label",
      "PHQ8_Score": "depression_severity",
      "Participant_ID": "participant_id",
    })

    transcripts_df = create_dataframe(split_df, transcripts_dir)
    transcripts_df["split"] = split_name

    df = pd.concat([df, transcripts_df], ignore_index=True)

  return df

def create_dataframe(split_df, transcripts_dir):
  df = {"text": [], "depression_label": []}

  for _, row in split_df.iterrows():
    participant_id = str(int(float(row.participant_id)))
    depression_label = int(row.depression_label)

    participant_text = ""
    transcript_file = os.path.join(transcripts_dir, f"{participant_id}_TRANSCRIPT.csv")
    if not os.path.exists(transcript_file):
      print(f"Transcript file not found for participant {participant_id}")
      continue

    transcripts = pd.read_csv(transcript_file, sep="\t")
    participant_transcripts = transcripts[transcripts['speaker'] == 'Participant']

    for _, transcript_row in participant_transcripts.iterrows():
      participant_text += str(transcript_row.value) + " "

    df["text"].append(participant_text.strip())
    df["depression_label"].append(depression_label)

  return pd.DataFrame(df)