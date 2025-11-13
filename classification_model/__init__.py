from data_loader import process_daic_data

if __name__ == "__main__":
  data_dir = "./daic_data/"
  processed_df = process_daic_data(data_dir)
  print("Processed DAIC data:")
  print(processed_df.head())

