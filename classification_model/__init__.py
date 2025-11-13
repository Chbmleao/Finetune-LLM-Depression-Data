from data_loader import process_daic_data
from model_trainer import train_model 

if __name__ == "__main__":
  data_dir = "./daic_data/"
  df = process_daic_data(data_dir)
  print(df.head())
 
  train_model(df)
