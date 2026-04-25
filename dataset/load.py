import kagglehub
from kagglehub import KaggleDatasetAdapter
from pathlib import Path
import pandas as pd

# Set the path to the file you'd like to load
_dataset_path = "student_mental_health_burnout.csv"

def load_dataset(dataset_local_path=None):
  # Load the latest version
  
  if(not Path(dataset_local_path).exists):
    df = kagglehub.load_dataset(
      KaggleDatasetAdapter.PANDAS,
      "sehaj1104/student-mental-health-and-burnout-dataset",
      _dataset_path,
      # Provide any additional arguments like 
      # sql_query or pandas_kwargs. See the 
      # documenation for more information:
      # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
    )
  else:
    df = pd.read_csv(dataset_local_path)
  
  return df