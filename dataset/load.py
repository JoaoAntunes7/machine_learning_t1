import kagglehub
from kagglehub import KaggleDatasetAdapter

# Set the path to the file you'd like to load
_dataset_path = "student_mental_health_burnout.csv"

def load_dataset():
  # Load the latest version
  df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "sehaj1104/student-mental-health-and-burnout-dataset",
    _dataset_path,
    # Provide any additional arguments like 
    # sql_query or pandas_kwargs. See the 
    # documenation for more information:
    # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
  )
  
  return df