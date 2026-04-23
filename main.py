from dataset.load import load_dataset
import dataset.view as view

def save_dataset(df, path="output.csv"):
    # Save the dataset to a CSV file
    df.to_csv(path, index=False)

def main():

    print("Loading dataset...")        
    df = load_dataset()    
    print("Dataset loaded successfully!")

    # save_dataset(df, path="./dataset/dataset.csv") to save dataset as localfile
    # print("Dataset saved successfully!")

    print("Ending program...")
    
    return



if __name__ == "__main__":
    main()