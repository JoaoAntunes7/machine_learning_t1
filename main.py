from sklearn.naive_bayes import MultinomialNB
from dataset.load import load_dataset
import dataset.view as view
import dataset.preprocess as preprocess
import models.train as train
import models.test as test

dataset_path = "./dataset/dataset.csv"  # Path to the dataset file
preprocessed_dataset_path = "./dataset/preprocessed_dataset.csv"  # Path to save the preprocessed dataset

def save_dataset(df, path="output.csv"):
    # Save the dataset to a CSV file
    df.to_csv(path, index=False)

def naive_bayes(df):
    
    df = preprocess.preprocess_naive_bayes(df)
        
    save_dataset(df, path=preprocessed_dataset_path)
    print("Preprocessed dataset saved successfully!")
    
    X_train, X_test, y_train, y_test = preprocess.dataset_split(df)
    print("Dataset split into training and testing sets successfully!")
    model = MultinomialNB()
    model = train.train_naive_bayes(model, X_train, y_train)
    print("Naive Bayes model trained successfully!")
    
    test.test_naive_bayes(model, X_test, y_test)
    print("Naive Bayes model tested successfully!")
    

def main():

    print("Loading dataset...")        
    df = load_dataset(dataset_path)    
    print("Dataset loaded successfully!")

    #save_dataset(df, path="./dataset/dataset.csv")  # to save dataset as localfile
    #print("Dataset saved successfully!")
    
    naive_bayes(df)
    

    print("Ending program...")        
    return



if __name__ == "__main__":
    main()