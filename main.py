from pathlib import Path

from sklearn.naive_bayes import MultinomialNB

from dataset.load import load_dataset
import dataset.preprocess as preprocess
import models.train as train
import models.test as test


BASE_DIR = Path(__file__).resolve().parent
dataset_path = BASE_DIR / "dataset" / "dataset.csv"
preprocessed_dataset_path = BASE_DIR / "dataset" / "preprocessed_dataset.csv"


def save_dataset(df, path):
    df.to_csv(path, index=False)


def naive_bayes(df):
    if df is None or df.empty:
        raise ValueError("Dataset vazio ou inválido.")

    df = preprocess.preprocess_naive_bayes(df)
    save_dataset(df, preprocessed_dataset_path)
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

    naive_bayes(df)
    print("Ending program...")


if __name__ == "__main__":
    main()