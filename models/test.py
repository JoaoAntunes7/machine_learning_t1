from sklearn.metrics import accuracy_score

def test_naive_bayes(mode, x_test, y_test):
    y_pred = mode.predict(x_test)

    acc = accuracy_score(y_test, y_pred)
    print("Accuracy:", acc)