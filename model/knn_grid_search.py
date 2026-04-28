from model import model_choice as mc
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import recall_score, make_scorer
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

def knn_grid_search(POS_LABEL, X_train, y_train, num_cols, cat_cols, cv):
    knn_base = mc.get_model(
        num_cols,
        cat_cols,
        model=mc.KNeighborsClassifier(),
        num_scaler=mc.RobustScaler(),
    )

    param_grid = {
        "classifier__n_neighbors": list(range(3, 32, 2)),
        "classifier__weights": ["uniform", "distance"],
        "classifier__p": [1, 2],
    }

    scoring = {
        "accuracy": "accuracy",
        "f1_macro": "f1_macro",
        "recall_yes": make_scorer(
            recall_score,
            average="binary",
            pos_label=POS_LABEL,
            zero_division=0,
        ),
    }

    knn_search = GridSearchCV(
        estimator=knn_base,
        param_grid=param_grid,
        scoring=scoring,
        refit="f1_macro",
        cv=cv,
        n_jobs=-1,          
        error_score="raise",
    )

    total_fits = len(ParameterGrid(param_grid)) * cv.get_n_splits(X_train, y_train)
    with tqdm_joblib(tqdm(total=total_fits, desc="GridSearchCV (KNN)")):
        knn_search.fit(X_train, y_train)
    
    return knn_search