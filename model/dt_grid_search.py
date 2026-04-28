from model import model_choice as mc
from sklearn.model_selection import GridSearchCV, ParameterGrid
from sklearn.metrics import recall_score, make_scorer
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

def dt_grid_search(POS_LABEL, RANDOM_STATE, X_train, y_train, num_cols, cat_cols, cv):
    dt_base = mc.get_model(
        num_cols,
        cat_cols,
        model=mc.DecisionTreeClassifier(random_state=RANDOM_STATE),
        num_scaler=mc.RobustScaler(),
    )

    dt_param_grid = {
        "classifier__criterion": ["gini", "entropy"],
        "classifier__max_depth": [3, 5, 8, 12, None],
        "classifier__min_samples_split": [2, 10, 30],
        "classifier__min_samples_leaf": [1, 5, 10],
        "classifier__max_features": ["sqrt", "log2"],
        "classifier__class_weight": ["balanced"],
        "classifier__ccp_alpha": [0.0, 1e-4, 1e-3],
    }

    recall_yes_scorer = make_scorer(recall_score, pos_label=POS_LABEL)

    dt_grid = GridSearchCV(
        estimator=dt_base,
        param_grid=dt_param_grid,
        scoring={
            "accuracy": "accuracy",
            "f1_macro": "f1_macro",
            "recall_yes": recall_yes_scorer,
        },
        refit="f1_macro", 
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )

    total_fits = len(ParameterGrid(dt_param_grid)) * cv.get_n_splits(X_train, y_train)
    with tqdm_joblib(tqdm(total=total_fits, desc="GridSearchCV (KNN)")):
        dt_grid.fit(X_train, y_train)
        
    return dt_grid