from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler, MinMaxScaler, StandardScaler, LabelEncoder
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier

def get_model(num_cols, cat_cols, model=GaussianNB(), num_scaler=RobustScaler(), cat_scaler=OneHotEncoder()):
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", num_scaler)
            ]), num_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("scaler", cat_scaler)
            ]), cat_cols),
        ]
    )
    
    ret_model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    return ret_model