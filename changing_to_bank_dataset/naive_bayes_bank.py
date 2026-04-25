import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

# =========================
# CONFIG
# =========================
CSV_PATH = "./bank_marketing.csv"  # arquivo UCI
TARGET_COL = "y"
DROP_COLS = ["duration"]                          # evita data leakage
TEST_SIZE = 0.2
RANDOM_STATE = 42

# =========================
# LOAD
# =========================  
df = pd.read_csv(CSV_PATH, sep=None, engine="python").drop_duplicates()

df.columns = df.columns.str.strip()

if TARGET_COL not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada. Colunas: {list(df.columns)}")

drop_existing = [c for c in DROP_COLS if c in df.columns and c != TARGET_COL]
X = df.drop(columns=[TARGET_COL] + drop_existing)
y = df[TARGET_COL].astype("string").str.strip()

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# =========================
# CHECKS
# =========================
print("=== CHECK ENUNCIADO ===")
print(f"Instâncias: {len(df)}")
print(f"Features usadas: {X.shape[1]}")
print(f"Numéricas: {len(num_cols)}")
print(f"Categóricas: {len(cat_cols)}")
print("Distribuição do target:")
print(y.value_counts(normalize=True).round(4))
print()

if len(df) < 100:
    raise ValueError("Dataset inválido: menos de 100 instâncias.")
if X.shape[1] < 5:
    raise ValueError("Dataset inválido: menos de 5 features.")
if len(num_cols) == 0 or len(cat_cols) == 0:
    raise ValueError("Dataset inválido: precisa ter variáveis numéricas e categóricas.")

# =========================
# PREPROCESS + MODEL
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), num_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
        ]), cat_cols),
    ]
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Baseline
dummy = DummyClassifier(strategy="most_frequent")
dummy.fit(X_train, y_train)
y_dummy = dummy.predict(X_test)
baseline_acc = accuracy_score(y_test, y_dummy)

# Treino real
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

# Sanity check (target embaralhado)
y_train_shuffled = y_train.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)
shuffle_model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", GaussianNB())
])
shuffle_model.fit(X_train.reset_index(drop=True), y_train_shuffled)
y_pred_shuffle = shuffle_model.predict(X_test)
shuffle_acc = accuracy_score(y_test, y_pred_shuffle)

# CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

# =========================
# RESULTADOS
# =========================
print("=== RESULTADOS ===")
print(f"Baseline (majoritária): {baseline_acc:.4f}")
print(f"Shuffle target acc:     {shuffle_acc:.4f}")
print(f"Naive Bayes acc:        {acc:.4f}")
print(f"Naive Bayes f1_macro:   {f1m:.4f}")
print(f"CV acc (5-fold):        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(y_test, y_pred, zero_division=0))

print("\n=== DIAGNÓSTICO ===")
if acc <= baseline_acc + 0.02 and acc <= shuffle_acc + 0.02:
    print("⚠️ Sinal fraco para NB neste dataset/target.")
else:
    print("✅ Há sinal preditivo acima de baseline/shuffle.")