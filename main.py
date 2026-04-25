import numpy as np
import pandas as pd
from dataset.load_uci_dataset import load_uci_dataset
import model.model_choice as mc

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
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
load_uci_dataset(CSV_PATH, ucirepo_id=222)

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
# PREPROCESS + MODEL
# =========================

# GaussianNB    -> para variáveis numéricas (assume distribuição normal), como são dados bancários,
#                  é razoável assumir que as variáveis numéricas seguem uma distribuição normal (ou próxima disso). 
#                  Portanto, o GaussianNB é a escolha mais apropriada para este dataset.
# MultinomialNB -> para variáveis categóricas (usa contagem de frequências)
#                  Não é o ideal pra esse tipo de dataset e portanto não teve um desempenho tão bom.
#                  Gerou muitos falsos negativos 
#                  (Precisa ser executado com o MinMaxScaler, por conta de valores naturalmente negativos)

# StandardScaler -> sensível a outiliers (usa média e desvio padrão)
# RobustScaler   -> menos sensível a outliers (usa mediana e IQR)
# MinMaxScaler   -> escala para [0, 1], mas pode ser distorcido por outliers
model = mc.get_model(num_cols, cat_cols, model=mc.GaussianNB(), scaler=mc.RobustScaler())

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# Treino real
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

# CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

# =========================
# RESULTADOS
# =========================
print("=== RESULTADOS ===")
print(f"Naive Bayes acc:        {acc:.4f}")
print(f"Naive Bayes f1_macro:   {f1m:.4f}")
print(f"CV acc (10-fold):        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(y_test, y_pred, zero_division=0))