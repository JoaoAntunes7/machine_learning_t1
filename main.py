import numpy as np
import pandas as pd
from interpretability.interpretability import explain_with_lime

from dataset.load_uci_dataset import load_uci_dataset
from model import dt_grid_search
from model.knn_grid_search import knn_grid_search
import model.model_choice as mc

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
)

# =========================
# CONFIG
# =========================
CSV_PATH = "./dataset/bank_marketing.csv"  # arquivo UCI
TARGET_COL = "y"
TEST_SIZE = 0.2
RANDOM_STATE = 42
KNN_GRID_SEARCH = False  # se True, executa GridSearchCV para KNN (pode ser demorado)
DT_GRID_SEARCH = False   # se True, executa GridSearchCV para Árvore de Decisão (muito demorado)

# =========================
# LOAD
# =========================
load_uci_dataset(CSV_PATH, ucirepo_id=222)

df = pd.read_csv(CSV_PATH, sep=None, engine="python").drop_duplicates()

df.columns = df.columns.str.strip()

if TARGET_COL not in df.columns:
    raise ValueError(f"Coluna alvo '{TARGET_COL}' não encontrada. Colunas: {list(df.columns)}")
else:
    df = df[df[TARGET_COL].notna()] 

# print(f"Shape após limpeza: {df.shape}")
# print(f"\nAntes do tratamento de valores ausentes:\n{df.isnull().sum()}")

# Preencher valores ausentes (numéricos com mediana, categóricos com moda)
df = df.fillna(df.median(numeric_only=True))

for col in df.select_dtypes(include=["object", "string"]).columns:
    if col != TARGET_COL:
        mode_val = df[col].mode().iloc[0] if not df[col].mode().empty else "Unknown"
        df[col] = df[col].fillna(mode_val)

# print(f"\nDepois do tratamento de valores ausentes:\n{df.isnull().sum()}")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL].astype(str).str.strip()

# codifica alvo: ex. no->0, yes->1
le = LabelEncoder()
y = le.fit_transform(y)

# opcional: garante o índice da classe positiva "yes"
POS_LABEL = int(le.transform(["yes"])[0])

num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()

# =========================
# SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

# =========================
# PREPROCESS + NB_MODEL
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
nb_model = mc.get_model(num_cols, cat_cols, model=mc.GaussianNB(), num_scaler=mc.RobustScaler())

# Treino real
nb_model.fit(X_train, y_train)
y_pred = nb_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
f1m = f1_score(y_test, y_pred, average="macro")

# CV
cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=RANDOM_STATE)
cv_scores = cross_val_score(nb_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

# =========================
# RESULTADOS
# =========================
print("=== NAIVE BAYES ===")
print(f"Naive Bayes acc:        {acc:.4f}")
print(f"Naive Bayes f1_macro:   {f1m:.4f}")
print(f"CV acc (10-fold):        {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, y_pred))

print("\nRelatório:")
print(classification_report(y_test, y_pred, zero_division=0))

# =========================
# ÁRVORE DE DECISÃO
# =========================

if(DT_GRID_SEARCH):
    dt_model = dt_grid_search(POS_LABEL, RANDOM_STATE, X_train, y_train, num_cols, cat_cols, cv).best_estimator_
else:
    dt_model = mc.get_model(num_cols, cat_cols, model=mc.DecisionTreeClassifier(ccp_alpha=0.001, class_weight="balanced", max_depth=12, criterion="gini",max_features="log2", min_samples_leaf=1, min_samples_split=2, random_state=RANDOM_STATE), num_scaler=mc.RobustScaler())

dt_model.fit(X_train, y_train)

dt_pred = dt_model.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_f1   = f1_score(y_test, dt_pred, average="macro")
dt_rec  = recall_score(y_test, dt_pred, pos_label=POS_LABEL)

dt_cv   = cross_val_score(dt_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("\n=== ÁRVORE DE DECISÃO ===")
print(f"Acurácia:         {dt_acc:.4f}")
print(f"F1-macro:         {dt_f1:.4f}")
print(f"CV acc (10-fold): {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")
print(f"Recall:           {dt_rec:.4f}")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, dt_pred))
print("\nRelatório:")
print(classification_report(y_test, dt_pred, zero_division=0))

# =========================
# KNN
# =========================

if(KNN_GRID_SEARCH):
    knn_model = knn_grid_search(POS_LABEL, X_train, y_train, num_cols, cat_cols, cv).best_estimator_
else: 
    knn_model = mc.get_model(
        num_cols,
        cat_cols,
        model=mc.KNeighborsClassifier(n_neighbors=3, p=2, weights="distance"),
        num_scaler=mc.StandardScaler(),
        cat_scaler=mc.OneHotEncoder(handle_unknown="ignore"),
    )

#n_neighbors = knn_model.n_neighbors
n_neighbors = knn_model.get_params()["classifier__n_neighbors"]

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc  = accuracy_score(y_test, knn_pred)
knn_f1   = f1_score(y_test, knn_pred, average="macro")
knn_rec  = recall_score(y_test, knn_pred, pos_label=POS_LABEL)
knn_cv   = cross_val_score(knn_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("\n=== KNN ===")
print(f"Acurácia:         {knn_acc:.4f}")
print(f"F1-macro:         {knn_f1:.4f}")
print(f"Recall YES:       {knn_rec:.4f}")
print(f"CV acc (10-fold): {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")

print("\nMatriz de confusão:")
print(confusion_matrix(y_test, knn_pred))
print("\nRelatório:")
print(classification_report(y_test, knn_pred, zero_division=0, target_names=le.classes_))

# =========================
# COMPARAÇÃO FINAL
# =========================
print("\n" + "="*60)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("="*60)
print(f"{'Modelo':<25} {'Acurácia':<12} {'F1-macro':<12} {'Recall YES'}")
print("-"*60)

print(f"{'Naive Bayes':<25} {acc:<12.4f} {f1m:<12.4f} {recall_score(y_test, y_pred, pos_label=POS_LABEL):.4f}")
print(f"{'Árvore de Decisão':<25} {dt_acc:<12.4f} {dt_f1:<12.4f} {recall_score(y_test, dt_pred, pos_label=POS_LABEL):.4f}")
print(f'{f"KNN (k={n_neighbors})":<25} {knn_acc:<12.4f} {knn_f1:<12.4f} {recall_score(y_test, knn_pred, pos_label=POS_LABEL):.4f}')
print("="*60)

# =========================
# INTERPRETABILIDADE (LIME)
# =========================
print("\n=== INTERPRETABILIDADE (LIME) ===")
explain_with_lime(
    fitted_pipeline=nb_model,
    X_train_df=X_train,
    X_test_df=X_test,
    y_test_arr=y_test,
    y_pred_arr=y_pred,
    class_names=le.classes_,
    model_name="Naive Bayes",
    random_state=RANDOM_STATE,
)

explain_with_lime(
    fitted_pipeline=dt_model,
    X_train_df=X_train,
    X_test_df=X_test,
    y_test_arr=y_test,
    y_pred_arr=dt_pred,
    class_names=le.classes_,
    model_name="Arvore de Decisao",
    random_state=RANDOM_STATE,
)

explain_with_lime(
    fitted_pipeline=knn_model,
    X_train_df=X_train,
    X_test_df=X_test,
    y_test_arr=y_test,
    y_pred_arr=knn_pred,
    class_names=le.classes_,
    model_name="KNN",
    random_state=RANDOM_STATE,
)
