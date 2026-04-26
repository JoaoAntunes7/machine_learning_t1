import numpy as np
import pandas as pd
from dataset.load_uci_dataset import load_uci_dataset
import model.model_choice as mc

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV, ParameterGrid
from sklearn.preprocessing import LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    recall_score,
    make_scorer,
)
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from tqdm_joblib import tqdm_joblib

# =========================
# CONFIG
# =========================
CSV_PATH = "./dataset/bank_marketing.csv"  # arquivo UCI
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
nb_model = mc.get_model(num_cols, cat_cols, model=mc.GaussianNB(), scaler=mc.RobustScaler())

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
# max_depth=5 limita a profundidade para evitar overfitting e manter legibilidade.
# criterion="gini" mede a impureza dos nós (padrão e eficiente).
dt_model = mc.get_model(num_cols, cat_cols, model=mc.DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE), scaler=mc.RobustScaler())

dt_model.fit(X_train, y_train)
dt_pred = dt_model.predict(X_test)
dt_acc  = accuracy_score(y_test, dt_pred)
dt_f1   = f1_score(y_test, dt_pred, average="macro")

dt_cv = cross_val_score(dt_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("\n=== ÁRVORE DE DECISÃO ===")
print(f"Acurácia:         {dt_acc:.4f}")
print(f"F1-macro:         {dt_f1:.4f}")
print(f"CV acc (10-fold): {dt_cv.mean():.4f} ± {dt_cv.std():.4f}")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, dt_pred))
print("\nRelatório:")
print(classification_report(y_test, dt_pred, zero_division=0))

# Recupera os nomes das features após o pipeline de pré-processamento
preprocessor      = dt_model.named_steps["preprocessor"]
ohe_feature_names = preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(cat_cols)
all_feature_names = np.array(num_cols + list(ohe_feature_names))

fig, ax = plt.subplots(figsize=(20, 8))

# classes do modelo (0/1) -> rótulos originais ("no"/"yes")
tree_class_names = [str(c) for c in le.inverse_transform(dt_model.classes_.astype(int))]

plot_tree(
    dt_model.named_steps["classifier"],
    feature_names=all_feature_names,
    class_names=tree_class_names,  # antes: dt_model.classes_
    filled=True,
    max_depth=5,
    ax=ax,
    fontsize=8,
)
plt.title("Árvore de Decisão (primeiros 5 níveis)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
print("\nÁrvore salva em decision_tree.png")

# =========================
# KNN
# =========================

'''
knn_base = mc.get_model(
    num_cols,
    cat_cols,
    model=mc.KNeighborsClassifier(),
    scaler=mc.RobustScaler(),
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
    n_jobs=-1,           # paralelo + barra de progresso
    error_score="raise", # mostra erro real sem esconder
)

total_fits = len(ParameterGrid(param_grid)) * cv.get_n_splits(X_train, y_train)
with tqdm_joblib(tqdm(total=total_fits, desc="GridSearchCV (KNN)")):
    knn_search.fit(X_train, y_train)

knn_model = knn_search.best_estimator_
'''

knn_model = mc.get_model(
    num_cols,
    cat_cols,
    model=mc.KNeighborsClassifier(n_neighbors=3, weights="distance", p=1),
    scaler=mc.RobustScaler(),
)

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc  = accuracy_score(y_test, knn_pred)
knn_f1   = f1_score(y_test, knn_pred, average="macro")
knn_rec  = recall_score(y_test, knn_pred, pos_label=POS_LABEL)
knn_cv   = cross_val_score(knn_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

'''
cv_res = pd.DataFrame(knn_search.cv_results_)
best_acc_row = cv_res.loc[cv_res["mean_test_accuracy"].idxmax()]
best_f1_row  = cv_res.loc[cv_res["mean_test_f1_macro"].idxmax()]
best_rec_row = cv_res.loc[cv_res["mean_test_recall_yes"].idxmax()]
'''

print("\n=== KNN ===")
#print(f"Melhor (refit=f1_macro): {knn_search.best_params_}")
print(f"Acurácia:         {knn_acc:.4f}")
print(f"F1-macro:         {knn_f1:.4f}")
print(f"Recall YES:       {knn_rec:.4f}")
print(f"CV acc (10-fold): {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")

'''
print("\nMelhor por métrica (CV):")
print("accuracy  :", best_acc_row["params"])
print("f1_macro  :", best_f1_row["params"])
print("recall_yes:", best_rec_row["params"])
'''

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
print(f"{'KNN (k=3)':<25} {knn_acc:<12.4f} {knn_f1:<12.4f} {recall_score(y_test, knn_pred, pos_label=POS_LABEL):.4f}")
print("="*60)