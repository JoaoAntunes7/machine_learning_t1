import numpy as np
import pandas as pd
from dataset.load_uci_dataset import load_uci_dataset
import model.model_choice as mc

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
from sklearn.tree import plot_tree, export_text
import matplotlib.pyplot as plt

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

# =========================
# ÁRVORE DE DECISÃO
# =========================
# max_depth=5 limita a profundidade para evitar overfitting e manter legibilidade.
# criterion="gini" mede a impureza dos nós (padrão e eficiente).
dt_model = mc.get_model(num_cols, cat_cols, model=mc.DecisionTreeClassifier(max_depth=5, criterion="gini", random_state=RANDOM_STATE))

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

importances = dt_model.named_steps["classifier"].feature_importances_
top_idx     = np.argsort(importances)[::-1][:10]  # top 10

print("\nTop 10 features mais importantes (Árvore de Decisão):")
for i in top_idx:
    print(f"  {all_feature_names[i]:<35} {importances[i]:.4f}")

# Visualização da árvore (primeiros 3 níveis para não poluir)
fig, ax = plt.subplots(figsize=(20, 8))
plot_tree(
    dt_model.named_steps["classifier"],
    feature_names=all_feature_names,
    class_names=dt_model.classes_,
    filled=True,
    max_depth=3,
    ax=ax,
    fontsize=8,
)
plt.title("Árvore de Decisão (primeiros 3 níveis)")
plt.tight_layout()
plt.savefig("decision_tree.png", dpi=150)
print("\nÁrvore salva em decision_tree.png")

# =========================
# KNN
# =========================
# KNN classifica um ponto novo pela votação dos K vizinhos mais próximos.
# DIFERENTE da Árvore de Decisão: KNN é baseado em distância, então
# a normalização das features é OBRIGATÓRIA (já feita pelo RobustScaler no pipeline).
# n_neighbors=11: ímpar para evitar empates em classificação binária.
from sklearn.inspection import permutation_importance
from lime.lime_tabular import LimeTabularExplainer

knn_model = mc.get_model(num_cols, cat_cols, model=mc.KNeighborsClassifier(n_neighbors=11), scaler=mc.RobustScaler())

knn_model.fit(X_train, y_train)
knn_pred = knn_model.predict(X_test)
knn_acc  = accuracy_score(y_test, knn_pred)
knn_f1   = f1_score(y_test, knn_pred, average="macro")

knn_cv = cross_val_score(knn_model, X, y, scoring="accuracy", cv=cv, n_jobs=-1)

print("\n=== KNN ===")
print(f"Acurácia:         {knn_acc:.4f}")
print(f"F1-macro:         {knn_f1:.4f}")
print(f"CV acc (10-fold): {knn_cv.mean():.4f} ± {knn_cv.std():.4f}")
print("\nMatriz de confusão:")
print(confusion_matrix(y_test, knn_pred))
print("\nRelatório:")
print(classification_report(y_test, knn_pred, zero_division=0))

# --- Permutation Importance (interpretabilidade global) ---
# Embaralha cada feature e mede quanto a acurácia cai.
# Se cair muito: a feature é importante. Se não mudar: o modelo não depende dela.
perm = permutation_importance(knn_model, X_test, y_test, n_repeats=10, random_state=RANDOM_STATE, n_jobs=-1)
perm_top = np.argsort(perm.importances_mean)[::-1][:10]

print("\nTop 10 features por Permutation Importance (KNN):")
for i in perm_top:
    print(f"  {X.columns[i]:<35} {perm.importances_mean[i]:.4f} ± {perm.importances_std[i]:.4f}")

fig, ax = plt.subplots(figsize=(10, 5))
ax.barh(X.columns[perm_top][::-1], perm.importances_mean[perm_top][::-1])
ax.set_xlabel("Queda na acurácia ao embaralhar a feature")
ax.set_title("Permutation Importance — KNN")
plt.tight_layout()
plt.savefig("permutation_importance_knn.png", dpi=150)
print("Gráfico salvo em permutation_importance_knn.png")

# --- LIME (interpretabilidade local) ---
# LIME explica UMA predição individual: cria variações do exemplo e treina
# um modelo linear simples ao redor dele para aproximar o comportamento do KNN.
X_train_transformed = knn_model.named_steps["preprocessor"].transform(X_train)

lime_explainer = LimeTabularExplainer(
    training_data=X_train_transformed,
    feature_names=all_feature_names,   # nomes das features após o pipeline
    class_names=knn_model.classes_,
    mode="classification",
    random_state=RANDOM_STATE,
)

sample_idx = 0  # explica a primeira amostra do conjunto de teste
sample = knn_model.named_steps["preprocessor"].transform(X_test.iloc[[sample_idx]])
explanation = lime_explainer.explain_instance(sample[0], knn_model.named_steps["classifier"].predict_proba)

print(f"\nLIME — explicação para a amostra {sample_idx} (real: {y_test.iloc[sample_idx]}, previsto: {knn_pred[sample_idx]}):")
for feat, weight in explanation.as_list():
    print(f"  {feat:<50} {weight:+.4f}")

explanation.save_to_file("lime_explanation.html")
print("Explicação LIME salva em lime_explanation.html")

# =========================
# COMPARAÇÃO FINAL
# =========================
print("\n" + "="*60)
print("COMPARAÇÃO FINAL DOS MODELOS")
print("="*60)
print(f"{'Modelo':<25} {'Acurácia':<12} {'F1-macro':<12} {'Recall YES'}")
print("-"*60)

from sklearn.metrics import recall_score
print(f"{'Naive Bayes':<25} {acc:<12.4f} {f1m:<12.4f} {recall_score(y_test, y_pred, pos_label='yes'):.4f}")
print(f"{'Árvore de Decisão':<25} {dt_acc:<12.4f} {dt_f1:<12.4f} {recall_score(y_test, dt_pred, pos_label='yes'):.4f}")
print(f"{'KNN (k=11)':<25} {knn_acc:<12.4f} {knn_f1:<12.4f} {recall_score(y_test, knn_pred, pos_label='yes'):.4f}")
print("="*60)