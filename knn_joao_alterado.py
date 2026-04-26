import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

df = pd.read_csv("./dataset/bank_marketing.csv", sep=',')

print(f"\nDataset shape: {df.shape}")
print(f"First 5 records:\n{df.head()}")

# Informações dos dados
print(f"\nShape do dataset: {df.shape}")
print(f"\nTipos de dados:\n{df.dtypes}")
print(f"\nEstatísticas básicas:\n{df.describe()}")

# Vaviável alvo: predição de inscrição do cliente (yes/no)
target_col = 'y'

if target_col:
    print(f"\nVariável alvo: {target_col}")
    print(f"\nDistribuição da variável alvo:\n{df[target_col].value_counts()}")
    print(f"\nBalanceamento de classes: {df[target_col].value_counts(normalize=True)}")

# Cria cópia para pré-processamento
df_proc = df.copy()

# Remove linhas com valores ausentes na variável alvo
if target_col:
    df_proc = df_proc[df_proc[target_col].notna()]

print(f"Shape após limpeza: {df_proc.shape}")
print(f"\nAntes do tratamento de valores ausentes:\n{df_proc.isnull().sum()}")

# Preencher valores ausentes (numéricos com mediana, categóricos com moda)
df_proc.fillna(df_proc.median(numeric_only=True), inplace=True)  # Numéricos
for col in df_proc.select_dtypes(include=['object']).columns:
    if col != target_col:
        df_proc[col].fillna(df_proc[col].mode()[0] if not df_proc[col].mode().empty else 'Unknown', inplace=True)

print(f"\nDepois do tratamento de valores ausentes:\n{df_proc.isnull().sum()}")

categorical_cols = [col for col in df_proc.select_dtypes(include=['object']).columns if col != target_col]

# Codificação de variáveis categóricas
le_dict = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_proc[col] = le.fit_transform(df_proc[col].astype(str))
    le_dict[col] = le
    print(f"Coluna {col} codificada: {dict(zip(le.classes_, le.transform(le.classes_)))}")

# Codificação da variável alvo
le_target = LabelEncoder()
y = le_target.fit_transform(df_proc[target_col])
print(f"\nVariável alvo codificada: {dict(zip(le_target.classes_, le_target.transform(le_target.classes_)))}")

# Separação entre features e variável alvo
X = df_proc.drop(target_col, axis=1)

# Split do treino e teste (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Set de treinamento: {X_train.shape}")
print(f"Set de teste: {X_test.shape}")
print(f"\nDistribuição de classe no set de treinamento:\n{pd.Series(y_train).value_counts()}")
print(f"\nDistribuição de classe no set de teste:\n{pd.Series(y_test).value_counts()}")

# Normaliza as features com StandardScaler
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Busca o melhor k usando validação cruzada
param_grid = {'n_neighbors': range(1, 31)}
knn = KNeighborsClassifier()
grid_search = GridSearchCV(knn, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_scaled, y_train)

print(f"Melhor k: {grid_search.best_params_['n_neighbors']}")
print(f"Melhor score de cross-validation: {grid_search.best_score_:.4f}")

# Treina o modelo KNN com o melhor k 
best_k = grid_search.best_params_['n_neighbors']
knn_model = KNeighborsClassifier(n_neighbors=best_k)
knn_model.fit(X_train_scaled, y_train)

# Avalia o modelo no treino e no teste
y_pred_train = knn_model.predict(X_train_scaled)
y_pred_test = knn_model.predict(X_test_scaled)

print(f"Modelo KNN treinado com k = {best_k}")
print(f"Acurácia no treinamento: {accuracy_score(y_train, y_pred_train):.4f}")
print(f"Acurácia no teste: {accuracy_score(y_test, y_pred_test):.4f}")

# Cálculo de métricas de avaliação
accuracy = accuracy_score(y_test, y_pred_test)
precision = precision_score(y_test, y_pred_test, average='weighted')
recall = recall_score(y_test, y_pred_test, average='weighted')
f1 = f1_score(y_test, y_pred_test, average='weighted')

print(f"\nMétricas de Avaliação no Teste:")
print(f"Acurácia:  {accuracy:.4f}")
print(f"Precisão: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1-Score:  {f1:.4f}")

# Matriz de confusão
print(f"\nMatriz de Confusão:\n{confusion_matrix(y_test, y_pred_test)}")