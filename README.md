# T1 — Interpretabilidade de Modelos de Aprendizado de Máquina

## Dataset

**Bank Marketing** (UCI Machine Learning Repository, ID 222)  
45.211 instâncias · 16 features · alvo binário: `y` (`yes`/`no`)

Problema: prever se um cliente irá subscrever um depósito a prazo após campanha de telemarketing.

A coluna `duration` foi removida por **data leakage** (só é conhecida após a ligação).

---

## Pré-processamento

- Remoção de duplicatas
- Tratamento de faltantes:
  - Numéricas: mediana
  - Categóricas: moda
- `LabelEncoder` no alvo (`no=0`, `yes=1`)
- Split estratificado (`test_size=0.2`, `random_state=42`)
- Validação cruzada com `StratifiedKFold(n_splits=10, shuffle=True, random_state=42)`

---

## Modelos e configuração atual

### Naive Bayes
- `GaussianNB`
- Pipeline com `RobustScaler` para variáveis numéricas

### Árvore de Decisão
- `DecisionTreeClassifier`
- Hiperparâmetros atuais:
  - `ccp_alpha=0.001`
  - `class_weight="balanced"`
  - `max_depth=12`
  - `criterion="gini"`
  - `max_features="log2"`
  - `min_samples_leaf=1`
  - `min_samples_split=2`
  - `random_state=42`
- Pipeline com `RobustScaler` para numéricas

### KNN
- `KNeighborsClassifier(n_neighbors=3, p=2, weights="distance")`
- Pipeline:
  - Numéricas: `StandardScaler`
  - Categóricas: `OneHotEncoder(handle_unknown="ignore")`

> `KNN_GRID_SEARCH` e `DT_GRID_SEARCH` estão desabilitados por padrão no `main.py`.

---

## Resultados da última execução (`main.py`)

### NAIVE BAYES

- **Accuracy:** `0.8519`
- **F1-macro:** `0.6773`
- **CV acc (10-fold):** `0.8499 ± 0.0031`
- **Recall YES:** `0.4972`

Matriz de confusão:
- `[[7178, 807], [532, 526]]`

Relatório (resumo):
- Classe `0`: precision `0.93`, recall `0.90`, f1 `0.91`
- Classe `1`: precision `0.39`, recall `0.50`, f1 `0.44`

---

### ÁRVORE DE DECISÃO

- **Accuracy:** `0.8286`
- **F1-macro:** `0.6876`
- **CV acc (10-fold):** `0.7707 ± 0.0274`
- **Recall YES:** `0.6701`

Matriz de confusão:
- `[[6784, 1201], [349, 709]]`

Relatório (resumo):
- Classe `0`: precision `0.95`, recall `0.85`, f1 `0.90`
- Classe `1`: precision `0.37`, recall `0.67`, f1 `0.48`

---

### KNN

- **Accuracy:** `0.8921`
- **F1-macro:** `0.6894`
- **CV acc (10-fold):** `0.8917 ± 0.0038`
- **Recall YES:** `0.3601`

Matriz de confusão:
- `[[7686, 299], [677, 381]]`

Relatório (resumo):
- Classe `no`: precision `0.92`, recall `0.96`, f1 `0.94`
- Classe `yes`: precision `0.56`, recall `0.36`, f1 `0.44`

---

## Comparação final

| Modelo | Acurácia | F1-macro | Recall YES |
|---|---:|---:|---:|
| Naive Bayes | 0.8519 | 0.6773 | 0.4972 |
| Árvore de Decisão | 0.8286 | 0.6876 | **0.6701** |
| KNN (k=3) | **0.8921** | **0.6894** | 0.3601 |

### Leitura rápida

- **Melhor acurácia e F1-macro:** KNN (`k=3`)
- **Melhor recall da classe positiva (`yes`):** Árvore de Decisão
- **Naive Bayes:** desempenho equilibrado, com recall `yes` intermediário

---

## Como executar

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy matplotlib seaborn ucimlrepo lime tqdm tqdm-joblib shap
python main.py
```