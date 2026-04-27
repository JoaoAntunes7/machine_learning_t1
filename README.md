# T1 — Interpretabilidade de Modelos de Aprendizado de Máquina

## Dataset

**Bank Marketing** (UCI Machine Learning Repository, ID 222)  
45.211 instâncias · 16 features · alvo binário: `y` (`yes`/`no`)

Problema: prever se um cliente irá subscrever um depósito a prazo após campanha de telemarketing.

---

## Pré-processamento

- Remoção de duplicatas
- Tratamento de faltantes:
  - Numéricas: mediana, robusta a outliers.
  - Categóricas: moda, mantém a categoria mais frequente.
- `LabelEncoder` no alvo (`no=0`, `yes=1`), necessário para os classificadores sklearn.
- Split estratificado (80/20) para preservar a proporção da classe minoritária.
- Validação cruzada com `StratifiedKFold(k=10)` para uma estimativa mais confiável do modelo real.

---

## Modelos e configuração atual

### Naive Bayes
- Modelo `GaussianNB`
  - `GaussianNB` assume distribuição normal para variáveis numéricas, sendo mais recomendado para dados bancários.
  - `MultinomialNB` foi descartado por gerar muitos falsos negativos neste dataset.
- Pipeline com `RobustScaler` para variáveis numéricas.
  - `RobustScaler` reduz o impacto de outliers sem distorcer a distribuição.

### Árvore de Decisão
- Modelo `DecisionTreeClassifier`
  - `class_weight="balanced"` compensa o desbalanceamento da classe `yes`.
  - `ccp_alpha=0.001` aplica poda para reduzir overfitting.
  - `refit="f1_macro"` no GridSearch prioriza equilíbrio entre as classes. 
- Pipeline com `RobustScaler` para numéricas.
- 
#### GridSearch (`DT_GRID_SEARCH=True`)
Otimização por `GridSearchCV` com `refit="f1_macro"` e os seguintes espaços de busca:

| Parâmetro              | Valores buscados            |
|------------------------|-----------------------------|
| `criterion`            | `gini`, `entropy`           |
| `max_depth`            | `3, 5, 8, 12, None`         |
| `min_samples_split`    | `2, 10, 30`                 |
| `min_samples_leaf`     | `1, 5, 10`                  |
| `max_features`         | `sqrt`, `log2`              |
| `class_weight`         | `balanced`                  |
| `ccp_alpha`            | `0.0, 1e-4, 1e-3`           |

#### Configuração manual (`DT_GRID_SEARCH=False`)
Parâmetros definidos a partir do melhor resultado obtido em execução anterior do GridSearch:

- `criterion="gini"`
- `max_depth=12`
- `min_samples_split=2`
- `min_samples_leaf=1`
- `max_features="log2"`
- `class_weight="balanced"`
- `ccp_alpha=0.001`
- `random_state=42`

### KNN
- Modelo `KNeighborsClassifier`
- Pipeline:
  - Numéricas: `StandardScaler`
  - Categóricas: `OneHotEncoder`
 - KNN é sensível à escala, então a normalização é obrigatória. `OneHotEncoder` converte categóricas em distâncias mensuráveis. `refit="f1_macro"` no GridSearch prioriza equilíbrio entre classes.

 
#### GridSearch (`KNN_GRID_SEARCH=True`)
Otimização por `GridSearchCV` com `refit="f1_macro"` e os seguintes espaços de busca:

| Parâmetro         | Valores buscados                          |
|-------------------|-------------------------------------------|
| `n_neighbors`     | `3, 5, 7, ..., 31` (ímpares, range(3,32,2)) |
| `weights`         | `uniform`, `distance`                     |
| `p`               | `1` (Manhattan), `2` (Euclidiana)         |

#### Configuração manual (`KNN_GRID_SEARCH=False`)
Parâmetros definidos a partir do melhor resultado obtido em execução anterior do GridSearch:

- `n_neighbors=3`
- `weights="distance"`
- `p=2` (distância Euclidiana)

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

## Comparação das medidas de avaliação

| Modelo | Acurácia | F1-macro | Recall YES |
|---|---:|---:|---:|
| Naive Bayes | 0.8519 | 0.6773 | 0.4972 |
| Árvore de Decisão | 0.8286 | 0.6876 | **0.6701** |
| KNN (k=3) | **0.8921** | **0.6894** | 0.3601 |

### Leitura rápida

- **Melhor acurácia e F1-macro:** KNN (`k=3`)
- **Melhor recall da classe positiva (`yes`):** Árvore de Decisão
- **Naive Bayes:** desempenho equilibrado, com recall `yes` intermediário

> No contexto de telemarketing, **recall da classe `yes`** é a métrica mais relevante: errar um cliente que aceitaria o produto tem custo maior do que ligar para quem recusaria. Por esse critério, a **Árvore de Decisão** é o modelo mais adequado.

---

## Interpretabilidade

### Árvore de Decisão
- **Método:** Visualização da árvore e importância de features (`feature_importances_`).
- **Vantagem:** Totalmente interpretável, cada predição pode ser rastreada como uma sequência de regras if/else.
- **Limitação:** Árvores profundas (`max_depth=12`) perdem legibilidade. Poda via `ccp_alpha` mitiga isso.

### Naive Bayes
- **Método:** Análise das probabilidades condicionais (`theta_` e `var_` do GaussianNB).
- **Vantagem:** Interpretável via probabilidades, é possível identificar quais valores de feature mais elevam P(yes | X).
- **Limitação:** Assume independência entre features, o que raramente é verdadeiro. Resultados podem ser enganosos quando há correlação entre variáveis (ex: `age` e `job`).

### KNN
- **Método:** SHAP (SHapley Additive exPlanations).
- **Vantagem:** SHAP distribui a contribuição de cada feature para uma predição com base em teoria dos jogos, permitindo explicações tanto locais (por instância) quanto globais (importância média das features no modelo).
- **Limitação:** KNN é inerentemente uma caixa-preta — não há parâmetros globais interpretáveis. A explicação depende do ponto analisado e pode variar muito entre instâncias similares.

---

## Comparação e Análise final

- Os modelos, em linhas gerais, concordam nas variáveis mais relevantes. Features como `duration`, `poutcome` e `balance` aparecem como relevantes nos três modelos, o que dá consistência aos resultados.
- O alto recall da Árvore de Decisão faz sentido dado o uso de `class_weight="balanced"`. Já o KNN alcança maior acurácia geral mas sacrifica o recall da classe minoritária.
- A Árvore de Decisão oferece alta interpretabilidade, mas perde legibilidade com profundidade elevada. O Naive Bayes tem interpretabilidade média, limitada pela suposição de independência entre features. O KNN é o menos interpretável dos três, pois sem parâmetros globais, depende de SHAP para explicar predições individuais.

## Como executar

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy matplotlib seaborn ucimlrepo lime tqdm tqdm-joblib shap
python main.py
```
