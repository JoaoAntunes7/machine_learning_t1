# T1 — Interpretabilidade de Modelos de Aprendizado de Máquina

## Dataset

**Bank Marketing** (UCI Machine Learning Repository, ID 222)  
45.211 instâncias · 16 features (7 numéricas, 9 categóricas) · Alvo binário: `y` (yes/no)

O problema é prever se um cliente de banco irá subscrever um depósito a prazo após uma campanha de telemarketing.

Features numéricas: `age`, `balance`, `day_of_week`, `campaign`, `pdays`, `previous`  
Features categóricas: `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`

A coluna `duration` foi **removida intencionalmente** (data leakage): ela representa a duração da ligação, que só é conhecida *após* a ligação acontecer — usá-la inflaria artificialmente a performance.

---

## Pré-processamento

- Remoção de duplicatas
- Split estratificado (`train_test_split(..., stratify=y)`) com `test_size=0.2`
- `RobustScaler` para variáveis numéricas
- `OneHotEncoder` para variáveis categóricas
- `SimpleImputer` para faltantes (mediana em numéricas / moda em categóricas)
- `LabelEncoder` no alvo (`no=0`, `yes=1`) para compatibilidade estável com KNN e métricas binárias
- `random_state=42` para reprodutibilidade

---

## Modelos avaliados
## Modelos e Hiperparâmetros

### Naive Bayes (`GaussianNB`)
Assume que cada feature segue distribuição normal dentro de cada classe e calcula a probabilidade de cada classe dado o conjunto de features (Teorema de Bayes). `GaussianNB` foi escolhido sobre `MultinomialNB` porque as features numéricas do dataset (age, balance, etc.) têm distribuição contínua compatível com a suposição gaussiana.

### Árvore de Decisão (`DecisionTreeClassifier`)
Aprende uma sequência de perguntas binárias sobre as features para separar as classes.  
- `max_depth=5`: evita overfitting e mantém a árvore legível  
- `criterion="gini"`: índice Gini mede impureza dos nós (padrão e computacionalmente eficiente)

### KNN com GridSearchCV

Foi usado `GridSearchCV` com múltiplas métricas:

- `accuracy`
- `f1_macro`
- `recall_yes` (classe positiva = `yes`)

Espaço de busca (`param_grid`):

- `classifier__n_neighbors`: 3 a 31 (ímpares)
- `classifier__weights`: `uniform`, `distance`
- `classifier__p`: 1 (Manhattan), 2 (Euclidiana)

#### Resultado do grid (execução debug)

- Melhor por `accuracy`: `k=31, p=1, weights=uniform`
- Melhor por `f1_macro`: `k=3, p=1, weights=uniform`
- Melhor por `recall_yes`: `k=3, p=1, weights=distance`

> Na decisão final do trabalho, a escolha principal para KNN prioriza **`recall_yes`**.

#### Observação

- O GridSearch foi desabilitado por padrão e manteve-se a configuração que melhor resultou o `recall_yes`, porém caso queira executa-lo, basta descomentar as linhas que o definem.

---

## Resultados (teste)

| Modelo | Acurácia | F1-macro | Recall YES |
|---|---:|---:|---:|
| Naive Bayes | 0.8466 | **0.6562** | **0.4376** |
| Árvore de Decisão | **0.8937** | 0.6268 | 0.2051 |
| KNN (seleção com foco em recall) | 0.8731 | 0.6117 | 0.2250 |

### Observações rápidas

- **Naive Bayes** teve o maior `recall` da classe positiva (`yes`), importante quando perder positivos custa caro.
- **Árvore de decisão** teve maior acurácia geral, mas baixo `recall_yes`.
- **KNN** ficou intermediário; com Grid Search, os melhores hiperparâmetros variam conforme a métrica escolhida.

---

## Interpretabilidade

### Árvore de Decisão
- **Visualização da árvore** (`decision_tree.png`): mostra as regras aprendidas diretamente — cada nó é uma pergunta, cada folha é uma classe
- **Feature Importance**: `poutcome_success` domina com 62% da importância — clientes que subscreveram em campanhas anteriores têm alta chance de subscrever novamente

### Naive Bayes
- **Probabilidades condicionais**: o modelo aprende P(feature | classe) para cada feature. Features com distribuições muito diferentes entre "yes" e "no" são as mais discriminativas
- O GaussianNB armazena média e desvio padrão de cada feature por classe — possível inspecionar via `model.theta_` e `model.var_`

### KNN
- **Permutation Importance** (`permutation_importance_knn.png`): embaralha cada feature e mede a queda de acurácia. `pdays` é a mais importante (0.016), porém todos os valores são baixos — o KNN distribui seu poder preditivo entre todas as features ao mesmo tempo
- **LIME** (`lime_explanation.html`): explica predições individuais criando perturbações locais e ajustando um modelo linear simples. Para a amostra analisada: ausência de contato prévio (`pdays=0`, `previous=0`) foi o maior fator para classificar como "no"

---

## Comparação dos Modelos

| Modelo | Acurácia | F1-macro | Recall "yes" | Interpretabilidade |
|---|---|---|---|---|
| Naive Bayes | 0.8466 | **0.6562** | **43.7%** | Média — via probabilidades condicionais |
| Árvore de Decisão | **0.8937** | 0.6268 | 20.5% | **Alta** — regras explícitas e visíveis |
| KNN (k=11) | 0.8910 | 0.6021 | 16.6% | Baixa — requer LIME/Permutation externamente |

**Os modelos concordaram?** Parcialmente. Todos identificaram `poutcome`, `pdays` e `month` como relevantes. Porém, apenas a Árvore isolou `poutcome_success` como feature dominante (62%) — KNN e NB diluem a importância por operarem com todas as features simultaneamente.

**Os resultados fazem sentido?** Sim. `poutcome_success` significa "o cliente subscreveu na campanha anterior" — comportamento passado é o melhor preditor de comportamento futuro. `pdays` (dias desde o último contato) e determinados meses (`mar`, `oct`) refletem a sazonalidade das campanhas.

**Limitações de interpretabilidade:**
- *Naive Bayes*: assume independência entre features (frequentemente violada), o que distorce as probabilidades condicionais reais
- *Árvore de Decisão*: com `max_depth` alto vira uma caixa-preta; `max_depth=5` é legível mas pode subrepresentar relações complexas
- *KNN*: sem modelo interno — interpreta-se o comportamento, não os parâmetros. Sensível ao desbalanceamento de classes

---

## Como executar

```bash
python3 -m venv venv
source venv/bin/activate
pip install scikit-learn pandas numpy matplotlib seaborn ucimlrepo lime tqdm tqdm-joblib
python main.py