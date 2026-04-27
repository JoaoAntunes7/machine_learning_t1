from pathlib import Path
from scipy import sparse
from lime.lime_tabular import LimeTabularExplainer
import numpy as np

def explain_with_lime(
    fitted_pipeline,
    X_train_df,
    X_test_df,
    y_test_arr,
    y_pred_arr,
    class_names,
    model_name,
    random_state=42,
    sample_size=2000,
    num_features=10,
):
    """
    Explica uma previsão com LIME para pipelines sklearn (pré-processador + classificador).
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    # Amostra de referência para o LIME
    n = min(sample_size, len(X_train_df))
    ref_df = X_train_df.sample(n=n, random_state=random_state)
    X_ref = preprocessor.transform(ref_df)
    if sparse.issparse(X_ref):
        X_ref = X_ref.toarray()

    feature_names = preprocessor.get_feature_names_out()

    explainer = LimeTabularExplainer(
        training_data=X_ref,
        feature_names=feature_names,
        class_names=list(class_names),
        mode="classification",
        random_state=random_state,
        discretize_continuous=True,
    )

    # escolhe 1 exemplo para explicar (prioriza erro)
    error_idx = np.where(y_pred_arr != y_test_arr)[0]
    idx = int(error_idx[0]) if len(error_idx) > 0 else 0

    x_row = X_test_df.iloc[[idx]]
    x_trans = preprocessor.transform(x_row)
    if sparse.issparse(x_trans):
        x_trans = x_trans.toarray()

    pred_label = int(fitted_pipeline.predict(x_row)[0])
    exp = explainer.explain_instance(
        data_row=x_trans[0],
        predict_fn=classifier.predict_proba,
        num_features=num_features,
        top_labels=1,
    )

    print(f"\n--- LIME: {model_name} | amostra idx={idx} ---")
    print(f"Classe prevista: {class_names[pred_label]}")
    for feat, weight in exp.as_list(label=pred_label):
        print(f"{feat}: {weight:+.4f}")

    out_dir = Path("./interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"lime_{model_name.lower().replace(' ', '_')}.html"
    exp.save_to_file(str(out_file))
    print(f"Explicação salva em: {out_file}")