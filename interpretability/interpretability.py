from pathlib import Path
from scipy import sparse
from lime.lime_tabular import LimeTabularExplainer
import numpy as np
import shap
import matplotlib.pyplot as plt

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

def explain_with_shap(
    fitted_pipeline,
    X_train_df,
    X_test_df,
    y_test_arr,
    y_pred_arr,
    class_names,
    model_name,
    random_state=42,
    background_size=200,
    nsamples=300,
    num_features=10,
):
    """
    SHAP model-agnostic (KernelExplainer) para pipelines sklearn.
    """
    preprocessor = fitted_pipeline.named_steps["preprocessor"]
    classifier = fitted_pipeline.named_steps["classifier"]

    # background (amostra de referência)
    n_bg = min(background_size, len(X_train_df))
    bg_df = X_train_df.sample(n=n_bg, random_state=random_state)
    X_bg = preprocessor.transform(bg_df)
    if sparse.issparse(X_bg):
        X_bg = X_bg.toarray()

    # escolhe 1 exemplo para explicar (primeiro erro; senão idx=0)
    error_idx = np.where(y_pred_arr != y_test_arr)[0]
    idx = int(error_idx[0]) if len(error_idx) > 0 else 0

    x_row_df = X_test_df.iloc[[idx]]
    x_row = preprocessor.transform(x_row_df)
    if sparse.issparse(x_row):
        x_row = x_row.toarray()

    pred_label = int(classifier.predict(x_row)[0])

    explainer = shap.KernelExplainer(classifier.predict_proba, X_bg)
    sv = explainer.shap_values(x_row, nsamples=nsamples)

    # compatibilidade entre versões do shap
    if isinstance(sv, list):
        shap_values_class = sv[pred_label][0]
        base_value = explainer.expected_value[pred_label]
    else:
        sv = np.array(sv)
        if sv.ndim == 3:  # (n_samples, n_features, n_classes)
            shap_values_class = sv[0, :, pred_label]
            base_value = explainer.expected_value[pred_label]
        else:  # (n_samples, n_features)
            shap_values_class = sv[0]
            base_value = explainer.expected_value

    feature_names = preprocessor.get_feature_names_out()
    order = np.argsort(np.abs(shap_values_class))[::-1][:num_features]

    print(f"\n--- SHAP: {model_name} | amostra idx={idx} ---")
    print(f"Classe prevista: {class_names[pred_label]}")
    print(f"Classe real:     {class_names[int(y_test_arr[idx])]}")
    print(f"Status:          {'ERRO' if pred_label != int(y_test_arr[idx]) else 'ACERTO'}")
    for i in order:
        print(f"{feature_names[i]}: {shap_values_class[i]:+.4f}")

    # salva waterfall
    out_dir = Path("./interpretability")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"shap_{model_name.lower().replace(' ', '_')}.png"

    explanation = shap.Explanation(
        values=shap_values_class,
        base_values=base_value,
        data=x_row[0],
        feature_names=feature_names,
    )

    plt.figure()
    shap.plots.waterfall(explanation, max_display=num_features, show=False)
    plt.tight_layout()
    plt.savefig(out_file, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Explicação SHAP salva em: {out_file}")