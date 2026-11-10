
def make_shap_explainer(model, X_background):
    try:
        import shap
        # SHAP's model-agnostic explainer supports arbitrary sklearn pipelines.
        return shap.Explainer(model.predict_proba, X_background)
    except Exception as exc:
        raise RuntimeError(f"Unable to initialize SHAP explainer: {exc}") from exc
