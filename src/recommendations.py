
def recommend(row, probability, shap_reasons=None):
    actions = []
    if row.get("Contract") == "Month-to-month":
        actions.append("Offer a discounted 1-year contract with clear cancellation terms.")
    if float(row.get("MonthlyCharges", 0)) >= 75:
        actions.append("Review the plan and offer a lower-cost bundle if margin permits.")
    if int(row.get("tenure", 0)) <= 12:
        actions.append("Offer an early-tenure loyalty benefit and proactive onboarding support.")
    if row.get("TechSupport") == "No":
        actions.append("Offer a trial or discounted technical-support package.")
    if row.get("OnlineSecurity") == "No":
        actions.append("Offer an optional online-security bundle.")
    if probability >= 0.60:
        actions.append("Prioritize a personal retention contact.")
    if not actions:
        actions.append("Maintain normal engagement and monitor future risk.")
    return actions
