
from src.recommendations import recommend

def test_high_risk_recommendations_include_priority_contact():
    actions = recommend({"Contract":"Month-to-month", "MonthlyCharges":90, "tenure":3}, 0.8)
    assert any("Prioritize" in action for action in actions)
