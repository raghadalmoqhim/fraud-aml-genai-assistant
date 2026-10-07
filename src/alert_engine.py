def get_risk_level(fraud_probability):
    if fraud_probability >= 0.80:
        return "HIGH"
    elif fraud_probability >= 0.50:
        return "MEDIUM"
    else:
        return "LOW"



def create_fraud_alert(fraud_probability, amount):
    fraud_probability = float(fraud_probability)
    amount = float(amount)

    risk_level = get_risk_level(fraud_probability)

    alert = {
        "fraud_probability": round(fraud_probability, 4),
        "risk_score": round(fraud_probability * 100, 2),
        "risk_level": risk_level,
        "transaction_amount": round(amount, 2),
        "requires_review": risk_level in ["HIGH", "MEDIUM"]
    }

    return alert


