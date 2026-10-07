import joblib
import pandas as pd

from src.alert_engine import create_fraud_alert



# ---------------------------------
# 1. Load Fraud Detection Model
# ---------------------------------

model = joblib.load("models/random_forest_fraud.pkl")


# ---------------------------------
# 2. Fraud Prediction Function
# ---------------------------------

def predict_fraud(transaction):
    transaction_df = pd.DataFrame([transaction])

    fraud_probability = model.predict_proba(
        transaction_df
    )[0][1]

    return fraud_probability

def predict_fraud_batch(transactions_df):
    fraud_probabilities = model.predict_proba(
        transactions_df
    )[:, 1]

    return fraud_probabilities


# ---------------------------------
# 3. Testing Pipeline
# ---------------------------------

if __name__ == "__main__":
    from src.rag_engine import (
        load_knowledge_base,
        split_into_chunks,
        create_embeddings,
        search_knowledge_base
    )

    from src.llm_engine import (
        build_analyst_prompt,
        generate_analyst_summary
    )

    # Load cleaned dataset
    df = pd.read_csv(
        "data/processed/creditcard_clean.csv"
    )

    # Select one known fraud transaction
    # This is only for pipeline demonstration/testing.
    fraud_transaction = df[df["Class"] == 1].iloc[0]

    # Remove the target column before prediction
    transaction = fraud_transaction.drop("Class").to_dict()


    # ---------------------------------
    # 4. Fraud Prediction
    # ---------------------------------

    probability = predict_fraud(transaction)


    # ---------------------------------
    # 5. Create Fraud Alert
    # ---------------------------------

    alert = create_fraud_alert(
        probability,
        transaction["Amount"]
    )


    # ---------------------------------
    # 6. Load AML Knowledge Base
    # ---------------------------------

    knowledge = load_knowledge_base(
        "knowledge_base/aml_red_flags.txt"
    )

    chunks = split_into_chunks(knowledge)

    embedding_model, embeddings = create_embeddings(
        chunks
    )


    # ---------------------------------
    # 7. Retrieve Relevant AML Guidance
    # ---------------------------------

    query = (
        f"Transaction with {alert['risk_level']} fraud risk "
        f"and fraud probability of {alert['risk_score']}%. "
        f"The transaction requires further review."
    )

    retrieved_context = search_knowledge_base(
        query,
        chunks,
        embeddings,
        embedding_model,
        top_k=3
    )


    # ---------------------------------
    # 8. Build LLM Prompt
    # ---------------------------------

    analyst_prompt = build_analyst_prompt(
        alert,
        retrieved_context
    )


    # ---------------------------------
    # 9. Generate AI Analyst Summary
    # ---------------------------------

    analyst_summary = generate_analyst_summary(
        analyst_prompt
    )


    # ---------------------------------
    # 10. Display Results
    # ---------------------------------

    print("Actual Class:")
    print(fraud_transaction["Class"])

    print("\nFraud Probability:")
    print(round(float(probability), 4))

    print("\nFraud Alert:")
    print(alert)

    print("\nRAG Query:")
    print(query)

    print("\nTop Retrieved AML Context:")

    for i, result in enumerate(
        retrieved_context,
        start=1
    ):
        print(f"\nResult {i}")
        print(result["chunk"])
        print(
            "Similarity:",
            round(result["similarity"], 4)
        )

    print("\n--- AI Analyst Summary ---")
    print(analyst_summary)