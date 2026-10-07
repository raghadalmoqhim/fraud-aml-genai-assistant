import os

from dotenv import load_dotenv
from google import genai



def generate_analyst_summary(prompt):
    load_dotenv()

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return {
            "success": False,
            "message": (
                "AI service configuration is missing. "
                "Please contact the system administrator."
            )
        }

    try:
        client = genai.Client(api_key=api_key)

        response = client.models.generate_content(
            model="gemini-3.5-flash",
            contents=prompt,
        )


        return {
            "success": True,
            "message": response.text
        }

    except Exception as error:
        print(f"Gemini API Error: {error}")

        return {
            "success": False,
            "message": (
                "The AI analyst service is temporarily unavailable. "
                "or its usage limit has been reached. "
                "Please try again later."
            )
        }

def build_analyst_prompt(alert, retrieved_context):

    context_text = ""

    for result in retrieved_context:
        # Ignore weak retrieval results
        if result["similarity"] >= 0.40:
            context_text += f"""
AML Guidance:
{result["chunk"]}

Similarity Score: {result["similarity"]:.2f}
"""

    prompt = f"""
You are an AI assistant supporting a Fraud and AML analyst.

Fraud Detection Result:
- Fraud Probability: {alert['risk_score']}%
- Risk Level: {alert['risk_level']}
- Transaction Amount: {alert['transaction_amount']}
- Requires Review: {alert['requires_review']}

Retrieved AML Guidance:
{context_text}

Instructions:
1. Write a short, professional summary for a Fraud/AML analyst.
2. Use ONLY the fraud detection result and the retrieved AML guidance provided above.
3. Do NOT introduce examples, explanations, scenarios, typologies, or facts that are not explicitly provided.
4. Do NOT infer why the transaction amount is 0.0.
5. Do NOT claim that the transaction itself has any AML red flag unless that fact is explicitly present in the transaction data.
6. Clearly distinguish the ML fraud score from the retrieved AML guidance.
7. Treat retrieved AML guidance as general reference information, not evidence about this transaction.
8. Do NOT conclude that fraud or money laundering occurred.
9. If information is unavailable, state that additional transaction or customer context is required.
10. Recommend manual review when the alert requires review.

Return the response in Markdown using exactly these three section headings:

### Risk Summary

Write the risk summary as a short paragraph.

### Relevant AML Considerations

Present relevant retrieved AML guidance clearly.
If no AML guidance was retrieved, state that no AML guidance was retrieved.
Do not create or infer AML red flags.

### Recommended Action

Write the recommended action as a short paragraph.

Do not format the three section headings as bullet points or numbered items.
"""

    return prompt
