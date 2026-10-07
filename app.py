import pandas as pd
import streamlit as st

from src.alert_engine import create_fraud_alert
from src.fraud_model import predict_fraud_batch
from src.llm_engine import build_analyst_prompt, generate_analyst_summary
from src.rag_engine import (
    create_embeddings,
    load_knowledge_base,
    search_knowledge_base,
    split_into_chunks,
)


# =========================================================
# CONFIGURATION
# =========================================================

REQUIRED_COLUMNS = (
    ["Time"]
    + [f"V{i}" for i in range(1, 29)]
    + ["Amount"]
)

HIGH_RISK_THRESHOLD = 80
MEDIUM_RISK_THRESHOLD = 50
RAG_SIMILARITY_THRESHOLD = 0.40


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Fraud & AML GenAI Assistant",
    page_icon="🔍",
    layout="wide",
)


# =========================================================
# CUSTOM UI
# =========================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1400px;
    }

    .main-title {
        font-size: 2.5rem;
        font-weight: 750;
        margin-bottom: 0.2rem;
    }

    .subtitle {
        color: #9CA3AF;
        font-size: 1.05rem;
        margin-bottom: 2rem;
    }

    .section-label {
        color: #9CA3AF;
        font-size: 0.8rem;
        font-weight: 700;
        letter-spacing: 0.08rem;
        text-transform: uppercase;
        margin-bottom: 0.3rem;
    }

    div.stButton > button[kind="primary"] {
        background-color: #2563EB !important;
        border: 1px solid #3B82F6 !important;
        color: white !important;
        border-radius: 8px !important;
        font-weight: 600 !important;
    }

    div.stButton > button[kind="primary"]:hover {
        background-color: #1D4ED8 !important;
        border-color: #60A5FA !important;
        color: white !important;
    }

    div.stButton > button[kind="primary"]:focus {
        box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.30) !important;
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 16px;
        margin-top: 22px;
        margin-bottom: 38px;
    }

    .metric-card {
        padding: 22px;
        border-radius: 14px;
        background: #141821;
        border: 1px solid #252b36;
        min-height: 125px;
    }

    .metric-label {
        font-size: 12px;
        font-weight: 700;
        letter-spacing: 1.2px;
        color: #9ca3af;
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        margin-top: 8px;
        color: #ffffff;
    }

    .metric-note {
        margin-top: 6px;
        font-size: 13px;
        color: #8b93a1;
    }

    .blue-card {
        border-top: 3px solid #3b82f6;
    }

    .red-card {
        border-top: 3px solid #ef4444;
    }

    .orange-card {
        border-top: 3px solid #f59e0b;
    }

    .purple-card {
        border-top: 3px solid #8b5cf6;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# =========================================================
# HELPERS
# =========================================================

@st.cache_resource
def load_rag_resources():
    knowledge = load_knowledge_base(
        "knowledge_base/aml_red_flags.txt"
    )

    chunks = split_into_chunks(knowledge)

    embedding_model, embeddings = create_embeddings(
        chunks
    )

    return chunks, embedding_model, embeddings


def build_results_dataframe(uploaded_df, probabilities):
    results_df = pd.DataFrame(
        {
            "Transaction ID": uploaded_df.index + 1,
            "Amount": uploaded_df["Amount"].astype(float).round(2),
            "Fraud Probability": (probabilities * 100).round(2),
        }
    )

    results_df["Risk Level"] = "LOW"

    results_df.loc[
        results_df["Fraud Probability"] >= MEDIUM_RISK_THRESHOLD,
        "Risk Level",
    ] = "MEDIUM"

    results_df.loc[
        results_df["Fraud Probability"] >= HIGH_RISK_THRESHOLD,
        "Risk Level",
    ] = "HIGH"

    results_df["Requires Review"] = results_df[
        "Risk Level"
    ].isin(["HIGH", "MEDIUM"])

    return results_df


def get_retrieved_context(selected_alert):
    (
        chunks,
        embedding_model,
        embeddings,
    ) = load_rag_resources()

    query = (
        f"Transaction with {selected_alert['risk_level']} fraud risk "
        f"and fraud probability of {selected_alert['risk_score']}%. "
        "The transaction requires further review."
    )

    retrieved_context = search_knowledge_base(
        query,
        chunks,
        embeddings,
        embedding_model,
        top_k=3,
    )

    return [
        result
        for result in retrieved_context
        if (
            result["similarity"] >= RAG_SIMILARITY_THRESHOLD
            and "Geographic Risk" not in result["chunk"]
        )
    ]


# =========================================================
# HEADER
# =========================================================

st.markdown(
    '<div class="main-title">Fraud & AML GenAI Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    (
        '<div class="subtitle">'
        "AI-powered transaction risk analysis combining "
        "machine learning, AML knowledge retrieval, and "
        "generative AI."
        "</div>"
    ),
    unsafe_allow_html=True,
)


# =========================================================
# TRANSACTION UPLOAD
# =========================================================

st.markdown(
    '<div class="section-label">Transaction Analysis</div>',
    unsafe_allow_html=True,
)

st.header("Upload Transaction Data")

st.write(
    "Upload a CSV file containing transactions. "
    "The fraud detection model will analyze each transaction "
    "and prioritize those requiring analyst review."
)

uploaded_file = st.file_uploader(
    "Upload transaction CSV",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "Upload a CSV file to begin transaction analysis."
    )
    st.stop()

uploaded_df = pd.read_csv(uploaded_file)

st.success(
    f"{len(uploaded_df):,} transactions loaded successfully."
)

st.caption(
    f"{len(uploaded_df):,} rows • "
    f"{len(uploaded_df.columns)} columns"
)


# =========================================================
# BATCH ANALYSIS
# =========================================================

st.divider()

analyze_clicked = st.button(
    "Analyze Transactions",
    type="primary",
    width="stretch",
)

if analyze_clicked:
    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in uploaded_df.columns
    ]

    if missing_columns:
        st.error(
            "The uploaded file is missing required columns: "
            + ", ".join(missing_columns)
        )
        st.stop()

    with st.spinner("Analyzing transactions..."):
        model_input = uploaded_df[
            REQUIRED_COLUMNS
        ].copy()

        probabilities = predict_fraud_batch(
            model_input
        )

        results_df = build_results_dataframe(
            uploaded_df,
            probabilities,
        )

        alerts_df = (
            results_df[
                results_df["Requires Review"]
            ]
            .sort_values(
                by="Fraud Probability",
                ascending=False,
            )
            .reset_index(drop=True)
        )

        st.session_state["results_df"] = results_df
        st.session_state["alerts_df"] = alerts_df

        # Prevent a summary from an earlier analysis
        # from appearing for newly analyzed data.
        st.session_state.pop(
            "investigation_summary",
            None,
        )
        st.session_state.pop(
            "summary_transaction_id",
            None,
        )

    st.success(
        f"{len(results_df):,} transactions "
        "analyzed successfully."
    )


# Stop here until an analysis has been completed.
if (
    "results_df" not in st.session_state
    or "alerts_df" not in st.session_state
):
    st.stop()

results_df = st.session_state["results_df"]
alerts_df = st.session_state["alerts_df"]


# =========================================================
# FRAUD INVESTIGATION DASHBOARD
# =========================================================

st.divider()

st.markdown(
    '<div class="section-label">Fraud Monitoring</div>',
    unsafe_allow_html=True,
)

st.header("Fraud Investigation Dashboard")

st.write(
    "Overview of transaction risk results generated "
    "by the fraud detection model."
)

total_transactions = len(results_df)
high_risk = (results_df["Risk Level"] == "HIGH").sum()
medium_risk = (results_df["Risk Level"] == "MEDIUM").sum()
requires_review = results_df["Requires Review"].sum()

st.markdown(
    f"""
<div class="metric-grid">
<div class="metric-card blue-card">
<div class="metric-label">TRANSACTIONS</div>
<div class="metric-value">{total_transactions}</div>
<div class="metric-note">Transactions analyzed</div>
</div>

<div class="metric-card red-card">
<div class="metric-label">HIGH RISK</div>
<div class="metric-value">{int(high_risk)}</div>
<div class="metric-note">Priority alerts</div>
</div>

<div class="metric-card orange-card">
<div class="metric-label">MEDIUM RISK</div>
<div class="metric-value">{int(medium_risk)}</div>
<div class="metric-note">Require attention</div>
</div>

<div class="metric-card purple-card">
<div class="metric-label">REQUIRES REVIEW</div>
<div class="metric-value">{int(requires_review)}</div>
<div class="metric-note">Analyst review queue</div>
</div>
</div>
""",
    unsafe_allow_html=True,
)


# =========================================================
# PRIORITIZED ALERTS
# =========================================================

st.subheader("Prioritized Alerts")

st.caption(
    "Transactions are ranked by fraud probability "
    "to help analysts review the highest-risk alerts first."
)

if alerts_df.empty:
    st.success(
        "No transactions currently require analyst review."
    )
    st.stop()

st.dataframe(
    alerts_df,
    width="stretch",
    hide_index=True,
    column_config={
        "Transaction ID": st.column_config.NumberColumn(
            "Transaction ID",
            format="%d",
        ),
        "Amount": st.column_config.NumberColumn(
            "Amount",
            format="%.2f",
        ),
        "Fraud Probability": st.column_config.ProgressColumn(
            "Fraud Probability",
            min_value=0,
            max_value=100,
            format="%.0f%%",
        ),
        "Risk Level": st.column_config.TextColumn(
            "Risk Level"
        ),
        "Requires Review": st.column_config.CheckboxColumn(
            "Requires Review"
        ),
    },
)


# =========================================================
# INVESTIGATION DETAIL
# =========================================================

st.markdown("<br>", unsafe_allow_html=True)

st.markdown(
    '<div class="section-label">Alert Investigation</div>',
    unsafe_allow_html=True,
)

st.header("Investigation Detail")

st.write(
    "Select a prioritized alert to review its fraud risk, "
    "retrieve relevant AML guidance, and generate an "
    "AI-assisted analyst summary."
)

alert_lookup = alerts_df.set_index(
    "Transaction ID"
)["Risk Level"].to_dict()

selected_alert_id = st.selectbox(
    "Select Alert",
    options=alerts_df["Transaction ID"].tolist(),
    format_func=lambda transaction_id: (
        f"Transaction {transaction_id} — "
        f"{alert_lookup[transaction_id]} Risk"
    ),
)

selected_result = results_df[
    results_df["Transaction ID"] == selected_alert_id
].iloc[0]

selected_transaction = uploaded_df.iloc[
    int(selected_alert_id) - 1
]

transaction = selected_transaction.to_dict()

# Class is ground truth for the demo dataset only.
# It is never passed into the fraud model.
transaction.pop("Class", None)

selected_probability = (
    float(selected_result["Fraud Probability"]) / 100
)

selected_alert = create_fraud_alert(
    selected_probability,
    transaction["Amount"],
)


# =========================================================
# ALERT OVERVIEW
# =========================================================

st.markdown("### Alert Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Transaction ID",
    selected_alert_id,
)

col2.metric(
    "Fraud Probability",
    f"{selected_alert['risk_score']}%",
)

col3.metric(
    "Risk Level",
    selected_alert["risk_level"],
)

col4.metric(
    "Amount",
    f"{selected_alert['transaction_amount']:.2f}",
)


# =========================================================
# AML GUIDANCE
# =========================================================

retrieved_context = get_retrieved_context(
    selected_alert
)

st.markdown("### Relevant AML Guidance")

st.caption(
    "Retrieved as contextual AML guidance. "
    "These results are not evidence that a red flag is present."
)

if retrieved_context:
    for result in retrieved_context:
        chunk_lines = result["chunk"].split(
            "\n",
            1,
        )

        title = chunk_lines[0]
        description = (
            chunk_lines[1]
            if len(chunk_lines) > 1
            else ""
        )

        with st.container(border=True):
            st.markdown(f"**{title}**")
            st.write(description)
            st.caption(
                "Semantic similarity: "
                f"{result['similarity']:.0%}"
            )
else:
    st.info(
        "No sufficiently relevant AML guidance "
        "was retrieved for this alert."
    )


# =========================================================
# AI ANALYST SUMMARY
# =========================================================

st.markdown("### AI Analyst Summary")

st.caption(
    "Generate a grounded analyst summary using "
    "the fraud result and retrieved AML guidance."
)

if st.button(
    "Generate AI Analyst Summary",
    type="primary",
):
    analyst_prompt = build_analyst_prompt(
        selected_alert,
        retrieved_context,
    )

    with st.spinner(
        "Generating analyst summary..."
    ):
        ai_result = generate_analyst_summary(
            analyst_prompt
        )

    if ai_result["success"]:
        st.session_state[
            "investigation_summary"
        ] = ai_result["message"]

        st.session_state[
            "summary_transaction_id"
        ] = selected_alert_id
    else:
        st.error(ai_result["message"])


# =========================================================
# DISPLAY SAVED SUMMARY
# =========================================================

if (
    "investigation_summary" in st.session_state
    and st.session_state.get(
        "summary_transaction_id"
    ) == selected_alert_id
):
    st.markdown(
        st.session_state[
            "investigation_summary"
        ]
    )
