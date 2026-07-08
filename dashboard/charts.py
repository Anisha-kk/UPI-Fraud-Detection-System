import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix

# ==========================================================
# 1. FRAUD PROBABILITY HISTOGRAM
# ==========================================================
def fraud_probability_histogram(df: pd.DataFrame):
    if df is None or df.empty or "probability" not in df.columns:
        st.info("No probability data available")
        return
    
    fig = px.histogram(
        df,
        x="probability",
        nbins=30,
        color="prediction",
        template="plotly_white",
    )
    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 2. ONLINE VS BATCH PREDICTIONS
# ==========================================================
def prediction_type_chart(df: pd.DataFrame):
    if df is None or df.empty or "prediction_type" not in df.columns:
        st.info("No prediction type data available")
        return
    
    counts = df["prediction_type"].value_counts().reset_index()
    counts.columns = ["type", "count"]

    fig = px.bar(
        counts,
        x="type",
        y="count",
        text="count",
        title="Online vs Batch Predictions",
        color="type",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 3. FRAUD TREND (ROLLING AVERAGE)
# ==========================================================
def fraud_trend_chart(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data available for trend")
        return

    if "pred_timestamp" not in df.columns:
        st.info("Timestamp missing")
        return

    df = df.copy()
    df["pred_timestamp"] = pd.to_datetime(df["pred_timestamp"])
    df = df.sort_values("pred_timestamp")

    df["fraud_rate"] = df["prediction"].rolling(window=50, min_periods=1).mean()
    #This computes the average using however many predictions are available at the beginning, then transitions to a 
    #full 50-prediction window once enough data has accumulated.

    fig = px.line(
        df,
        x="pred_timestamp",
        y="fraud_rate",
        title="Fraud Rate Trend (Rolling 50)",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 4. PREDICTIONS PER DAY
# ==========================================================
def predictions_per_day(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data available")
        return

    if "pred_timestamp" not in df.columns:
        st.info("Timestamp missing")
        return

    df = df.copy()
    df["pred_timestamp"] = pd.to_datetime(df["pred_timestamp"])

    daily = (
        df.groupby(df["pred_timestamp"].dt.date)
        .size()
        .reset_index(name="count")
    )

    daily.columns = ["date", "count"]

    fig = px.line(
        daily,
        x="date",
        y="count",
        markers=True,
        title="Predictions per Day",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 5. TOP SENDERS
# ==========================================================
def top_senders(df: pd.DataFrame):
    if df is None or df.empty or "sender_id" not in df.columns:
        st.info("No sender data available")
        return

    top = df["sender_id"].value_counts().head(10).reset_index()
    top.columns = ["sender", "count"]

    fig = px.bar(
        top,
        x="sender",
        y="count",
        color="count",
        title="Top Senders",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 6. TOP RECEIVERS
# ==========================================================
def top_receivers(df: pd.DataFrame):
    if df is None or df.empty or "receiver_id" not in df.columns:
        st.info("No receiver data available")
        return

    top = df["receiver_id"].value_counts().head(10).reset_index()
    top.columns = ["receiver", "count"]

    fig = px.bar(
        top,
        x="receiver",
        y="count",
        color="count",
        title="Top Receivers",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 7. FRAUD BOXPLOT
# ==========================================================
def fraud_boxplot(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data available")
        return

    if "prediction" not in df.columns or "probability" not in df.columns:
        st.info("Required columns missing")
        return

    fig = px.box(
        df,
        x="prediction",
        y="probability",
        title="Fraud Probability by Prediction",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 8. PERFORMANCE (ROC / PR OVER TIME)
# ==========================================================
def performance_chart(batch_df: pd.DataFrame):
    if batch_df is None or batch_df.empty:
        st.info("No batch performance data")
        return

    if "run_time" not in batch_df.columns:
        st.info("run_time missing")
        return

    df = batch_df.copy()
    df["run_time"] = pd.to_datetime(df["run_time"])
    df = df.sort_values("run_time")

    fig = go.Figure()

    if "roc_auc" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["run_time"],
                y=df["roc_auc"],
                mode="lines+markers",
                name="ROC-AUC",
            )
        )

    if "pr_auc" in df.columns:
        fig.add_trace(
            go.Scatter(
                x=df["run_time"],
                y=df["pr_auc"],
                mode="lines+markers",
                name="PR-AUC",
            )
        )

    fig.update_layout(
        title="Model Performance Over Time",
        template="plotly_white",
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 9. DRIFT FEATURE CHART
# ==========================================================
def drift_feature_chart(diagnostics: dict):
    if not diagnostics:
        st.info("No drift data available")
        return
    anomalies = diagnostics.get("anomalies", [])
    if not anomalies:
        st.success("No drift detected")
        return
    rows = []
    for a in anomalies:
        if not isinstance(a, dict):
            continue
        issue_type = a.get("anomaly_type", "unknown")
        feature = a.get("feature", "unknown")
        value = a.get("value", None)

        # severity-based weighting (important upgrade)
        severity = a.get("severity", "low")

        rows.append({
            "issue_type": issue_type,
            "feature": feature,
            "severity" :severity,
            "details": str(value)
        })

    if not rows:
        st.success("No drift detected")
        return

    df = pd.DataFrame(rows)
    severity_map = {"low": 1, "medium": 2, "high": 3}
    df["weight"] = df["severity"].map(severity_map).fillna(1)
    agg = df.groupby(["feature", "issue_type"]).agg(
        frequency=("feature", "count"),
        severity_score=("weight", "sum"),
        severity=(
            "weight",
            lambda x: {
                1: "low",
                2: "medium",
                3: "high"
            }[x.max()]
        ),
        details=("details", lambda x: "<br>".join(sorted(set(x))))
    ).reset_index()
    # Sort so highest-risk features appear first
    agg = agg.sort_values(
        ["severity_score", "frequency"],
        ascending=False
    )

    #PLOT
    severity_colors = {
    "low": "#2ca02c",      # Green
    "medium": "#ffb000",   # Orange
    "high": "#d62728"      # Red
    }
    fig = px.bar(
    agg,
    x="frequency",
    y="feature",
    orientation="h",
    color="severity",
    pattern_shape="issue_type",
    color_discrete_map=severity_colors,
    text="frequency",
    custom_data=[
        "issue_type",
        "severity",
        "details",
        "severity_score"
    ]
    )

    fig.update_traces(
        hovertemplate="""
    <b>Feature:</b> %{y}<br>
    <b>Issue Type:</b> %{customdata[0]}<br>
    <b>Frequency:</b> %{x}<br>
    <b>Severity:</b> %{customdata[1]}<br>
    <b>Risk Score:</b> %{customdata[3]}<br><br>
    <b>Details:</b><br>
    %{customdata[2]}
    <extra></extra>
    """
    )
    fig.update_layout(
    xaxis_title="Frequency",
    yaxis_title="Feature",
    legend_title="Severity",
    template="plotly_white",
    height=max(450, 45 * len(agg)),
    bargap=0.25
    )

    st.plotly_chart(fig, use_container_width=True)

# ==========================================================
# 10. CONFUSION MATRIX (SAFE FOR NULL LABELS)
# ==========================================================
def confusion_matrix_chart(df: pd.DataFrame):
    if df is None or df.empty:
        st.info("No data for confusion matrix")
        return

    if "true_label" not in df.columns:
        st.info("true_label not available (online predictions)")
        return

    df = df.dropna(subset=["true_label"])

    if df.empty:
        st.info("No labeled data available")
        return

    cm = confusion_matrix(df["true_label"], df["prediction"])

    fig = px.imshow(
        cm,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Confusion Matrix",
        labels=dict(x="Predicted", y="Actual"),
        x=["Normal", "Fraud"],
        y=["Normal", "Fraud"],
    )

    st.plotly_chart(fig, use_container_width=True)

#======================================================
# LATENCY CHART
#======================================================
def latency_chart(latency_df):
    fig = px.line(
        latency_df,
        x="pred_timestamp",
        y="latency",
        markers=True,
        title="Model Latency Over Time (ms)"
    )

    fig.update_layout(
        xaxis_title="Timestamp",
        yaxis_title="Latency (ms)",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)