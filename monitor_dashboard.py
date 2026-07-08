import json
import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pandas as pd
from dashboard.db_utils import (
    get_predictions,
    get_batch_runs,
    removed_rows,
)
from dashboard.db_utils import orm_to_dict
from dashboard import charts
from collections import defaultdict

#Helper function
def safe_has_columns(df, cols):
    return not df.empty and all(c in df.columns for c in cols)

# -----------------------
# Loading Data
# -----------------------
prediction_objects = get_predictions()
batch_objects = get_batch_runs()
predictions = pd.DataFrame([orm_to_dict(x) for x in prediction_objects]) if prediction_objects else pd.DataFrame()
batch_runs = pd.DataFrame([orm_to_dict(x) for x in batch_objects]) if batch_objects else pd.DataFrame()
# Both tables empty
if predictions.empty and batch_runs.empty:
    st.info("No data available to display.")
    st.stop()

if safe_has_columns(predictions, ["prediction_type"]):
    online_df = predictions[predictions["prediction_type"] == "online"]
    batch_df = predictions[predictions["prediction_type"] == "batch"]
else:
    online_df = pd.DataFrame()
    batch_df = pd.DataFrame()

if safe_has_columns(predictions, ["pred_timestamp"]):
    predictions["pred_timestamp"] = pd.to_datetime(predictions["pred_timestamp"])

if safe_has_columns(batch_runs, ["run_time"]):
    batch_runs["run_time"] = pd.to_datetime(batch_runs["run_time"])

# -----------------------
# Sidebar
# -----------------------
st.sidebar.title("Fraud Monitoring")
page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Prediction Logs",
        "Batch Runs",
        "Model Monitoring",
        "Drift Monitoring",
        "Alerts",
    ],
)

# ==========================================================
# OVERVIEW
# ==========================================================
if page == "Overview":
    st.title("Fraud Monitoring Dashboard")
    if predictions.empty:
        st.warning("No prediction data available")
        st.stop()

    total_predictions = len(predictions)
    online_predictions = (predictions["prediction_type"] == "online").sum()
    batch_predictions = (predictions["prediction_type"] == "batch").sum()
    fraud_rate = predictions["prediction"].mean()
    avg_probability = predictions["probability"].mean()

    today = pd.Timestamp.today().date()
    today_predictions = (pd.to_datetime(predictions["pred_timestamp"]).dt.date == today).sum()

    # ---------------- KPI ----------------
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Predictions",total_predictions,)
    c2.metric("Online Predictions",online_predictions,)
    c3.metric("Batch Predictions",batch_predictions,)

    c4, c5, c6 = st.columns(3)
    c4.metric("Fraud Rate",f"{fraud_rate:.2%}",)
    c5.metric("Average Fraud Probability",f"{avg_probability:.3f}",)
    c6.metric("Today's Predictions",today_predictions)

    # ---------------- CHARTS ----------------

    st.subheader("Fraud Probability Distribution")
    charts.fraud_probability_histogram(predictions)#Hist of number of trxns for each prob, olour coded by predicted class

    st.subheader("Prediction Type Split")
    charts.prediction_type_chart(predictions)#Count of trxns in online and batch predictions

    st.subheader("Fraud Trend")
    charts.fraud_trend_chart(predictions)#calculates the rolling average of 50 predictions.
    #Since the values are only 0s and 1s, the average is the fraction of fraud predictions in the last 50 transactions.
    #Without smoothing, a plot of individual predictions (0 or 1) would jump sharply between values.
    #The rolling mean smooths these fluctuations, making overall trends easier to see.

    st.subheader("Predictions per Day")
    charts.predictions_per_day(predictions)#Number of predictions done per day

# ==========================================================
# PREDICTION LOGS
# ==========================================================
elif page == "Prediction Logs":
    st.title("Prediction Logs for all inferences")
    if predictions.empty:
        st.warning("No data available")
        st.stop()

    txn = st.text_input("Search Transaction ID")
    sender = st.text_input("Search Sender")
    receiver = st.text_input("Search Receiver")

    df = predictions.copy()
    if txn:
        df = df[df["txn_id"].astype(str).str.contains(txn)]
    if sender:
        df = df[df["sender_id"].astype(str).str.contains(sender)]
    if receiver:
        df = df[df["receiver_id"].astype(str).str.contains(receiver)]
    
    st.subheader("Latest 100 Predictions")
    st.dataframe(df.head(100))

    # ----------------  CHARTS ----------------
    st.subheader("Top Senders")
    charts.top_senders(df)

    st.subheader("Top Receivers")
    charts.top_receivers(df)

    st.subheader("Fraud Probability Boxplot")
    charts.fraud_boxplot(df) #To check for outliers (ie, fraud cases with low prob or non fraud case with high prob)
    #And seperation between classes

# ==========================================================
# BATCH RUNS
# ==========================================================
elif page == "Batch Runs":
    if batch_runs.empty:
        st.warning("No data available")
        st.stop()

    st.title("Batch Runs")
    cols = [
        "batch_id",
        "run_time",
        "roc_auc",
        "pr_auc",
        "accuracy",
        "drift_present",
    ]
    st.dataframe(batch_runs[cols])
    # ---------------- CHART ----------------
    st.subheader("Performance Over Time")
    charts.performance_chart(batch_runs)#Shows ROC and PR_AUC values for different batches
    
    #Confusion matrix
    if predictions.empty:
        st.info("No predictions available")
    else:
        st.subheader("Confusion Matrix")
        if "true_label" in predictions.columns:
            valid = predictions.dropna(subset=["true_label"])
            if not valid.empty:
                charts.confusion_matrix_chart(valid)
            else:
                st.info("No labeled data available for confusion matrix")
        else:
            st.info("true_label not available")
    #Displaying batch wise confusion matrix
    st.subheader("Confusion Matrix for each batch")
    for batch in batch_runs.itertuples(index=False):
        with st.expander(f"Batch {batch.batch_id}"):
            if batch.confusion_matrix:
                cm = batch.confusion_matrix
                if cm:
                    if isinstance(cm, str):
                        cm = json.loads(cm)
                cm_df = pd.DataFrame(cm)
                st.dataframe(cm_df, use_container_width=True)
# ==========================================================
# MODEL MONITORING
# ==========================================================
elif page == "Model Monitoring":
    st.title("Fraud Trend")
    if predictions.empty:
        st.warning("No data available")
        st.stop()
    st.subheader("Fraud Trend Chart")
    charts.fraud_trend_chart(predictions)
    st.subheader("Latency for online inference")
    charts.latency_chart(online_df)
# ==========================================================
# DRIFT MONITORING
# ==========================================================
elif page == "Drift Monitoring":
    st.title("Feature Drift Monitoring")

    # ======================================================
    # 1. CHECK DATA
    # ======================================================

    if batch_runs.empty and predictions.empty:
        st.warning("No data available for drift monitoring")
        st.stop()

    # ======================================================
    # 2. LATEST BATCH DRIFT (GLOBAL DRIFT)
    # ======================================================

    st.subheader("Batch Drift Analysis")

    if not batch_runs.empty:
        latest_batch = batch_runs.iloc[0]

        #Evidently report
        st.subheader("1. Batch Drift (Global Model Drift from Evidently Report)")
        report_path = latest_batch.get("drift_report_path", None)
        if report_path and os.path.exists(report_path):
           
            with open(report_path, "r", encoding="utf-8") as f:
                report_dict = json.load(f)

            # Extract dataset drift info safely
            dataset_metrics = report_dict.get("metrics", [])
            # ---------------------------------------------
            # Feature Drift Details
            # ---------------------------------------------
            rows = []
            for metric in dataset_metrics:
                # Skip the dataset summary metric
                if metric.get("config", {}).get("type") != "evidently:metric_v2:ValueDrift":
                    continue
                config = metric["config"]
                feature = config.get("column", "Unknown")
                method = config.get("method", "")
                threshold = config.get("threshold", 0.1)
                score = metric.get("value", 0)
                drift_ratio = score / threshold

                if drift_ratio < 1:
                    status = "🟢 Stable"
                elif drift_ratio < 2:
                    status = "🟡 Mild Drift"
                elif drift_ratio < 4:
                    status = "🟠 Moderate Drift"
                else:
                    status = "🔴 Severe Drift"
                rows.append({
                    "Feature": feature,
                    "Method": method,
                    "Threshold": threshold,
                    "Drift Score": drift_ratio,#round(score, 4),
                    "Status": "Drifted" if drift_ratio > 1 else "Stable",
                    "Severity": status
                })
            df = pd.DataFrame(rows)
            st.subheader("Feature Drift Summary")
            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True
            )

            drifted_df = df[df["Status"] == "Drifted"]
            st.subheader("⚠ Features Requiring Attention")
            if drifted_df.empty:
                st.success("No feature drift detected.")
            else:
                st.dataframe(
                    drifted_df.sort_values("Drift Score", ascending=False),
                    use_container_width=True,
                    hide_index=True
                )
            fig = px.bar(
                drifted_df.sort_values("Drift Score"),
                x="Drift Score",
                y="Feature",
                color="Drift Score",
                orientation="h",
                color_continuous_scale="Reds",
                text="Drift Score"
            )
            fig.update_layout(
                title="Drifted Features",
                height=450,
                yaxis_title="",
                xaxis_title="Drift Score"
            )
            st.plotly_chart(fig, use_container_width=True)
            status_counts = df["Status"].value_counts().reset_index()
            status_counts.columns = ["Status", "Count"]
            fig = px.pie(
                status_counts,
                names="Status",
                values="Count",
                color="Status",
                color_discrete_map={
                    "Stable":"green",
                    "Drifted":"red"
                }
            )
            fig.update_layout(title="Feature Health")
            st.plotly_chart(fig, use_container_width=True)
            st.subheader("Top 5 Most Drifted Features")
            top5 = df.sort_values(
                "Drift Score",
                ascending=False
            ).head(5)
            st.table(
                top5[
                    ["Feature","Drift Score","Severity"]
                ]
            )
            high = df[df["Status"] == "Drifted"]
            high = high.sort_values("Drift Score",ascending=False)
            if high.empty:
                st.success("✅ No feature drift detected.")
            else:
                st.warning(
                    f"""
                    **{len(high)} of {len(df)} monitored features have drifted.**
                    The highest drift was observed in **{high.iloc[0]['Feature']}**
                    (score = **{high.iloc[0]['Drift Score']:.3f}**).
                    The most affected features are:
                    • {", ".join(high.sort_values('Drift Score', ascending=False).head(5)["Feature"])}
                    """
                )
        else:
            st.warning("No Evidently report found")

        #Custom diagnostics
        st.subheader("2. Batch Drift (Global Model Drift from Custom diagnostics)")
        batch_diagnostics = latest_batch.get("diagnostics", None)
        if batch_diagnostics:
            anomalies = batch_diagnostics.get("anomalies", [])
            if not anomalies:
                st.success("No batch drift detected")
            else:
                charts.drift_feature_chart(batch_diagnostics)
                #Show rows removed if any due to validation errors
                removed = removed_rows()
                if removed:  
                    # Group rows by batch
                    batches = {}
                    for row in removed:
                        batches.setdefault(row.batch_id, []).append(row.row_index)
                        for batch_id, row_indices in batches.items():
                            row_indices = sorted(row_indices)
                            row_list = ", ".join(map(str, row_indices))
                            if len(row_indices) == 1:
                                st.warning(
                                    f"⚠️ The row **{row_list}** of batch **{batch_id}** "
                                    "was removed due to validation errors."
                                )
                            else:
                                st.warning(
                                    f"⚠️ The rows **{row_list}** of batch **{batch_id}** "
                                    "were removed due to validation errors."
                                )
        else:
            st.success("No batch drift detected")
            
    else:
        st.info("No batch runs available")

    # ======================================================
    # 3. ONLINE DRIFT (REAL-TIME INFERENCE DRIFT)
    # ======================================================
    st.subheader("Online Drift (Real-time Transactions)")
    if not online_df.empty:
        diagnostics_list = online_df["diagnostics"].dropna().tolist()
        if len(diagnostics_list) == 0:
            st.success("No online drift detected")
        else:
            all_anomalies = []
            for d in diagnostics_list:
                if isinstance(d, dict):
                    all_anomalies.extend(d.get("anomalies", []))
            if len(all_anomalies) == 0:
                st.success("No online drift detected")
            else:
                aggregated = {}
                for a in all_anomalies:
                    key = (a.get("anomaly_type"), a.get("feature"))
                    if key not in aggregated:
                        aggregated[key] = {
                            "issue_type": a.get("anomaly_type"),
                            "feature": a.get("feature"),
                            "count": 0,
                            "details": set()
                        }
                    aggregated[key]["count"] += 1
                    if a.get("value") is not None:
                        aggregated[key]["details"].add(str(a["value"]))
                rows = []
                for v in aggregated.values():
                    rows.append({
                        "issue_type": v["issue_type"],
                        "feature": v["feature"],
                        "count": v["count"],
                        "details": ", ".join(list(v["details"]))[:200]
                    })
                charts.drift_feature_chart({"anomalies": all_anomalies})
    else:
       st.info("No online available")     

# ==========================================================
# ALERTS
# ==========================================================
elif page == "Alerts":
    st.title("System Alerts")
    alerts_found = False
    if not batch_runs.empty:
        latest = batch_runs.iloc[0]
        if latest.get("drift_present", False):
            st.error("Drift detected in batch transaction data")
            alerts_found = True
        if latest["roc_auc"] < 0.90:
            st.error(f"ROC-AUC dropped to {latest['roc_auc']:.3f} for batch {latest['batch_id']}")
    
    alerts=[]
    if not predictions.empty:  
        high_conf_fraud = (predictions["probability"] > 0.9).mean()
        if high_conf_fraud > 0.12:#Occurrence of high probability fraud cases are high from all the cases(batch and online)
            alerts.append("Fraud spike detected")
        if "drift_present" in online_df.columns and online_df["drift_present"].any():
            alerts.append(" Drift detected in online transaction data")
        if alerts:
            for a in alerts:
                st.error(a)
        else:
            st.success("No active alerts")