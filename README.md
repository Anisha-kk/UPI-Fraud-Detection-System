# Overview
This application can be used to detect fraudulent UPI transactions.<br>
The application uses FastAPI to perform the inference. There are 2 input modes- single input prediction and batch prediction. The prediction and errors/anomalies encounterd during detection are logged in a Postgres DB. <br>
The detection model is trained on a synthetic dataset using XGBoost algorithm.<br>
A monitoring dashboard is created using Streamlit that shows analysis of predictions and anomalies stored in DB, as well as results of custom diagnostics and Evidently drift report via charts and tables.<br>
The application is tested and automatically built into a Docker image using GitHub Actions.

## Dataset
A synthetic dataset is created to simulate normal and fraudulent UPI transactions. The datset is then divided into train,test and production datasets based on chronology. The production data is further modified to simulate data drift. Data varying features such as increase in transactions during festivals,paydays etc are considered during data syntheses.

## Tech stack
Python
FastAPI
Streamlit
Postgres DB
Docker
GitHub Actions

## Project Architecture

Training Pipeline  
      │  
      ▼  
Train Model  
      │  
      ▼  
Save Model Artifacts  
      │  
      ▼  
FastAPI Loads Model  
      │  
      ▼  
Single/Batch Predictions  
      │  
      ▼  
Store Predictions + Errors + Anomalies   
      │  
      ▼  
Save Monitoring Results to PostgreSQL  
      │   
      ▼  
Streamlit Dashboard Reads Database  
      │<── Evidently Reports  
      ▼  
Visual Monitoring & Alerts  

### Model training
Training is done using XGBoost algorithm.<br>
The training dataset is highly imbalanced - about 93% of transactions in the dataset are of normal transactions and 7% of fraud transactions. The class imbalance is handled during training using scale_pos_weight parameter of the XGBoost class.<br>
The model pipeline is stored as .pkl file. The threshold used for prediction is also stored in a .pkl file. The model is tested on test data and following are the resultant metrics:<br>
**Classification Report**

|CLASS     | Precision | Recall | 
|----------|-----------|--------|
|0 (Normal)| 0.98      |  0.99  |
|1 (Fraud) | 0.82      |  0.80  | 

The recall for fraud transactions is 0.80, meaning the model correctly identifies 80% of fraudulent cases.

In an imbalanced classification problem, this is generally a strong result because the model is able to capture most of the minority class (fraud cases), which is typically the primary objective.

|ROC_AUC | PR_AUC | Threshold | Accuracy|
|--------|--------|-----------|---------|
|0.96    | 0.88   |  0.79      |0.97    |

The model shows strong performance in separating fraud from normal transactions (ROC-AUC = 0.96). The PR-AUC of 0.88 confirms good performance under class imbalance. However, the relatively high threshold (0.79) suggests the model is tuned to reduce false positives, potentially at the cost of missing some fraud cases. The high accuracy (0.97) is less meaningful due to class imbalance.

**Confusion matrix**
|Pred_Normal |Pred_Fraud|
|------------|----------|
|30356       |425       |
|507         |1990      |

The model is effective at identifying fraud with good precision and recall (~80%). However, it still misses a non-trivial number of fraud cases (507 false negatives), which may need further optimization depending on business risk tolerance.

### FastAPI endpoints
There are 2 endpoints:<br>
1. **/predict**<br>
Accepts a single transaction as input<br>
Returns predicted class with probability<br>
Uses pydantic validation of inputs.<br>
Custom diagnostics on the data is performed before inference to detect invalid data. <br>
Stores the prediction and pydantic validation errors if any in postgres db.<br>
2. **/batch_predict**<br>
Accepts a csv file with multiple transactions as input <br>
Returns Success / Error <br>
Stores the prediction results in DB <br>
Custom diagnostics on the data is performed before inference to detect invalid data. <br>
Evidently reports are created on this data for drift analysis. The result is stored in the DB.<br>
The uploaded CSV file  is saved along with batch id in filename in data/batches folder

### Streamlit Dashboard
Shows prediction reports,drift analysis, batch and online inference results etc. (Please refer to the Screenshots below)

### Postgres DB
There are 4 tables in the DB:<br>
1. **Prediction**<br>
Stores the transaction details, predicted class with probability, whether the prediction is correct or not (for batch predictions where tru label may be available) latency, diagnostics report (data anomalies such as empty values, negative values, invalid inputs etc are added). and whether drift is present or not.<br>
2. **BatchRun**<br>
Stores details for each batch such as batch id, latency, metrics such as roc_auc,pr_auc,accuracy,confusion matrix, diagnostics result for batch, drift present or not and path to drift report created by evidently. It is stored in reports/. <br>
3. **Validation Error**<br>
Stores the transaction input and pydantic validation errors, if any, that arouse during single input inference of the specified transaction.<br>
4. **RemovedBatchRow**<br>
In case of batch inference, during custom diagnostics, rows with missing or invalid data are deleted. The information about these deleted rows are stored in this table, along with the row index(in the original csv file), batch id etc.

## Execution commands
1. data generation:<br>
python -m data.data_generator<br>
2. Data splitter<br>
python -m data.data_splitter<br>
3. Training<br>
python -m training.train<br>
4. FastAPI inference<br>
uvicorn app:app<br>
Local URL: http://127.0.0.1:8000/docs<br>
5. DB table creation<br>
python -m db.init_db<br>
6. Streamlit app<br>
streamlit run monitor_dashboard.py<br>
Local URL: http://localhost:8501<br>
7. pytest :
python -m pytest -v <br>

## Result Screenshots

### Endpoints


### Monitor dashboard

## Future Scope

