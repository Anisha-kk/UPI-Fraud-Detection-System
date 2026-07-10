# Overview
An end-to-end MLOps project for detecting fraudulent UPI transactions using XGBoost, FastAPI, PostgreSQL, Streamlit, Docker, and GitHub Actions. The project includes model training, online and batch inference, monitoring, drift detection, and CI/CD automation. 

The application uses FastAPI to perform the inference. There are 2 input modes- single input prediction and batch prediction. The prediction and errors/anomalies encountered during detection are logged in a Postgres DB. 

The detection model is trained on a synthetic dataset using XGBoost algorithm.

A monitoring dashboard is created using Streamlit that shows analysis of predictions and anomalies stored in DB, as well as results of custom diagnostics and Evidently drift report via charts and tables.

The project includes a GitHub Actions CI pipeline that runs automated tests, validates the Docker Compose configuration, and builds Docker images which are pushed to Docker Hub.

## Features
- Single and batch UPI fraud prediction using FastAPI
- XGBoost-based fraud detection model
- Custom data validation and diagnostics
- Data drift detection using Evidently
- PostgreSQL-based prediction and monitoring logs
- Streamlit monitoring dashboard
- Dockerized application
- GitHub Actions CI pipeline with automated testing and image builds

## Dataset
A synthetic dataset is created to simulate normal and fraudulent UPI transactions. The dataset is then divided into train,test and production datasets based on chronology. The production data is further modified to simulate data drift. Data varying features such as increase in transactions during festivals,paydays etc are considered during data synthesis.

## Tech stack
Python<br>
FastAPI<br>
Streamlit<br>
Postgres DB<br>
Docker<br>
GitHub Actions<br>

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
Training is done using XGBoost algorithm. Category variables are encoded using one hot encoding. Cyclical features such as month and day of the week are encoded using sine transformations.<br>
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
Shows prediction reports, drift analysis, batch and online inference results etc. (Please refer to the Screenshots below)

### Postgres DB
There are 4 tables in the DB:<br>
1. **Prediction**<br>
Stores the transaction details, predicted class with probability, whether the prediction is correct or not (for batch predictions where true label may be available) latency, diagnostics report (data anomalies such as empty values, negative values, invalid inputs etc are added). and whether drift is present or not.<br>
2. **BatchRun**<br>
Stores details for each batch such as batch id, latency, metrics such as roc_auc,pr_auc,accuracy,confusion matrix, diagnostics result for batch, drift present or not and path to drift report created by evidently. It is stored in reports/. <br>
3. **Validation Error**<br>
Stores the transaction input and pydantic validation errors, if any, that arose during single input inference of the specified transaction.<br>
4. **RemovedBatchRow**<br>
In case of batch inference, during custom diagnostics, rows with missing or invalid data are deleted. The information about these deleted rows are stored in this table, along with the row index(in the original csv file), batch id etc.

## CI/CD Pipeline

This project uses GitHub Actions to automate testing and deployment.

On every push and pull request:

- Install project dependencies
- Start a PostgreSQL service
- Initialize the database schema
- Execute the Pytest test suite
- Validate the Docker Compose configuration
- Build the Docker image

On every push to the `main` branch:

- Build the production Docker image
- Tag the image with `latest` and the Git commit SHA
- Push the image to Docker Hub

## Local Execution commands
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
8. Docker compose commands:<br>
To locally create docker image:<br>
docker compose up --build<br>
To stop and remove the containers, networks, and other resources created by docker compose up <br>
docker compose down

## Development Setup using Docker
### Local Development
Clone the repository:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

Build the Docker image and start all services:

```bash
docker compose up --build
```

This command:

- Builds the application image from the local source code
- Starts the FastAPI service
- Starts the Streamlit dashboard
- Starts a PostgreSQL database

Access the application:

| Service | URL |
|---------|-----|
| FastAPI | http://localhost:8000 |
| Swagger UI | http://localhost:8000/docs |
| Streamlit Dashboard | http://localhost:8501 |

To stop the application:

```bash
docker compose down
```

### Production Deployment
To run the published Docker image from Docker Hub without building locally:

Clone the repository:

```bash
git clone https://github.com/<username>/<repo-name>.git
cd <repo-name>
```

Start the application:

```bash
docker compose -f docker-compose.prod.yml up
```

Docker Compose automatically downloads the latest application image from Docker Hub if it is not already available locally.

To stop the application:

```bash
docker compose -f docker-compose.prod.yml down
```

## Result Screenshots

### Endpoints

<br><img width="1920" height="1080" alt="Screenshot (5035)" src="https://github.com/user-attachments/assets/c5d8f0ef-f8ea-448a-b42e-6a2261d27d41" />
<br> Screenshot 1 - Endpoints<br>


<br><img width="1920" height="1080" alt="Screenshot (5036)" src="https://github.com/user-attachments/assets/48e12140-2f55-4b2c-b0e6-057f257472c9" />
<br> Screenshot 2 - /predict endpoint input<br>


<br><img width="1920" height="1080" alt="Screenshot (5037)" src="https://github.com/user-attachments/assets/dca30a69-b4d4-4030-b9eb-e9285dc783b0" />
<br> Screenshot 3 - /predict output<br>


<br><img width="1920" height="1080" alt="Screenshot (5038)" src="https://github.com/user-attachments/assets/fcf47bb7-9358-4bed-89e8-cf2d96954168" />
<br> Screenshot 4 - /batch_predict endpoint<br>


<br><img width="1920" height="1080" alt="Screenshot (5066)" src="https://github.com/user-attachments/assets/5b9a3504-05b1-4371-a55f-b06b5d6c32b2" />
<br> Screenshot 5 - Validation Error at /predict<br>


### Monitor dashboard
<img width="1920" height="1080" alt="Screenshot (5039)" src="https://github.com/user-attachments/assets/adc1e305-f01a-4d97-8fd9-c87089d933ab" />

<img width="1920" height="1080" alt="Screenshot (5040)" src="https://github.com/user-attachments/assets/e3a0f49d-6290-4310-b1f6-a28363b47aed" />

<img width="1920" height="1080" alt="Screenshot (5041)" src="https://github.com/user-attachments/assets/6376dceb-fc96-4ff3-a052-7812e8f5496b" />

<img width="1920" height="1080" alt="Screenshot (5042)" src="https://github.com/user-attachments/assets/3eb37395-d60b-4014-8bc1-2f58a3cbe0e0" />

<img width="1920" height="1080" alt="Screenshot (5044)" src="https://github.com/user-attachments/assets/ce282a00-6133-4177-bcfc-c45c7d19cede" />

<img width="1920" height="1080" alt="Screenshot (5045)" src="https://github.com/user-attachments/assets/14a61c28-f1f4-4603-bc63-8e0a8b15b274" />

<img width="1920" height="1080" alt="Screenshot (5046)" src="https://github.com/user-attachments/assets/7af729c7-e7c0-4f8b-9f06-e4afde6ace4b" />

<img width="1920" height="1080" alt="Screenshot (5047)" src="https://github.com/user-attachments/assets/b24ca212-4ced-4253-8144-d93650bb5c98" />

<img width="1920" height="1080" alt="Screenshot (5048)" src="https://github.com/user-attachments/assets/33928d78-72bd-49a3-a9b7-47d45459e851" />

<img width="1920" height="1080" alt="Screenshot (5049)" src="https://github.com/user-attachments/assets/f7ef5bf4-e9dd-4503-b019-ba8bf7985670" />

<img width="1920" height="1080" alt="Screenshot (5050)" src="https://github.com/user-attachments/assets/a891e05e-1737-4af5-8de3-32f3e6d7117f" />

<img width="1920" height="1080" alt="Screenshot (5051)" src="https://github.com/user-attachments/assets/6c0e9286-8ffa-48a5-bc2e-494bcaaa6b3d" />

<img width="1920" height="1080" alt="Screenshot (5052)" src="https://github.com/user-attachments/assets/b2a7585a-824c-4516-a6a2-826b5c27de6c" />

<img width="1920" height="1080" alt="Screenshot (5053)" src="https://github.com/user-attachments/assets/98e22df6-fd43-47ef-a581-9b5d4acb504f" />

<img width="1920" height="1080" alt="Screenshot (5056)" src="https://github.com/user-attachments/assets/d2e07a53-07a6-40cd-9875-252411287a5c" />

<img width="1920" height="1080" alt="Screenshot (5057)" src="https://github.com/user-attachments/assets/9a3872a0-b8b0-4c79-a03f-c911868e5c8c" />

<img width="1920" height="1080" alt="Screenshot (5058)" src="https://github.com/user-attachments/assets/aeb7530e-8873-4cb5-a934-734774acfbd3" />

<img width="1920" height="1080" alt="Screenshot (5059)" src="https://github.com/user-attachments/assets/5f0ddcf7-4471-4792-9707-afe0fed0ae73" />

<img width="1920" height="1080" alt="Screenshot (5060)" src="https://github.com/user-attachments/assets/a217f19a-dfa4-42d0-b34c-0711237ccf8b" />

<img width="1920" height="1080" alt="Screenshot (5062)" src="https://github.com/user-attachments/assets/c9529855-d404-43a0-8177-f7e52693d13a" />

<img width="1920" height="1080" alt="Screenshot (5063)" src="https://github.com/user-attachments/assets/70b8a287-2154-482c-830e-4d4c62fbf3eb" />

<img width="1920" height="1080" alt="Screenshot (5064)" src="https://github.com/user-attachments/assets/c977f456-a65f-46ec-8cc7-cf9550bd31e8" />

<img width="1920" height="1080" alt="Screenshot (5065)" src="https://github.com/user-attachments/assets/6f87020a-b5e9-4f09-95a9-435e6c789af6" />

## Future Scope
- Automate deployment by pulling versioned Docker images from Docker Hub to a production environment.
- Add continuous deployment after successful CI pipeline execution.
- Integrate a managed PostgreSQL database for production scalability.
- Add advanced monitoring and alerting for model performance and data drift.
- Implement model retraining pipelines as new transaction data becomes available.
