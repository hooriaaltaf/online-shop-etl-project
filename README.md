# Online Shop ETL Project

An end-to-end data engineering and machine learning project for an
online shop. The pipeline extracts data from relational and document
databases and a weather API, transforms and validates the data, builds a
dimensional warehouse, processes Delta tables with Apache Spark, and
trains a customer churn classification model tracked with MLflow.

## Project objectives

-   Extract customer and order data from PostgreSQL.
-   Extract mobile-app order documents from MongoDB.
-   Retrieve weather data from the Open-Meteo API for Lahore, Karachi,
    and Islamabad.
-   Clean, standardize, combine, and validate data.
-   Build a dimensional data warehouse using a star-schema approach and
    Type 2 customer history.
-   Store and update warehouse tables in Databricks Delta Lake.
-   Create aggregated Gold-layer tables using Spark.
-   Build and evaluate a customer churn model and track experiments with
    MLflow.

## Repository structure

``` text
Online-Shop-ETL-Project/
├── notebooks/
│   ├── 01_data_extraction.ipynb
│   ├── 02_data_transformation.ipynb
│   ├── 03_data_warehouse.ipynb
│   ├── 04_delta_tables.ipynb
│   ├── 05_spark_processing.ipynb
│   └── 06_churn_prediction.ipynb
├── data/
│    ├──  raw/ 
│    ├──  processed/    
│    ├──  warehouse/                            
├── .env                    
├── .gitignore
└── README.md
```

## Data sources

  --------------------------------------------------------------------------
  Source                Data                                   Project scope
  --------------------- ----------------------- ----------------------------
  PostgreSQL            Customers and orders                2,000 customers;
                                                approximately 200,000 orders

  MongoDB               Mobile application                   5,000 documents
                        order documents         

  Open-Meteo API        Weather                         Lahore, Karachi, and
                        observations/forecast                      Islamabad
                        data                    
  --------------------------------------------------------------------------


### Environment variables

Create a local `.env` file in the project root with the connection
variables expected by the extraction notebook:

``` dotenv
PG_URL=your_postgresql_connection_string
MONGO_URL=your_mongodb_connection_string
```


## Technology stack

-   Python and Jupyter notebooks
-   pandas and PySpark
-   PostgreSQL and SQLAlchemy
-   MongoDB and PyMongo
-   Open-Meteo API and `requests`
-   Databricks, Apache Spark, and Delta Lake
-   MLflow for experiment tracking and model loading
-   Git and GitHub

## Setup

### 1. Clone the repository

``` bash
git clone https://github.com/hooriaaltaf/online-shop-etl-project.git
cd online-shop-etl-project
```

### 2. Create and activate a virtual environment

Windows PowerShell:

``` powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

Git Bash:

``` bash
python -m venv .venv
source .venv/Scripts/activate
```

### 3. Install dependencies

``` bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Configure credentials

Create the `.env` file as described above. Confirm `.env` is listed in
`.gitignore` before committing or pushing changes.

### 5. Run the local notebooks

Open the project in VS Code or Jupyter and run the notebooks in order:

1.  `01_data_extraction.ipynb`
2.  `02_data_transformation.ipynb`
3.  `03_data_warehouse.ipynb`

The Databricks notebooks are then run in the Databricks workspace using
the prepared warehouse files and Delta tables.

> **Important:** The project seed script can recreate or replace source
> tables. Do not rerun it against an existing populated environment
> unless you intentionally want to reseed the data.

## ETL workflow

### 1. Extraction --- Notebook 01

-   Connects to PostgreSQL and extracts customer and order data.
-   Reads order data in chunks to handle the larger orders table.
-   Retrieves MongoDB mobile-app order documents.
-   Calls the Open-Meteo API for the three project cities.
-   Saves extracted data for downstream processing.

### 2. Transformation --- Notebook 02

-   Cleans and standardizes source datasets.
-   Normalizes nested MongoDB order data for tabular processing.
-   Handles data types and missing or inconsistent values.
-   Combines data from the different sources.
-   Performs data-quality checks before warehouse loading.

### 3. Data warehouse --- Notebook 03

-   Builds a dimensional model using fact and dimension tables.
-   Maintains customer history using Slowly Changing Dimension Type 2
    (SCD Type 2).
-   Prepares the fact-order dataset and dimension keys for analytical
    queries.

### 4. Delta Lake --- Notebook 04

The Databricks catalog and schema used are `workspace.default`.

Created Delta tables:

  Table                 Rows
  ---------------- ---------
  `dim_customer`       2,057
  `dim_date`             730
  `dim_product`            6
  `dim_payment`            4
  `dim_status`             5
  `fact_orders`      199,989

The notebook also loads an incremental order extract containing 8,593
records and demonstrates Delta `MERGE` and table history/time-travel
functionality. The 8,593 incoming order IDs matched existing fact-order
IDs in the tested dataset, so the demonstrated merge updated matching
records rather than inserting new order IDs.

### 5. Spark processing --- Notebook 05

The notebook joins the warehouse tables and creates three Gold-layer
analytical tables:

  Gold table                      Rows
  ---------------------------- -------
  `gold_sales_summary`              22
  `gold_customer_summary`        2,057
  `gold_product_performance`         6

These tables provide aggregated views for sales, customer, and product
analysis.

### 6. Churn prediction --- Notebook 06

The churn model uses a cutoff date of **2026-08-01**. A customer is
labelled as churned (`1`) when they place no order during **2026-08-01
through 2026-10-24**; otherwise the label is `0`.

Customer features are calculated from historical orders before the
cutoff:

-   `total_orders`
-   `total_spent`
-   `average_order_value`
-   `total_quantity`
-   `recency_days`

The feature snapshot is saved as the Delta table:

``` text
workspace.default.churn_features
```

The recorded feature Delta version is **1**.

Three Logistic Regression configurations were evaluated:

  Model       `regParam`   `maxIter`
  --------- ------------ -----------
  Model 1           0.01          50
  Model 2            0.1          50
  Model 3            1.0          50

The selected Model 1 run was logged in MLflow. Its run ID is:

``` text
34366b9a526b4d909c7b8ae2773e04f3
```

The project experiment is named `/Shared/Online-Shop-Churn-Prediction`.
During development, runs were also created under the notebook's personal
experiment. When presenting the experiment, use the run comparison that
includes the three distinct `regParam` settings; avoid counting
duplicate Model 1 runs as additional configurations.

#### Evaluation results

The test set contained 359 records: 358 non-churned customers and only 1
churned customer. The observed weighted metrics were:

  Metric                          Result
  -------------------- -----------------
  Accuracy               0.9972 (99.72%)
  Weighted F1            0.9958 (99.58%)
  Weighted precision     0.9944 (99.44%)
  Weighted recall        0.9972 (99.72%)

**Important model limitation:** The confusion matrix showed that the
model predicted all 359 test customers as not churned. It missed the one
churned customer, so churn-class recall was 0%. The high accuracy and
weighted metrics are driven by the severe class imbalance (1,998
non-churned versus 2 churned customers overall). These results should
not be presented as evidence that the model reliably detects churn. More
churn examples and imbalance-aware evaluation/training would be needed
for a useful churn detector.

The selected model was reloaded from MLflow by run ID and used to score
five customers.

## Running in Databricks

The Databricks notebooks are:

-   `04_delta_tables`
-   `05_spark_processing`
-   `06_churn_prediction`

The project workspace used the catalog/schema `workspace.default` and a
Unity Catalog volume at:

``` text
/Volumes/workspace/default/online_shop_files/
```

The MLflow Spark model temporary directory was configured under that
volume:

``` text
/Volumes/workspace/default/online_shop_files/mlflow_tmp
```

Run the Databricks notebooks in sequence after the warehouse data has
been prepared and uploaded to the volume. Notebook variables are not
automatically shared across separate notebooks; each notebook should
load the Delta tables it needs.

## Results summary

-   Extracted data from PostgreSQL, MongoDB, and Open-Meteo.
-   Prepared a dimensional warehouse with SCD Type 2 customer history.
-   Created six Delta warehouse tables.
-   Demonstrated incremental Delta processing and table history.
-   Created three Spark Gold tables.
-   Built three Logistic Regression configurations and tracked
    metrics/parameters with MLflow.
-   Saved a versioned feature table, reloaded the selected model, and
    scored five customers.
-   Identified severe class imbalance as a limitation of the current
    churn model.

## Security and data handling

-   Keep `.env`, database credentials, API secrets, and private exports
    out of Git.
-   Do not paste connection strings or passwords into notebooks, README
    files, screenshots, or public issue trackers.
-   Review `.gitignore` before every push.
-   Use synthetic or approved project data in public screenshots and
    demonstrations.

GitHub repository:
[hooriaaltaf/online-shop-etl-project](https://github.com/hooriaaltaf/online-shop-etl-project)
