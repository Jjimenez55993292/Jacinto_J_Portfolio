# Jacinto_Jimenez_Portfolio
### Data Science, AL/ML and Data Engineering Portfolio

Hello, my name is Jacinto Jimenez, and welcome to my portfolio showcasing my work in Data Science, Data Engineering, and Artificial Intelligence (AI) & Machine Learning (ML). I am deeply passionate about utilizing data and intelligent systems to answer complex questions and create meaningful solutions. AI and ML allow us to uncover patterns, automate processes, and generate predictions that drive real-world impact. In this portfolio, I have included a selection of projects developed during my academic studies that demonstrate my skills in data analysis, modeling, and AI/ML integration. These projects incorporate various Python and R libraries, data engineering tools, and analytical techniques to build predictive and data-driven models. Thank you for visiting my portfolio.

# Data Analyst & Data Engineering 
# [Project 1: CMS-64 Federal Reporting Explorer](https://github.com/Jjimenez55993292/cms64-federal-reporting-explorer)
🌐 **Live Demo:**
[Dashboard App](https://cms64-federal-reporting-explorer.fly.dev/)
## Scenario:
I built a small Medicaid EDW-style pipeline and dashboard that mirrors the work of a CMS-64/21 federal reporting data analyst, using the official CMS Medicaid Financial Management – National Totals (MBES/CBES) dataset as the source. An ETL pipeline (data_pipeline.py) downloads the CSV from medicaid.gov, cleans types and whitespace, and loads it into a SQLite warehouse (cms_edw.db) with indexes on Year, Program, and Service Category. On top of that, a Plotly Dash app (app.py) provides an Overview view for trends in Total Computable, Federal Share, and State Share by year (with YoY change), a By Service Category view that shows CMS-64-style spend by service line (top 20 categories), a By Program view that compares Medicaid, Admin, CHIP, etc., and a Data Quality view that checks Total Computable ≈ Federal + State, flags negative values and missing Service Categories, and plots the differences. The stack is Python, pandas, SQLite, Plotly Dash, Docker, and Fly.io.
This project shows I can:
- Work with **Medicaid/CMS-64-style data**
- Build and query a **mini EDW** for federal reporting
- Implement **data quality and reconciliation rules**
- Deliver a **deployed interactive dashboard** that supports reporting and QA.
<img width="1445" height="867" alt="dashboard" src="https://github.com/user-attachments/assets/3c80cce7-88c2-46e8-bd9d-9f9ff7f005fc"/>
<img alt="dashboard" src="https://github.com/user-attachments/assets/3c80cce7-88c2-46e8-bd9d-9f9ff7f005fc" width="100%"/>



# Data Analyst & Data Engineering  
# [Project 2: openFDA Safety Explorer](https://github.com/Jjimenez55993292/dashboard-openfda)
🌐 **Live Demo:**  
[openFDA Safety Explorer Dashboard](https://dashboard-openfda.fly.dev/)
## Scenario:
I built an interactive safety monitoring dashboard on top of the public [openFDA Drug Event API](https://open.fda.gov/apis/drug/event/) to simulate the kind of exploratory analysis a pharmacovigilance or drug safety analyst would do. The app lets users query real FDA adverse event data by lookback window or full calendar year, filter by drug name or reaction term, and then explores the results with summary metrics, trend lines, and reaction frequency charts.
Under the hood, the app builds parameterized API queries to openFDA, handles paging through JSON results, normalizes the nested event/reaction structure into tidy `pandas` DataFrames, and calculates metrics like total reports, unique drugs, and top reactions. A Plotly Dash front end (app.py) then turns this into an interactive dashboard with time series for events over time, a bar chart of top reaction terms, and a sample table of individual case reports. The app is containerized with Docker and deployed on Fly.io behind Gunicorn.
The stack is Python, `requests`, `pandas`, Plotly Dash, Docker, and Fly.io.

This project shows I can:
- Work with **real-world healthcare / pharmacovigilance data** from the openFDA Drug Event API  
- Design and implement **API-driven data pipelines** (query, paginate, normalize nested JSON)  
- Build **interactive analytical dashboards** with filters, charts, and tabular drill-downs  
- Apply **data profiling and summarization** to make large event datasets interpretable  
- Package and deploy a **production-style Dash app** using Docker, Gunicorn, and Fly.io  
<img width="3759" height="1795" alt="image" src="https://github.com/user-attachments/assets/f12ed8ee-8511-4cb3-b3c8-e230f5b9770b" />



# AI & Data Engineering 
# [Project 3: A Full-stack AI Pipeline for Medical Data Extraction and Retrieval-Augmented Chatbots](https://github.com/Jjimenez55993292/medextract_full_project)
## Scenario:
### 🩺 MedIntel — AI-Powered Clinical Data Hub 
I built a full medical data processing system that integrates **data engineering pipelines** with **AI-powered document intelligence**.  
The project extracts structured information from unstructured medical prescriptions using **OCR (Tesseract/PyMuPDF)**, performs **data validation and correction** through a **human-in-the-loop Streamlit interface**, and persists cleaned data into a **SQLite database**. The **frontend** allows clinicians to **upload, review, and search patient records**, while the **backend (FastAPI)** orchestrates the complete **ETL workflow** — from file ingestion to database persistence. To enhance usability, I integrated a **Retrieval-Augmented Generation (RAG) chatbot** built with **FAISS** and **sentence-transformers**, capable of answering domain-specific questions using both the stored patient data and uploaded clinical documents. This architecture demonstrates a complete **AI + Data Engineering lifecycle** — covering data ingestion, transformation, indexing, and intelligent retrieval. The modular design supports **pluggable LLMs (OpenAI or local vLLM)**, making the system scalable, reproducible, and extensible for real-world healthcare applications such as **medical record search**, **prescription verification**, and **patient insight generation**.
<img width="975" height="361" alt="image" src="https://github.com/user-attachments/assets/9543fab3-f60f-4205-91fe-dd08f9333a4c" />


# AI Engineering
# [Project 4: Document-Aware RAG Chatbot for Healthcare Support](https://github.com/Jjimenez55993292/support-rag-chatbot)
## Scenario:
### RAG Chatbot for Healthcare Support: 
I built a secure, closed-domain assistant that answers patient and staff questions using a clinic’s own PDFs, DOCX, CSV, and TXT files. Users upload documents in a Streamlit UI; the app parses text (PyMuPDF, docx2txt, pandas), splits it (~1,000 chars, 100 overlap), creates embeddings (OpenAIEmbeddings), and indexes them in FAISS. At query time, it runs semantic search (k=5) and uses a Retrieval-Augmented Generation flow with LangChain + GPT to produce grounded, empathetic answers, showing the source snippets for transparency. An optional DistilBERT intent classifier routes technical vs. general queries. Ops follow best practices (env-based secrets, local index persistence; cloud-ready). Result: faster, more accurate intake/FAQ responses, reduced staff load, and a clear path to scale (hybrid search, metadata filters) with strong safety controls (PII scrubbing, RBAC, audit logs).
[Read Full Report](https://github.com/Jjimenez55993292/support-rag-chatbot/blob/main/Report.pdf)
<img width="975" height="461" alt="image" src="https://github.com/user-attachments/assets/5726844e-db5a-4a9b-b31a-cb143f07fbf9" />


# Data Engineering
# [Project 5: Earthquake Data Engineering Project](https://github.com/Jjimenez55993292/EarthquakeDataEngineering)
## Scenario:
### Earthquake Data Pipeline: 
This project aims to ingest and process earthquake data from the USGS, create a reliable, scalable, and cost-effective data pipeline, and make it available for analysis and visualization. To achieve this, a range of open-source technologies such as Terraform, Airflow, Docker, GCS, BigQuery, dbt, and Looker Studio were utilized. The data is acquired through a public REST API provided by USGS, and the seismic events data is stored in a denormalized format using the One Big Table (OBT) method. The project is implemented as an Airflow DAG that runs daily, parameterizing the API endpoint to use the dates passed in through Airflow template variables. The seismic events data is then transformed and materialized as a view and an incremental table using dbt, with a built-in tool to visually display data lineage as it travels across different data layers such as raw, stage, and final. Finally, the data is made accessible for analysis and visualization in Looker Studio using BigQuery as the data source. The project is designed to be automated, reliable, scalable, and affordable, ensuring that it can be easily maintained and expanded over time. See the attached image for further details.

![image](https://user-images.githubusercontent.com/79177516/219904889-225158d2-6afb-4aa2-a1e4-28abfd44ac23.png)



# [Project 6: BERT Natural Language Processing (NLP) Model - Python](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/BERTModel_HomeWork.html)
## Scenario: 
### Store Logistics Accuracy Modeling: 
In this project, we aim to construct a learning-based entity extraction model, specifically a custom BERT model, which will extract the store number from the transaction descriptor. The dataset is divided into three sections: train, validate, and test, and we utilize Python libraries such as NumPy, Keras, TensorFlow, and Pandas. The project will involve preprocessing the data to make it more usable, as well as training and developing an algorithm model using sequence-based datasets.
For those unfamiliar with BERT, it is a group of Transformers encoders stacked on top of each other. In more technical terms, BERT is a precise, huge transformer masked language model. Models are the output of an algorithm run on data, which includes the procedures used to make predictions on data. Once the custom part model is developed, it can be implemented in real-life examples given the data provided. Overall, this project showcases the use of cutting-edge NLP techniques and advanced machine learning algorithms to solve complex data extraction problems.

### [Link - Data Set for Model ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Example.Dataset.Homework.html)

![image](https://user-images.githubusercontent.com/79177516/180825125-298cec42-47ab-4d9d-b5eb-0178e2c27562.png)

# ML Engineering
# [🫀 Project 7: CardioGuard — End-to-End Cardiovascular Disease Risk Predictor](https://github.com/Jjimenez55993292/cardio_vasicular_disease_detector)
## Scenario: 
## A complete machine learning pipeline with API deployment, Streamlit UI, and cloud hosting
I built a full ML system that predicts cardiovascular disease risk using structured patient data such as age, blood pressure, cholesterol levels, and lifestyle indicators. This project includes the entire ML lifecycle — from data preparation and EDA, to model training, evaluation, deployment, and a polished UI for real-world usage. The system is powered by a Gradient Boosting classifier, trained on the 70,000-row Cardiovascular Disease dataset from Kaggle. Features are validated, encoded with a DictVectorizer, passed through the model, and returned as both a binary risk prediction and a probability score. A FastAPI backend exposes a public /predict endpoint, containerized with Docker, and deployed to Fly.io. A separate Streamlit web interface lets users interactively test patient scenarios with sliders and see risk levels instantly. This modular design makes the system reproducible, scalable, and deployable for future healthcare ML applications.
<img width="872" height="797" alt="image" src="https://github.com/user-attachments/assets/798e969f-a31d-4ea0-a297-f24d9c792d3e" />
🚀 **Live API Endpoint:**
[CVD Detector API](https://cvd-detector-main-v2.fly.dev/)
<img width="1582" height="770" alt="image" src="https://github.com/user-attachments/assets/5606ca5c-d5f6-4d6b-bbe9-baefec60d224" />
🌐 **Streamlit Web App:**
[CVD Streamlit App](https://cvd-streamlit-app.fly.dev/)

# Data Science

# [Project 8: Automated Machine Learning Techniques (EvalML)](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/HeartAttackRiskPredictor.html)
## Scenario:
### Heart Attack Risk Predictor Modeling: 
The primary objective of this project is to develop an automated machine learning model to predict the risk of heart attacks. The dataset includes various attributes that are crucial for detecting heart diseases. We will compare the performance of our model with other conventional machine learning models such as Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbor, and SVM. The automated machine learning process involves using AutoML to automate the tasks of applying machine learning to real-world problems. We will utilize the EvalML library to automate a more significant part of the machine learning process. EvalML enables us to evaluate which machine learning pipeline works best for the given dataset. By comparing the performance of different models, we aim to build a more efficient and accurate heart attack risk prediction model.

![image](https://user-images.githubusercontent.com/79177516/165724113-5a4ebe24-9eb0-4546-bf4d-3cb7295cf050.png)



# [Project 9: Learning (Long Short-term Memory) LSTM Model](https://jjimenez55993292.github.io/Deep-Learning-LSTM-model/PredictingNextWordInASentence.html)
## Scenario:
### LSTM Model - Predicting The Next Word In A Sentence: 
In this project, we will be building and deploying data science, machine learning, and deep learning models. Our tools will include Python libraries such as NumPy, pickle, Keras, TensorFlow, and Pandas, as well as Django, GCP, and Heroku Cloud. The focus of this project will be on a text dataset, specifically a book. We will preprocess the data to ensure it is in a more usable format and use sequence-based datasets to train and develop an algorithm model. Our choice of model for this project will be the Deep Learning (Long Short-term Memory) LSTM model. The LSTM model is a neural net-architecture that performs exceptionally well on sequence-based datasets. Its feedback structure enables the model to remember the sequence of data input and the changes in the output, making it possible to predict the next word in a sentence accurately.

![image](https://user-images.githubusercontent.com/79177516/163972466-97233b06-1bf2-4d0a-a2fd-d57d815ce0df.png)


## Implementation:
### Predicting The Next Word In A Setence Website: 
### [Website: The Next Word In A Setence Website - Link ](http://jacintojimenez606.pythonanywhere.com/home/)

We utilized Django, an open-source web application framework written in Python, to integrate our developed Python model for predicting the next word into our website. It is important to note that while our current model has a prediction accuracy of 27%, increasing the size of our training dataset can improve the model's performance. We successfully implemented the model on our website and created a functional prototype, which is ready for testing and further improvement. This project's final stage involves deploying the website on the Django platform, making it accessible to users worldwide. Once completed, users will be able to run the model and access its predictions from anywhere with internet access.


### [GitHub-Link ](https://github.com/Jjimenez55993292/Deep-Learning-LSTM-model)


![image](https://user-images.githubusercontent.com/79177516/163972673-bf0c7181-7694-4622-9217-bf885a0db423.png)



# [Project 10: Logistic Regression - Pyhton - Sentiment Analysis ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Sentiment%20Analysis%20Using%20Logistic%20Regression%20Model.html)

## Scenario: 
###  Sentiment Analysis Modeling: 

In this project, we focus on developing a logistic regression model for sentiment analysis using data science techniques. Python libraries (NumPy, pickle, Keras, TensorFlow, and Pandas) and Django web framework will be utilized in the development process. Our dataset consists of tweets which will be preprocessed to make it more usable for analysis. We will be using a sequence-based dataset to train and develop an algorithm model for sentiment analysis. Sentiment analysis, also known as opinion mining, is a natural language processing technique used to determine the sentiment (positive, negative or neutral) of the given data. Developing models for sentiment analysis is crucial in understanding customer sentiment for a specific scenario. We will start by using Python to build features and a logistic regression model. Firstly, we will closely examine the textured data and extract the necessary features from the dataset. Our logistic regression model will be developed to predict the sentiment of the given data. With the developed model, we can effectively analyze and classify the sentiment of tweets into positive, negative, or neutral categories. The implementation of such a model can be useful in a wide range of fields, including social media analytics and marketing research. With the help of the Django web framework, we will be able to deploy and host the model on a website to make it accessible to a larger audience.

![image](https://user-images.githubusercontent.com/79177516/164423025-d5c8c139-aeca-49de-8c80-7f070375b553.png)

## Implementation:
### Sentiment Analysis Using Logistic Regression Model Website: 
### [Website:Sentiment Analysis Sentiment Analyzer Website - Link ](https://jimenez55993292.pythonanywhere.com/polls/)

In this project, we have built and deployed a data science model for sentiment analysis using Django, an open-source web application framework written in Python. Sentiment analysis, also known as opinion mining, is a natural language processing technique used to determine whether data is positive, negative, or neutral. Our logistic regression model has achieved an impressive accuracy rate of 99.50 percent. With the functions that were developed, we can predict the sentiment of a string input through our website. The final step of this project involved hosting the newly built website on the Django platform, allowing us to access the website and run the model from anywhere across the internet. This project showcases the importance of developing specific models for specific scenarios in order to effectively analyze customer sentiment, and highlights the power of Python libraries such as NumPy, pickle, Keras, TensorFlow, and Pandas in building accurate and efficient machine learning models.

![image](https://user-images.githubusercontent.com/79177516/164618063-9625d13e-8476-4318-a276-9ab4814bd2c0.png)



# [Project 11: Logistic Regression](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/project_4.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, we aim to create a logistic regression model in R to investigate the dataset and answer questions related to credit default risk. The dataset provided is from a credit card company, containing historical data of customer characteristics and their likelihood of defaulting on credit. It is crucial for the company to assess the risk of credit default in their customers and take proactive measures to mitigate the risk. Using logistic regression, we can analyze the relationship between various customer characteristics and their probability of defaulting on credit, enabling the company to make data-driven decisions and optimize their risk management strategies.

![image](https://user-images.githubusercontent.com/79177516/137412784-2fe2bd4f-e615-41f4-857c-7095df391b34.png)




# [Project 10: Decision Trees](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Three.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, our aim is to analyze the historical data provided by a credit card company to identify the relationships between customer characteristics and the likelihood of credit default. To achieve this, we use Decision Tree models which are a powerful tool for analyzing and classifying data. The Decision Tree algorithm works by recursively splitting the data based on the features that best discriminate between the classes. We will train our model on the historical data and use it to predict the probability of default for new customers. By accurately assessing the risk of credit default, the credit card company can take proactive measures to minimize their financial losses.

![image](https://user-images.githubusercontent.com/79177516/137412692-3a8c369e-022d-498a-8b24-a612f6f6b46f.png)




# [Project 11: Logistic Regression and Random Forests](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Two.html)
## Scenario:
### Heart Disease Modeling: 
The University Hospital is conducting a research on the risk factors associated with heart disease. They have access to a large set of historical data that contains information on different health indicators, such as fasting blood sugar and maximum heart rate, which can be used to analyze patterns and identify the presence of heart disease. The project aims to create multiple logistic regression models to predict the likelihood of an individual being at risk of heart disease. Such models can prove to be highly valuable in evaluating medical records and identifying hidden risks that may not be apparent to human doctors. In addition, the project also involves creating a classification random forest model to predict the risk of heart disease and a regression random forest model to predict the maximum heart rate achieved. The project is aimed at developing various models to analyze the Heart Disease dataset using Logistic Regression and Random Forests.

### [GitHub - Jupyter Notes ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Logistic_Regression_%20jupyter_notes_2.html)

![image](https://user-images.githubusercontent.com/79177516/137412663-55e2d96f-9453-4a3f-a1b6-2f164cd143ab.png)

