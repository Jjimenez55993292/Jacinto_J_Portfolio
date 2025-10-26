# Jacinto_Jimenez_Portfolio
### Data Science, AL/ML and Data Engineering Portfolio

Hello, my name is Jacinto Jimenez, and welcome to my portfolio showcasing my work in Data Science, Data Engineering, and Artificial Intelligence (AI) & Machine Learning (ML). I am deeply passionate about utilizing data and intelligent systems to answer complex questions and create meaningful solutions. AI and ML allow us to uncover patterns, automate processes, and generate predictions that drive real-world impact. In this portfolio, I have included a selection of projects developed during my academic studies that demonstrate my skills in data analysis, modeling, and AI/ML integration. These projects incorporate various Python and R libraries, data engineering tools, and analytical techniques to build predictive and data-driven models. Thank you for visiting my portfolio.

# AI Engineering
# [Project 1: Document-Aware RAG Chatbot for Healthcare Support](https://github.com/Jjimenez55993292/support-rag-chatbot)
## Scenario:
### RAG Chatbot for Healthcare Support: 
I built a secure, closed-domain assistant that answers patient and staff questions using a clinic’s own PDFs, DOCX, CSV, and TXT files. Users upload documents in a Streamlit UI; the app parses text (PyMuPDF, docx2txt, pandas), splits it (~1,000 chars, 100 overlap), creates embeddings (OpenAIEmbeddings), and indexes them in FAISS. At query time it runs semantic search (k=5) and uses a Retrieval-Augmented Generation flow with LangChain + GPT to produce grounded, empathetic answers, showing the source snippets for transparency. An optional DistilBERT intent classifier routes technical vs. general queries. Ops follow best practices (env-based secrets, local index persistence; cloud-ready). Result: faster, more accurate intake/FAQ responses, reduced staff load, and a clear path to scale (hybrid search, metadata filters) with strong safety controls (PII scrubbing, RBAC, audit logs).
[Read Full Report](https://github.com/Jjimenez55993292/support-rag-chatbot/blob/main/Report.pdf)


<img width="975" height="461" alt="image" src="https://github.com/user-attachments/assets/5726844e-db5a-4a9b-b31a-cb143f07fbf9" />


# Data Engineering

# [Project 2: Earthquake Data Engineering Project](https://github.com/Jjimenez55993292/EarthquakeDataEngineering)
## Scenario:
### Earthquake Data Pipeline: 
This project aims to ingest and process earthquake data from the USGS, create a reliable, scalable, and cost-effective data pipeline, and make it available for analysis and visualization. To achieve this, a range of open-source technologies such as Terraform, Airflow, Docker, GCS, BigQuery, dbt, and Looker Studio were utilized. The data is acquired through a public REST API provided by USGS, and the seismic events data is stored in a denormalized format using the One Big Table (OBT) method. The project is implemented as an Airflow DAG that runs daily, parameterizing the API endpoint to use the dates passed in through Airflow template variables. The seismic events data is then transformed and materialized as a view and an incremental table using dbt, with a built-in tool to visually display data lineage as it travels across different data layers such as raw, stage, and final. Finally, the data is made accessible for analysis and visualization in Looker Studio using BigQuery as the data source. The project is designed to be automated, reliable, scalable, and affordable, ensuring that it can be easily maintained and expanded over time. See the attached image for further details.

![image](https://user-images.githubusercontent.com/79177516/219904889-225158d2-6afb-4aa2-a1e4-28abfd44ac23.png)



# [Project 3: BERT Natural Language Processing (NLP) Model - Python](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/BERTModel_HomeWork.html)
## Scenario: 
### Store Logistics Accuracy Modeling: 
In this project, we aim to construct a learning-based entity extraction model, specifically a custom BERT model, which will extract the store number from the transaction descriptor. The dataset is divided into three sections: train, validate, and test, and we utilize Python libraries such as NumPy, Keras, TensorFlow, and Pandas. The project will involve preprocessing the data to make it more usable, as well as training and developing an algorithm model using sequence-based datasets.
For those unfamiliar with BERT, it is a group of Transformers encoders stacked on top of each other. In more technical terms, BERT is a precise, huge transformer masked language model. Models are the output of an algorithm run on data, which includes the procedures used to make predictions on data. Once the custom part model is developed, it can be implemented in real-life examples given the data provided. Overall, this project showcases the use of cutting-edge NLP techniques and advanced machine learning algorithms to solve complex data extraction problems.

### [Link - Data Set for Model ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Example.Dataset.Homework.html)

![image](https://user-images.githubusercontent.com/79177516/180825125-298cec42-47ab-4d9d-b5eb-0178e2c27562.png)



# Data Science

# [Project 4: Automated Machine Learning Techniques (EvalML)](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/HeartAttackRiskPredictor.html)
## Scenario:
### Heart Attack Risk Predictor Modeling: 
The primary objective of this project is to develop an automated machine learning model to predict the risk of heart attacks. The dataset includes various attributes that are crucial for detecting heart diseases. We will compare the performance of our model with other conventional machine learning models such as Logistic Regression, Decision Tree, Random Forest, K Nearest Neighbor, and SVM. The automated machine learning process involves using AutoML to automate the tasks of applying machine learning to real-world problems. We will utilize the EvalML library to automate a more significant part of the machine learning process. EvalML enables us to evaluate which machine learning pipeline works best for the given dataset. By comparing the performance of different models, we aim to build a more efficient and accurate heart attack risk prediction model.

![image](https://user-images.githubusercontent.com/79177516/165724113-5a4ebe24-9eb0-4546-bf4d-3cb7295cf050.png)



# [Project 5: Learning (Long Short-term Memory) LSTM Model](https://jjimenez55993292.github.io/Deep-Learning-LSTM-model/PredictingNextWordInASentence.html)
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



# [Project 6: Logistic Regression - Pyhton - Sentiment Analysis ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Sentiment%20Analysis%20Using%20Logistic%20Regression%20Model.html)

## Scenario: 
###  Sentiment Analysis Modeling: 

In this project, we focus on developing a logistic regression model for sentiment analysis using data science techniques. Python libraries (NumPy, pickle, Keras, TensorFlow, and Pandas) and Django web framework will be utilized in the development process. Our dataset consists of tweets which will be preprocessed to make it more usable for analysis. We will be using a sequence-based dataset to train and develop an algorithm model for sentiment analysis. Sentiment analysis, also known as opinion mining, is a natural language processing technique used to determine the sentiment (positive, negative or neutral) of the given data. Developing models for sentiment analysis is crucial in understanding customer sentiment for a specific scenario. We will start by using Python to build features and a logistic regression model. Firstly, we will closely examine the textured data and extract the necessary features from the dataset. Our logistic regression model will be developed to predict the sentiment of the given data. With the developed model, we can effectively analyze and classify the sentiment of tweets into positive, negative, or neutral categories. The implementation of such a model can be useful in a wide range of fields, including social media analytics and marketing research. With the help of the Django web framework, we will be able to deploy and host the model on a website to make it accessible to a larger audience.

![image](https://user-images.githubusercontent.com/79177516/164423025-d5c8c139-aeca-49de-8c80-7f070375b553.png)

## Implementation:
### Sentiment Analysis Using Logistic Regression Model Website: 
### [Website:Sentiment Analysis Sentiment Analyzer Website - Link ](https://jimenez55993292.pythonanywhere.com/polls/)

In this project, we have built and deployed a data science model for sentiment analysis using Django, an open-source web application framework written in Python. Sentiment analysis, also known as opinion mining, is a natural language processing technique used to determine whether data is positive, negative, or neutral. Our logistic regression model has achieved an impressive accuracy rate of 99.50 percent. With the functions that were developed, we can predict the sentiment of a string input through our website. The final step of this project involved hosting the newly built website on the Django platform, allowing us to access the website and run the model from anywhere across the internet. This project showcases the importance of developing specific models for specific scenarios in order to effectively analyze customer sentiment, and highlights the power of Python libraries such as NumPy, pickle, Keras, TensorFlow, and Pandas in building accurate and efficient machine learning models.

![image](https://user-images.githubusercontent.com/79177516/164618063-9625d13e-8476-4318-a276-9ab4814bd2c0.png)



# [Project 7: Logistic Regression](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/project_4.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, we aim to create a logistic regression model in R to investigate the dataset and answer questions related to credit default risk. The dataset provided is from a credit card company, containing historical data of customer characteristics and their likelihood of defaulting on credit. It is crucial for the company to assess the risk of credit default in their customers and take proactive measures to mitigate the risk. Using logistic regression, we can analyze the relationship between various customer characteristics and their probability of defaulting on credit, enabling the company to make data-driven decisions and optimize their risk management strategies.

![image](https://user-images.githubusercontent.com/79177516/137412784-2fe2bd4f-e615-41f4-857c-7095df391b34.png)




# [Project 8: Decision Trees](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Three.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, our aim is to analyze the historical data provided by a credit card company to identify the relationships between customer characteristics and the likelihood of credit default. To achieve this, we use Decision Tree models which are a powerful tool for analyzing and classifying data. The Decision Tree algorithm works by recursively splitting the data based on the features that best discriminate between the classes. We will train our model on the historical data and use it to predict the probability of default for new customers. By accurately assessing the risk of credit default, the credit card company can take proactive measures to minimize their financial losses.

![image](https://user-images.githubusercontent.com/79177516/137412692-3a8c369e-022d-498a-8b24-a612f6f6b46f.png)




# [Project 9: Multiple Regression, Qualitative Variables Interactions, Quadratic Regression](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_One.html)
## Scenario:
### Real Estate Regression Modeling:
In this project, we will leverage a large set of historical data to analyze the relationships between various attributes of a house, such as square footage or the number of bathrooms, and the house's selling price. Our goal is to create accurate regression models that can predict house sale prices based on key variable factors. To achieve this, we will utilize Multiple Regression, Qualitative Variables Interactions, and Quadratic Regression to develop the most effective models. By developing these regression models, we aim to help a real estate company set more informed and accurate prices when listing a home for a client. Ultimately, our goal is to ensure that listings can be sold within a reasonable amount of time, and our regression models will play a crucial role in this process.

![image](https://user-images.githubusercontent.com/79177516/137412480-cea56d11-e9c0-4bf6-be74-c523107f6db3.png)



# [Project 10: Logistic Regression and Random Forests](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Two.html)
## Scenario:
### Heart Disease Modeling: 
The University Hospital is conducting a research on the risk factors associated with heart disease. They have access to a large set of historical data that contains information on different health indicators, such as fasting blood sugar and maximum heart rate, which can be used to analyze patterns and identify the presence of heart disease. The project aims to create multiple logistic regression models to predict the likelihood of an individual being at risk of heart disease. Such models can prove to be highly valuable in evaluating medical records and identifying hidden risks that may not be apparent to human doctors. In addition, the project also involves creating a classification random forest model to predict the risk of heart disease and a regression random forest model to predict the maximum heart rate achieved. The project is aimed at developing various models to analyze the Heart Disease dataset using Logistic Regression and Random Forests.

### [GitHub - Jupyter Notes ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Logistic_Regression_%20jupyter_notes_2.html)

![image](https://user-images.githubusercontent.com/79177516/137412663-55e2d96f-9453-4a3f-a1b6-2f164cd143ab.png)

