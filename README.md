# Jacinto_Jimenez_Portfolio
### Data Science and Data Engineering Portfolio

Hello, my name is Jacinto Jimenez, and welcome to my data science and data engineerng portfolio. I am very passionate about data. Data science helps us answer many questions using data. Data is all around us, and data science allows us to answer questions we wonder about with using datasets. I have included a couple of projects that I have worked on in courses during college. These projects use different types of libraries in Python and the R programming languages. I also use various data analysis techniques to develop models.  The projects I have chosen cover different modeling. 

# Data Engineering

# [Project 1: Earthquake Data Engineering Project](https://github.com/Jjimenez55993292/EarthquakeDataEngineering)
## Scenario:
### Earthquake Data Pipeline: 
This project is to ingest and process earthquake data from the USGS and create a data pipeline, store it in a datalake, and make it available for analysis and visualization. The project has been designed to be automated, reliable, scalable, and affordable, using a variety of open-source technologies such as Terraform, Airflow, Docker, GCS, BigQuery, dbt, and Looker Studio.
The data is obtained by accessing a public REST API provided by USGS, and the seismic events data is stored in a denormalized format using the One Big Table (OBT) method. The project is implemented as an Airflow DAG that runs daily and parameterizes the API endpoint to use the dates passed in by using the Airflow provided template variables. The seismic events data is transformed and materialized as a view and an incremental table using dbt, with a built-in tool to visually show data lineage as it travels across the data layers: raw -> stage -> final.
Finally, the data is made available for analysis and visualization in Looker Studio using BigQuery as the data source. See image below.


![image](https://github.com/Jjimenez55993292/EarthquakeDataEngineering/blob/main/images/architecture_earthquake.excalidraw.png)




# [Project 2: BERT Natural Language Processing (NLP) Model - Python](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/BERTModel_HomeWork.html)
## Scenario: 
### Store Logistics Accuracy Modeling: 
In this project, we build a learning-based entity extraction model (a custom Bert Model) to extract the store_number from the transaction_descriptor. The dataset contains three sections: train, validate, and test. We use Python libraries (NumPy, Keras, TensorFlow, Pandas). This project will preprocess the data into a more usable format. Will train and develop an algorithm model using the sequence-based datasets. What is BERT NLP Model? BERT NLP model is a group of Transformers encoders stacked on each other. BERT is a precise, huge transformer masked language model in more technical terms. Models are the output of an algorithm run on data, including the procedures used to make predictions on data. Once we develop the custom part model we can use it for implementation in a real-life example given the data provided.

### [Link - Data Set for Model ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Example.Dataset.Homework.html)

![image](https://user-images.githubusercontent.com/79177516/180825125-298cec42-47ab-4d9d-b5eb-0178e2c27562.png)



# Data Science

# [Project 3: Automated Machine Learning Techniques (EvalML)](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/HeartAttackRiskPredictor.html)
## Scenario:
### Heart Attack Risk Predictor Modeling: 
The main purpose of this project is to predict whether a person is at risk of a heart attack or not, using automated Automated Machine Learning Techniques. We are given a data set in which various attributes are provided, crucial for heart disease detection. We are going to compare our data to other ML Models such as, Logistic Regression,Decision Tree,Random Forest,K Nearest Neighbor and SVM models. Then we will compare the models using automated Automated Machine Learning Techniques. AutoML is the process of automating the tasks of applying machine learning to real-world problems. We will be using the EvalML Library. EvalML automates a more significant part of the machine learning process. We can quickly evaluate which machine learning pipeline works better for the given data set. 

![image](https://user-images.githubusercontent.com/79177516/165724113-5a4ebe24-9eb0-4546-bf4d-3cb7295cf050.png)



# [Project 4: Learning (Long Short-term Memory) LSTM Model](https://jjimenez55993292.github.io/Deep-Learning-LSTM-model/PredictingNextWordInASentence.html)
## Scenario:
### LSTM Model - Predicting The Next Word In A Sentence: 
In this project, we build and deploy Data Science, Machine Learning, Deep Learning models. We use Python libraries (NumPy, pickle, Keras, TensorFlow, Pandas), Django, GCP, and Heruko Cloud. We will be working on a text dataset, a book. This project will preprocess the data into a more usable format. Will train and develop an algorithm model using the sequence-based datasets. We will be using Deep Learning (Long Short-term Memory) LSTM model to develop our algorithm model. This model is based on Neural Net-Architecture and provides very high performance on sequence-based datasets. It has a feedback structure helping the model remember the sequence of data input and the changes in the output depending on what is happening to predict the next word in a Sentence.

![image](https://user-images.githubusercontent.com/79177516/163972466-97233b06-1bf2-4d0a-a2fd-d57d815ce0df.png)


## Implementation:
### Predicting The Next Word In A Setence Website: 
### [Website: The Next Word In A Setence Website - Link ](http://jacintojimenez606.pythonanywhere.com/home/)

We use Django which is an open-source web application framework written in Python. This gave us the ability to use our developed python model for predicting the next word. Although our model predicts words based on the model we develop, we also must consider that our current model is only about 27 percent accurate. As our training dataset increases, the model's accuracy also increases. We were able to implement the model on our website, and we were able to get something that's workable; however, it does need Improvement. This is the last stage or step in the process of building the project where we host the newly made website using the Django platform. With this completed, we would be able to access our website and run the model from anywhere across the internet.



### [GitHub-Link ](https://github.com/Jjimenez55993292/Deep-Learning-LSTM-model)


![image](https://user-images.githubusercontent.com/79177516/163972673-bf0c7181-7694-4622-9217-bf885a0db423.png)



# [Project 5: Logistic Regression - Pyhton - Sentiment Analysis ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Sentiment%20Analysis%20Using%20Logistic%20Regression%20Model.html)

## Scenario: 
###  Sentiment Analysis Modeling: 

In the project, we build and deploy Data Science to develop a logistical regression model for Sentiment analysis in this project. We use Python libraries (NumPy, pickle, Keras, TensorFlow, Pandas), Django. We will be working on tweet datasets. This project will preprocess the data into a more usable format. Will train and develop an algorithm model using the sequence-based datasets. Sentiment analysis or opinion mining is a natural language processing technique used to determine whether data is positive-negative or neutral. Developing models is extremely important in item analysis because he could develop specific models for a specific scenario to analyze customer sentiment. We will use python to build features and a logistical regression model. First, we would understand the textured data and the procedure features from the data set in detail. We will develop our logistical regression model to predict the sentiment of the data.

![image](https://user-images.githubusercontent.com/79177516/164423025-d5c8c139-aeca-49de-8c80-7f070375b553.png)

## Implementation:
### Sentiment Analysis Using Logistic Regression Model Website: 
### [Website:Sentiment Analysis Sentiment Analyzer Website - Link ](https://jimenez55993292.pythonanywhere.com/polls/)

We use Django, an open-source web application framework written in Python. This gave us the ability to use our developed python model for Sentiment Analysis. Sentiment analysis or opinion mining is a natural language processing technique used to determine whether data is positive-negative or neutral. Our logistic regression model accuracy is at 99.50 percent. Then we can use the functions that were developed to predict the sentiment of the string input in our website. This is the last stage or step in the process of building the project where we host the newly made website using the Django platform. With this completed, we would be able to access our website and run the model from anywhere across the internet.

![image](https://user-images.githubusercontent.com/79177516/164618063-9625d13e-8476-4318-a276-9ab4814bd2c0.png)



# [Project 6: Logistic Regression](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/project_4.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, we create a model in R and we create a logistic regression model to help answer the questions question we have about the dataset. A credit card company has access to a set of historical data that can be used to study the relationships between customer characteristics and whether or not they are likely to default on their credit. It is important for the company to calculate the risk that their customers will default on their credit.

![image](https://user-images.githubusercontent.com/79177516/137412784-2fe2bd4f-e615-41f4-857c-7095df391b34.png)




# [Project 7: Decision Trees](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Three.html)
## Scenario: 
### Credit Card Risks Modeling: 
In this project, we use Decision Trees models to create a model to help answer the questions question we have about the dataset.  A credit card company has access to a set of historical data that can be used to study the relationships between customer characteristics and whether or not they are likely to default on their credit. It is important for the company to calculate the risk that their customers will default on their credit. 

![image](https://user-images.githubusercontent.com/79177516/137412692-3a8c369e-022d-498a-8b24-a612f6f6b46f.png)




# [Project 8: Multiple Regression, Qualitative Variables Interactions, Quadratic Regression](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_One.html)
## Scenario:
### Real Estate Regression Modeling:
Have access to a large set of historical data to help analyze relationships between a house's different attributes (such as square footage or the number of bathrooms) and the house's selling price. The regression models will predict house sale prices based on critical variable factors.  The developed regression models will use Multiple Regression, Qualitative Variables Interactions, and Quadratic Regression to develop models. The regression models will help a real estate company set better prices when listing a home for a client. Also, Setting better prices will ensure that listings can be sold within a reasonable amount of time.
![image](https://user-images.githubusercontent.com/79177516/137412480-cea56d11-e9c0-4bf6-be74-c523107f6db3.png)



# [Project 9: Logistic Regression and Random Forests](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Project_Two.html)
## Scenario:
### Heart Disease Modeling: 
University Hospital is researching risk factors for heart disease. Have access to a large set of historical data that you can use to analyze patterns between different health indicators (e.g. fasting blood sugar, maximum heart rate, etc.) and the presence of heart disease. Create different logistic regression models that predict whether or not a person is at risk for heart disease. A model like this could eventually be used to evaluate medical records and look for risks that might not be obvious to human doctors.  Also, another model will create a classification random forest model to predict the risk of heart disease and a regression random forest model to predict the maximum heart rate achieved. For this project we create different models analyzing a Heart Disease data set using Logistic Regression and Random Forests. 
### [GitHub - Jupyter Notes ](https://jjimenez55993292.github.io/Jacinto_J_Portfolio/Logistic_Regression_%20jupyter_notes_2.html)

![image](https://user-images.githubusercontent.com/79177516/137412663-55e2d96f-9453-4a3f-a1b6-2f164cd143ab.png)

