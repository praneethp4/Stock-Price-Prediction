# Stock-Price-Prediction
We will predict a signal that indicates whether buying a particular stock will be helpful or not 
by using ML. Stock price prediction is a challenging problem in finance, and machine learning 
has shown promise in this field. The goal of this project is to develop a machine learning 
model for predicting stock prices using historical data and financial features. 

## SOLUTION
The process involves collecting and preprocessing financial data, selecting, and training a 
machine learning model, and evaluating its performance. The project will explore different 
techniques for feature engineering and model selection and aim to achieve high accuracy 
and robustness in stock price prediction. The results of this project will contribute to the 
understanding of the use of machine learning in finance and have practical applications in 
investment portfolio management.

## DESIGN
##### 1. Import Libraries
Importing libraries such as Numpy,Pandas, Scikit-Learn which helps us to handle data 
manipulation, preprocessing, modelling tasks. 
##### 2. Collecting Data
In machine learning, algorithms are trained on data, and the quality of the data directly 
affects the quality of the model's predictions. The model learns from this data and uses 
it to make predictions on new, unseen data. Collecting a diverse range of data is critical 
to avoid bias in the model's predictions
##### 3. Data Cleaning and Preprocessing
Perform data cleaning tasks such as removing duplicates, filling in missing data, and 
handling outliers. Also, pre-process the data to make it suitable for machine learning 
algorithms by scaling, normalizing, and transforming features as needed.
##### 4. Feature engineering
Feature Engineering helps to derive some valuable features from the existing ones. 
These extra features sometimes help in increasing the performance of the model 
significantly and certainly help to gain deeper insights into the data.
##### 5. Selecting Machine Learning Model
Select an appropriate machine learning algorithm for the problem at hand. Some 
popular algorithms for stock price prediction include Linear Regression, Decision Trees, 
Random Forests, Gradient Boosting, and Deep Learning algorithms such as LSTM and 
CNN.
##### 6. Model Training
Train the selected model using the preprocessed data. This involves splitting the data 
into training and testing sets, training the model on the training set, and evaluating its 
performance on the testing set.
##### 7. Monitoring and Maintenance:
Continuously monitor the performance of the deployed model and maintain it by 
updating it with new data and retraining it as needed.

## REQUIREMENTS
Flask,
PyMySQL,
Keras,
Tensorflow,
Libraries:numpy,pandas,scikit-learn

## OVERVIEW
### USE CASE DIAGRAM
![image](https://github.com/praneethp4/Stock-Price-Prediction/assets/123055147/6209c135-3ef3-4a65-83a2-7e5cecd690a8)

### CLASS DIAGRAM
![image](https://github.com/praneethp4/Stock-Price-Prediction/assets/123055147/0e49ec89-d796-4e59-8c18-cccbbd33854c)

### ENTITY RELATIONSHIP DIAGRAM
![image](https://github.com/praneethp4/Stock-Price-Prediction/assets/123055147/087394d1-a858-45c7-87d3-4e8220aa1c51)

### DATA FLOW DIAGRAM
![image](https://github.com/praneethp4/Stock-Price-Prediction/assets/123055147/45ab79c3-026a-4e7d-88c8-e309d5dcf41e)

## COMPARING MODELS
![image](https://github.com/praneethp4/Stock-Price-Prediction/assets/123055147/13238e9d-0cad-4df3-a86b-551a7f9dcebb)

## Generating LSTM Model
For making the prediction live we used LSTM model # Generating LSTM Model. Basically LSTM (Long 
Short-Term Memory) is a type of recurrent neural network (RNN) that is designed to address the 
problem of vanishing and exploding gradients that can occur in traditional RNNs. LSTM is particularly 
useful for tasks that involve sequential data, such as speech recognition, natural language 
processing, and time series prediction, where it is important to maintain information over longer 
time periods.
### Backend for LSTM model
First, we will import dataset to work on and convert date into required format splitting data into 
train and test data generate lstm model and save it into saved_lstm_model.h5 this is all about 
backend part
### Frontend for LSTM model
Coming to the frontend part By using FLASK a link will be generated where users will register if they 
are new users and login. They will enter the closing date and the saved model delivers the predicted 
price

