📧 Spam Email Classification using Machine Learning
This project demonstrates a simple machine learning pipeline to classify emails as Spam or Not Spam using text data. It uses a Naive Bayes classifier and CountVectorizer to train a model on a sample dataset.

📂 Project Structure

📁 Spam Email Classification
├── spam_email_classifier.ipynb       # Jupyter Notebook with step-by-step code
├── spam_dataset.csv                  # Sample dataset of emails
├── spam_classifier_model.pkl         # Trained Naive Bayes model
└── spam_vectorizer.pkl               # Trained CountVectorizer

📌 Requirements
-Python 3.x
-pandas
-scikit-learn
-pickle
-Jupyter Notebook

You can install the required libraries using:

pip install pandas scikit-learn notebook


🧠 Model Overview
-Algorithm: Multinomial Naive Bayes
-Vectorizer: CountVectorizer
-Training/Test Split: 80/20
-Accuracy: 100% (on sample dataset)

🚀 How to Run
Open the notebook:
jupyter notebook spam_email_classifier.ipynb

Follow the steps in each cell to:
-Load the dataset
-Vectorize the email text
-Train the Naive Bayes classifier
-Evaluate the model
-Save the model and vectorizer

📈 Sample Dataset
A small sample dataset is used with the following structure:
------------------------------------
|      text	              | label  |
------------------------------------
|"Win a free iPhone now"	|  1     |
|"Meeting at 3 PM"	      |  0     |
|"Buy now and save big"	  |  1     |
------------------------------------

->1: Spam
->0: Not Spam


💾 Model Deployment (Optional)
To use the model in a production environment:
1.Load spam_classifier_model.pkl and spam_vectorizer.pkl.
2.Transform new email text using the vectorizer.
3,Predict using the trained model.

---------------------------------------------------------------
import pickle

# Load model and vectorizer
with open("spam_classifier_model.pkl", "rb") as f:
    model = pickle.load(f)

with open("spam_vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)

# Predict new sample
sample = ["Congratulations, you've won!"]
X_sample = vectorizer.transform(sample)
prediction = model.predict(X_sample)

print("Spam" if prediction[0] == 1 else "Not Spam")
---------------------------------------------------------------
