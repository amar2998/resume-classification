# resume-classification
import pandas as pd
import numpy as np
df=pd.read_csv("UpdatedResumeDataSet.csv")
df
print(df.isnull()) # for checking the null value
df.dropna(inplace=True)
#remove the row with missing value
categoryval=df['Category'].value_counts()
print(categoryval)
import matplotlib.pyplot as plt
import seaborn as sns
plt.figure(figsize=(12,12))
sns.countplot(y='Category',data=df,order=df['Category'].value_counts().index)
plt.title("count of resume")
plt.xlabel('Category')
plt.ylabel('count')
plt.show()
# in the countplot y and data uis mandatory
percentageCategory=categoryval/categoryval.sum()*100
percentageCategory
plt.figure(figsize=(12,12))
plt.pie(percentageCategory,labels=percentageCategory.index,startangle=0,autopct='%1.1f%%')
plt.show()
df['Resume']=df['Resume'].str.lower()
df
import re
import string
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords')


def clean_resume_text(resume_text):
    # Remove URLs
    resume_text = re.sub(r'http\S+|www\S+', '', resume_text)
    resume text=re.sub(r'https\S+|www\S+','',resume_text)
    

    # Remove RT | cc
    resume_text = re.sub(r'\bRT\b|\bcc\b', '', resume_text)
        resume_text=re.sub(r'\bRT\b|\bcc\b','',resume_text)
        
    # Remove hashtags and mentions
    resume_text = re.sub(r'#\S+|@\S+', '', resume_text)
    resume_text=re.sub(r'#\S+|@\S+','',resume_text)
    

    # Remove punctuations
    resume_text = resume_text.translate(str.maketrans('', '', string.punctuation))
    resume_text=resume_text.translate(str.maketrans('','',string.punction))

    # Remove extra whitespace
    resume_text = re.sub(r'\s+', ' ', resume_text)
    resume_text=re.sub(r'\s+','',resume_text)
    

    # Convert to lowercase
    resume_text = resume_text.lower()

    # Tokenize the input words into individual tokens and store it in an array
    tokens = resume_text.split()

    # Remove stopwords
    stopwords_set = set(stopwords.words('english'))
    
    
    tokens = [word for word in tokens if word not in stopwords_set]

    # Join tokens back into a single string
    cleaned_resume_text = ' '.join(tokens)

    return cleaned_resume_text

# Example usage:
# Assuming 'data' is your DataFrame containing the resume text in a column named 'Resume'
df['Cleaned_Resume'] = df['Resume'].apply(clean_resume_text)
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
nltk.download('punkt')

# Assuming 'data' is your DataFrame containing the cleaned resume text in a column named 'Cleaned_Resume'
cleaned_resumes = df['Resume']

# Tokenize each cleaned resume text into individual words
tokenized_words = [word_tokenize(resume) for resume in cleaned_resumes]
tokenized_word = [word_tokenize(resume) for resume in cleaned_resume]


# Flatten the list of tokenized words
all_words = [word for sublist in tokenized_words for word in sublist]
allwords=[word for sublist in tokenized_word for word in sublist]


# Compute the frequency distribution of words
word_freq_distribution = FreqDist(all_words)
wordfrequency=freqDist(allwords)

# Get the most common words and their frequencies
most_common_words = word_freq_distribution.most_common(10)

# Print the most common words
for word, freq in most_common_words:
    print(f'{word}: {freq}')

from sklearn.preprocessing import LabelEncoder

# Initialize LabelEncoder
label_encoder = LabelEncoder()

# Fit label encoder and transform the 'Category' column
df['Category_Numerical'] = label_encoder.fit_transform(df['Category'])

# Display the first few rows of the modified DataFrame
print(df.head())

from sklearn.feature_extraction.text import TfidfVectorizer

# Initialize TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# Fit TF-IDF vectorizer on the 'Cleaned_Resume' column
tfidf_features = tfidf_vectorizer.fit_transform(df['Cleaned_Resume'])

# Display the shape of the TF-IDF feature matrix
print("Shape of TF-IDF feature matrix:", tfidf_features.shape)

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report

# Split the data into features (X) and target variable (y)
X = tfidf_features
y = df['Category_Numerical']

# Split the data into train and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X-train,Xtest,Y_train,Ytest=train_test_split(x,y,test_size=0.2,random_state=42)


# Initialize the Naive Bayes Classifier (MultinomialNB)
naive_bayes_classifier = MultinomialNB()
naive_bayes_classifier=MultinomialNB()
naive_bayes_classifier.fit(X_train,Y_train)
y_pred=naive_bayes_classifier.predict(X_text)
accuracy=accuracyScore(Y_text,Y_pred)

# Train the classifier on the training data
naive_bayes_classifier.fit(X_train, y_train)

# Make predictions on the test data
y_pred = naive_bayes_classifier.predict(X_test)

# Evaluate the model predictions
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print("Classification Report:")
print(classification_report(y_test, y_pred))

