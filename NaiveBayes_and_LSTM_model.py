import pandas as pd

df= pd.read_csv("Reviews.csv")
df.head()


df.shape

df.isna().sum()

df.dropna(subset=["Summary"], inplace=True)
df.dropna(subset=["ProfileName"], inplace=True)
df.info()
df.isna().sum()

duplicates = df.duplicated()
print(duplicates.sum())
duplicates

print(df.isnull().sum())
df[df.isnull().any(axis=1)]

invalid_score = df[(df["Score"] < 1) | (df["Score"] > 5)]
print(invalid_score)

invalid_date = df[(df["Time"] < 938736000) | (df["Time"] > 1351728000)]
print(invalid_date)

df["Text"] = df["Text"].str.lower().str.replace(r'[^a-z\s]', '', regex=True)
df["Summary"] = df["Summary"].str.lower().str.replace(r'[^a-z\s]', '', regex=True)

from sklearn.model_selection import train_test_split

df['Sentiment_label'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else 'negative')

test_size = 0.2  # 20% test split
train_size = 0.8  # 80% train split

# Split the test set first
train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["Score"], random_state=42)
test_df=test_df.sample(n=12500, random_state=42)
train_oversampled_df = pd.concat([train_df[train_df["Sentiment_label"]=='positive'].sample(n=25000, random_state=42), train_df[train_df["Sentiment_label"]=='negative'].sample(n=25000, random_state= 42)], ignore_index=True)


import matplotlib.pyplot as plt
import seaborn as sns
# Count positive and negative reviews
original_distribution = train_df["Sentiment_label"].value_counts()

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=original_distribution.index, y=original_distribution.values, palette=["red", "green"])
plt.title("Original Sentiment Distribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Review Count")
plt.show

# Count positive and negative reviews
original_distribution = train_oversampled_df["Sentiment_label"].value_counts()

# Plot the distribution
plt.figure(figsize=(6, 4))
sns.barplot(x=original_distribution.index, y=original_distribution.values, palette=["red", "green"])
plt.title("Oversampled Sentiment Distribution")
plt.xlabel("Sentiment Label")
plt.ylabel("Review Count")
plt.show()

# Compute class distributions
train_score_distribution = train_oversampled_df["Score"].value_counts(normalize=True)
test_score_distribution = test_df["Score"].value_counts(normalize=True)

# Set up subplots
fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)

# Train Set Score Distribution
sns.barplot(x=train_score_distribution.index, y=train_score_distribution.values, ax=axes[0])
axes[0].set_title("Train Set Score Distribution")
axes[0].set_ylabel("Proportion")
axes[0].set_xlabel("Score")

# Test Set Score Distribution
sns.barplot(x=test_score_distribution.index, y=test_score_distribution.values, ax=axes[1])
axes[1].set_title("Test Set Score Distribution")
axes[1].set_xlabel("Score")

plt.tight_layout()
plt.show()


import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import subjectivity
from nltk.sentiment import SentimentAnalyzer
from nltk.sentiment.util import *
from nltk.tokenize import word_tokenize


train_oversampled_df["Score"].mean()
#sampled_df["Score"].median()

print(len(train_oversampled_df))
print(len(test_df))
print(len(train_oversampled_df[train_oversampled_df['Sentiment_label']=='positive']))
print(len(train_oversampled_df[train_oversampled_df['Sentiment_label']=='negative']))


# Create document lists as (review_text, label) pairs
train_docs = list(zip(train_oversampled_df['Summary'] + " " + train_oversampled_df['Text'], train_oversampled_df['Sentiment_label']))
test_docs = list(zip(test_df['Summary'] + " " + test_df['Text'], test_df['Sentiment_label']))

# Initialize the SentimentAnalyzer
sentim_analyzer = SentimentAnalyzer()

# Build a list of all words from the training data
all_words = []
for review, _ in train_docs:
    tokens = word_tokenize(review)
    all_words.extend(tokens)

# Create unigram features with a minimum frequency threshold
unigram_feats = sentim_analyzer.unigram_word_feats(all_words, min_freq=5) # testing

# Add the unigram feature extractor to the analyzer using NLTK's utility function
sentim_analyzer.add_feat_extractor(extract_unigram_feats, unigrams=unigram_feats)


# Convert the raw documents into feature sets
training_set = sentim_analyzer.apply_features(train_docs)
test_set = sentim_analyzer.apply_features(test_docs)

# Train the Naive Bayes Classifier using the training feature set
classifier = NaiveBayesClassifier.train(training_set)

# Evaluate the classifier on the test set
accuracy = nltk.classify.accuracy(classifier, test_set)
print("Accuracy:", accuracy)

# Display the 10 most informative features
classifier.show_most_informative_features(10)

from sklearn.metrics import classification_report

# Get predicted labels
true_labels = [label for (_, label) in test_docs]  # Actual labels
predicted_labels = [classifier.classify(feats) for (feats, label) in test_set]  # Predicted labels

# Compute precision, recall, f1-score, and support
print(classification_report(true_labels, predicted_labels))


from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

## Download NLTK stopwords
nltk.download('stopwords')
nltk.download('punkt')
stop_words = set(stopwords.words('english'))

# Now split out the text and sentiment labels for the test set
X_train_text = train_oversampled_df['Summary'] + " " + train_oversampled_df['Text']
X_test_text = test_df['Summary'] + " " + test_df['Text']
y_train = train_oversampled_df['Sentiment_label']
y_test = test_df['Sentiment_label']


# Then, transform the text using TF-IDF
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(ngram_range=(1,2), max_features=5000)
X_train = vectorizer.fit_transform(X_train_text)
X_test = vectorizer.transform(X_test_text)

# Train Naïve Bayes Classifier
nb_model = MultinomialNB()
nb_model.fit(X_train, y_train)


import numpy as np
import pandas as pd

# Make predictions
y_pred = nb_model.predict(X_test)


# Find misclassified indices
misclassified_indices = np.where(y_pred != y_test)[0]

# Create a DataFrame of misclassified reviews
misclassified_reviews = pd.DataFrame({
    "Review": X_test_text.iloc[misclassified_indices],
    "Actual": y_test.iloc[misclassified_indices],
    "Predicted": y_pred[misclassified_indices]
})

print(misclassified_reviews.head(10))



# accuracy measures the overall correctness of the model(correct_pred / total_pred)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy:.4f}")

#Classification report

#Precision:How many of the predicted positives were actually positive? (Lower False Positives)
#Recall :How many of the actual positives were correctly identified? (Lower False Negatives)
#F1score: The harmonic mean of precision & recall (a balance between the two)
#Support:The number of actual samples in that class

print('Classification report:\n',classification_report(y_test, y_pred))

cm= confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense
import tensorflow as tf

# Create document lists as (review_text, label) pairs
# Separate texts and labels
train_texts, train_labels = train_oversampled_df['Summary'] + " " + train_oversampled_df['Text'], train_oversampled_df['Sentiment_label']
test_texts, test_labels = test_df['Summary'] + " " + test_df['Text'], test_df['Sentiment_label']

# Convert sentiment labels to binary (positive -> 1, negative -> 0)
label_map = {'positive': 1, 'negative': 0}
train_labels = np.array([label_map[label] for label in train_labels])
test_labels = np.array([label_map[label] for label in test_labels])

# Tokenization and sequence padding
max_features = 25000  # Vocabulary size
maxlen = 500          # Maximum sequence length

tokenizer = Tokenizer(num_words=max_features, oov_token='<OOV>')
tokenizer.fit_on_texts(train_texts)

x_train = tokenizer.texts_to_sequences(train_texts)
x_test = tokenizer.texts_to_sequences(test_texts)

x_train = pad_sequences(x_train, maxlen=maxlen, padding='post', truncating='post')
x_test = pad_sequences(x_test, maxlen=maxlen, padding='post', truncating='post')

print("x_train shape:", x_train.shape)
print("x_test shape:", x_test.shape)



# Build the LSTM model
embedding_size = 128
lstm_units = 64
batch_size = 32
epochs = 5

model = Sequential()
model.add(Embedding(input_dim=max_features, output_dim=embedding_size, input_length=maxlen))
model.add(LSTM(lstm_units, dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

# Train the model (using 20% of training data for validation)
history = model.fit(x_train, train_labels,
                    validation_split=0.2,
                    batch_size=batch_size,
                    epochs=epochs)

# Evaluate the model on the test set
score, accuracy = model.evaluate(x_test, test_labels, batch_size=batch_size)
print("Test loss:", score)
print("Test accuracy:", accuracy)

print("Test loss:", score)
print("Test accuracy:", accuracy)



from sklearn.metrics import classification_report, confusion_matrix

# Generate predicted probabilities
y_pred_prob = model.predict(x_test)

# Convert probabilities to binary predictions (0 or 1) using threshold 0.5
y_pred = (y_pred_prob > 0.5).astype(int)

print('Classification report:\n',classification_report(test_labels, y_pred))

print(test_labels)
print(y_pred)

cm= confusion_matrix(test_labels, y_pred)
sns.heatmap(cm, annot=True, fmt='d',cmap='Blues')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()





