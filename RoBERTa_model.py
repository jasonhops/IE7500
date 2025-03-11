#pip install transformers datasets torch scikit-learn pandas tqdm


import pandas as pd

#from google.colab import drive


#drive.mount('/content/drive')


df = pd.read_csv("Reviews.csv")
df.head()


df.shape

df.isnull().sum()

# Drop NaN values (if any)
df.dropna(inplace=True)


df.isnull().sum()

df.info()

invalid_score = df[(df["Score"] < 1) | (df["Score"] > 5)]
invalid_score

invalid_date = df[(df["Time"] < 938736000) | (df["Time"] > 1351728000)]
print(invalid_date)

df.shape

#### Randomly selecting 50000 reviews from Amazon Fine Food Reviews dataset(df)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# Convert Score to binary sentiment labels (positive/negative)
df['Sentiment_label'] = df['Score'].apply(lambda x: 'positive' if x >= 4 else 'negative')

#Create `Combined_Text` column
df["Combined_Text"] = df["Summary"].fillna('') + " " + df["Text"].fillna('')

test_size = 0.2  # 20% test set

#Stratify train-test split based on `Sentiment_label`
#train_df, test_df = train_test_split(df, test_size=test_size, stratify=df["Sentiment_label"], random_state=42)

# Stratify on `Score` instead of `Sentiment_label`
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df["Score"], random_state=42)

#Print distribution of scores in train and test sets
print("Train Set Score Distribution:\n", train_df["Score"].value_counts(normalize=True))
print("\nTest Set Score Distribution:\n", test_df["Score"].value_counts(normalize=True))

# Sample exactly 12,500 reviews for `test_df`
test_df = test_df.groupby("Sentiment_label", group_keys=False).apply(lambda x: x.sample(n=int(12500 * len(x) / len(test_df)), random_state=42))

#Ensure `test_df` includes `Combined_Text`
test_df["Combined_Text"] = test_df["Summary"].fillna('') + " " + test_df["Text"].fillna('')

# Oversample training set for balanced classes
train_pos_oversampled = resample(train_df[train_df["Sentiment_label"] == 'positive'],
                                 replace=True, n_samples=25000, random_state=42)
train_neg_oversampled = resample(train_df[train_df["Sentiment_label"] == 'negative'],
                                 replace=True, n_samples=25000, random_state=42)

#Create final balanced training dataset
train_oversampled_df = pd.concat([train_pos_oversampled, train_neg_oversampled], ignore_index=True)

#Ensure `train_oversampled_df` includes `Combined_Text`
train_oversampled_df["Combined_Text"] = train_oversampled_df["Summary"].fillna('') + " " + train_oversampled_df["Text"].fillna('')

#Shuffle the training dataset
train_oversampled_df = train_oversampled_df.sample(frac=1, random_state=42).reset_index(drop=True)

#Check the final class distribution
print("Train Set Distribution (After Oversampling):\n", train_oversampled_df["Sentiment_label"].value_counts())
print("\nTest Set Distribution:\n", test_df["Sentiment_label"].value_counts())


test_df.head()

#pip install transformers pandas torch tqdm


import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Load RoBERTa tokenizer & model
MODEL_NAME = "siebert/sentiment-roberta-large-english"  # Pretrained RoBERTa model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# Load RoBERTa model for binary classification (2 output labels)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=2)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)


test_df["Sentiment_label"].isnull().sum()

import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score

# convert test set reviews into a list
test_texts = test_df["Combined_Text"].tolist()

# Tokenize dataset in chunks
batch_size = 32  # Adjust to fit GPU memory (try 16 if still OOM)
encoded_test = tokenizer(test_texts, truncation=True, padding=True, max_length=512, return_tensors="pt")

# Create DataLoader for batch processing
dataset = TensorDataset(encoded_test["input_ids"], encoded_test["attention_mask"])
dataloader = DataLoader(dataset, batch_size=batch_size)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Store predictions
all_preds = []

# Process batches to avoid memory overload
with torch.no_grad():
    for batch in dataloader:
        torch.cuda.empty_cache()  # Free unused memory
        batch = [t.to(device) for t in batch]  # Move batch to GPU
        inputs = {"input_ids": batch[0], "attention_mask": batch[1]}
        outputs = model(**inputs)
        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()  # Move back to CPU
        all_preds.extend(preds)

# Map numerical predictions back to sentiment labels
sentiment_map = {0: "negative", 1: "positive"}
test_df["Predicted_Sentiment"] = [sentiment_map[label] for label in all_preds]

# Convert labels to numerical values for evaluation
test_df["Sentiment_label"] = test_df["Sentiment_label"].map({"negative": 0, "positive": 1})
test_df["Predicted_Sentiment"] = test_df["Predicted_Sentiment"].map({"negative": 0, "positive": 1})

#Compute accuracy
accuracy = accuracy_score(test_df["Sentiment_label"], test_df["Predicted_Sentiment"])
print(f"Model Accuracy on Test Set: {accuracy:.2%}")

precision = precision_score(test_df["Sentiment_label"], test_df["Predicted_Sentiment"])
print(f"Model Precision on Test Set: {precision:.2%}")

recall = recall_score(test_df["Sentiment_label"], test_df["Predicted_Sentiment"])
print(f"Model Recall on Test Set: {recall:.2%}")

f1 = 2 * (precision * recall) / (precision + recall)
print(f"Model F1 Score on Test Set: {f1:.2%}")

#Print classification report
print(classification_report(test_df["Sentiment_label"], test_df["Predicted_Sentiment"], target_names=["Negative", "Positive"]))


def predict_sentiment(text):
    # Tokenize input text
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs = {key: value.to(device) for key, value in inputs.items()}  # Move to GPU if available

    # Get model predictions
    with torch.no_grad():
        outputs = model(**inputs)

    # Apply softmax to get probabilities
    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_class = torch.argmax(probs).item()

    # Map labels (0 = Negative, 1 = Positive)
    sentiment_map = {0: "negative", 1: "positive"}

    return sentiment_map[predicted_class], probs.tolist()

# Example reviews
sample_reviews = [
    "The product was amazing! I loved it.",
    "It was okay, nothing special but not bad either.",
    "This is the worst thing I have ever bought!",
    "As an avid chess enthusiast, I've always sought a set that marries classic aesthetics with modern convenience. The Hurdaos Magnetic Wooden Chess Set has exceeded my expectations in every way. The 15-inch handcrafted board is not only visually stunning but also folds into a compact, portable case, making it ideal for both home and travel. The magnetic pieces stay firmly in place during play, yet are easy to move, ensuring an uninterrupted game even on the go. Each piece is meticulously crafted, adding a touch of elegance to every match. Whether you're a seasoned player or a beginner, this set offers a seamless blend of functionality and style. Highly recommended",
    "They are all dry. Only 9 markers worked out of 36 markers it has in the box. You are better off just getting the brand Expo."
]

# Predict for each review
for review in sample_reviews:
    sentiment, probabilities = predict_sentiment(review)
    print(f"Review: {review}\nPredicted Sentiment: {sentiment}\nProbabilities: {probabilities}\n")


import matplotlib.pyplot as plt

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(test_df["Sentiment_label"], test_df["Predicted_Sentiment"])
plt.figure(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Negative", "Positive"], yticklabels=["Negative", "Positive"])

plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()





