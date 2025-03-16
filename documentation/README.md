# IE7500
# Sentiment Analysis on Amazon Fine Food Reviews
### Dataset  
- **Source:** [Amazon Fine Food Reviews](https://www.kaggle.com/datasets/snap/amazon-fine-food-reviews/data)  
- **Size:** 568,454 reviews  
- **Attributes Used:**
  - `Summary` (short text summary of the review)
  - `Text` (full review text)
  - `Score` (ratings from 1 to 5, used for sentiment classification)
### **Sentiment Classification Criteria**
1. **Initial Approach:**  
   - **Positive:** Score = **5**  
   - **Negative:** Score < **5**  

2. **Adjusted Approach (for better balance):**  
   - **Positive:** Score = **4, 5**  
   - **Negative:** Score = **1, 2,3**

### Implementation Steps  

### **1. Data Preprocessing**
- Combined `Summary` and `Text` for better sentiment representation.  
- Removed **stopwords, punctuation, and special characters**.  
- Used **TF-IDF and Byte Pair Encoding (BPE)** for feature extraction.

### **2. Model Development**
Implemented three different models for sentiment classification:

#### **Traditional Machine Learning Model**
1. **Naïve Bayes using NLTK**  
   - Extracted unigram features.  
   - Took **30+ minutes to train** and achieved **50% accuracy**.
2. **MultinomialNB from scikit-learn**  
   - Used **TF-IDF vectorization**.  
   - Faster training compared to NLTK's implementation.
   - Received accuracy of 86% but lower precision for negative reviews suggests the model struggles with identifying negative sentiment accurately.

  #### **Deep Learning**
3. **LSTM (Long Short-Term Memory)**
   - Used **word embeddings**.
   - The LSTM model with a F1 score of 0.88 is quite good for positive sentiment but is unable to pick out negative reviews well with F1 score of 0.01. The relatively small training set for LSTM and a lack of large pre-training corpus compared to Transformer models severely impacted suitability for sentiment analysis.

#### **Transformer-Based Model**
4. **RoBERTa [Pretrained](https://huggingface.co/siebert/sentiment-roberta-large-english)**
   - Used **`siebert/sentiment-roberta-large-english`**.
   - Outperformed previous models, providing **better contextual understanding**.
   - Required **GPU acceleration** due to high computational cost.

Research Paper Reference: [A survey on sentiment analysis methods, applications, and challenges](https://link.springer.com/article/10.1007/s10462-022-10144-1)

### **Setup Instructions**
#### **1. Install Jupyter Notebook**
```bash
pip install jupyter
 ``` 

#### **2. Install Dependencies**
```bash
pip install torch transformers nltk tensorflow scikit-learn pandas numpy matplotlib seaborn
 ```
#### **3. Run Jupyter Notebook**
```bash
jupyter notebook
 ```
#### **5. Download the Reviews.csv.xz file , unzip it and upload it in the code directory.**
#### **4. Navigate to the notebook files (.ipynb) and execute them step by step.**
- NaiveBayes_and_LSTM_model
- RoBERTa_model



