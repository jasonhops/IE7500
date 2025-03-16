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
   - **Negative:** Score = **1, 2, 3**
  ## ⚙️ Implementation Steps  

### **1. Data Preprocessing**
- Combined `Summary` and `Text` for better sentiment representation.  
- Removed **stopwords, punctuation, and special characters**.  
- Used **TF-IDF and Byte Pair Encoding (BPE)** for feature extraction.

### **2. Model Development**
Implemented three different models for sentiment classification:

#### **🟠 Traditional Machine Learning**
1. **Naïve Bayes using NLTK**  
   - Extracted unigram features.  
   - Took **2+ hours to train** and achieved **61% accuracy** initially.  
   - Adjusting the sentiment threshold improved accuracy to **71.45%**, but the model was still **biased towards positive reviews**.  

2. **MultinomialNB from scikit-learn**  
   - Used **TF-IDF vectorization**.  
   - Faster training compared to NLTK's implementation.  
     
