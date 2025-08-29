# 📧 Spam Email Classifier  

A machine learning project to classify emails as **Spam** or **Ham (Not Spam)** using **NLP (Natural Language Processing)** techniques and machine learning algorithms.  

This project was built and tested in **Google Colab** using the [SMS Spam Collection Dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset).  

---

## 🚀 Project Workflow  

1. **Dataset Loading**  
   - Used Kaggle’s SMS Spam dataset with 5,572 labeled messages.  
   - Labels: `ham` (not spam) and `spam`.  

2. **Data Preprocessing**  
   - Lowercasing text  
   - Removing punctuation, stopwords, and special characters  
   - Tokenization  
   - Lemmatization using `nltk`  

3. **Feature Extraction**  
   - Used **TF-IDF (Term Frequency–Inverse Document Frequency)** to convert text into numerical feature vectors.  

4. **Model Training**  
   - **Logistic Regression**  
   - **Naive Bayes (MultinomialNB)**  

5. **Evaluation**  
   - Achieved ~96–97% accuracy.  
   - Evaluated with **classification reports** (precision, recall, F1-score) and **confusion matrices**.  

---

## 📊 Results  

- **Logistic Regression**: ~96% accuracy  
- **Naive Bayes**: ~97% accuracy  

Confusion matrices show that both models correctly classify most `ham` messages but sometimes misclassify `spam` messages.  

---

## 🧠 Possible Improvements  

- Use **deep learning** models (LSTM, BiLSTM, or Transformers like BERT) for larger datasets.  

---

## 🛠️ Tech Stack  

- **Python**  
- **Pandas, NumPy** → Data handling  
- **NLTK** → Preprocessing (stopwords, tokenization, lemmatization)  
- **scikit-learn** → TF-IDF, Logistic Regression, Naive Bayes, evaluation  
- **Matplotlib / Seaborn** → Visualization  

---

# 📧 Spam Email/SMS Classifier

A machine learning project to classify messages as **Spam** or **Ham (Not Spam)** using **Natural Language Processing (NLP)** and supervised learning algorithms.

---

## 📂 Dataset
We used the **SMS Spam Collection Dataset**:  
- UCI Repository: [https://archive.ics.uci.edu/ml/datasets/sms+spam+collection](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)  
- Kaggle Mirror: [https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset)  

---

## ⚙️ Installation

Clone this repository and install dependencies:

```bash
git clone https://github.com/harmanjeet27/spam-email.git
cd spam-email
pip install -r requirements.txt


