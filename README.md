# sms_spam_claasifi
1. Data Import
   Loaded the SMS spam dataset (CSV format) into a pandas DataFrame for analysis and preprocessing.

2. Data Cleaning
Dropped irrelevant columns (last 3 columns not useful for prediction).
Renamed columns for clarity (v1 → target, v2 → text).
Mapped target labels (spam, ham) to numerical values.
Checked and handled missing values.
Removed duplicate messages to avoid bias in training.

3. Exploratory Data Analysis (EDA)
Counted the number of spam vs ham messages — identified class imbalance.
Analyzed:
Number of characters, words, and sentences per message.
Computed correlation matrix for numerical features to understand relationships.

4. Data Preprocessing
Converted all messages to lowercase.

Tokenized text into individual words.

Removed special characters, punctuation, and stopwords.

Applied Stemming to normalize words.

Visualized most frequent words:

Top 30 words in Spam messages.

Top 30 words in Ham messages.

5. Model Building
Converted text data to numerical format using:

Bag of Words (CountVectorizer)

TF-IDF (TfidfVectorizer)
Split the dataset into training and test sets.

Trained multiple classifiers:

Naive Bayes: GaussianNB, MultinomialNB, BernoulliNB
➤ TfidfVectorizer + MultinomialNB performed best

6. Model Evaluation
   Achieved highest accuracy and precision, especially on imbalanced classes.

7. Deployment Ready
Saved final model (MultinomialNB) and vectorizer (TfidfVectorizer) using pickle.

Built a Streamlit web app for interactive predictions.
