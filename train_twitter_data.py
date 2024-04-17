import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer

data = pd.read_csv("data/project_dataset.csv")
new_data = pd.read_csv("data/reviews.csv")

data['target'] = data['target'].astype(int)
new_data['Sentiment'] = new_data['Sentiment'].astype(int)

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(new_data['Text'])

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(new_data['Sentiment'], y_pred)
print("Accuracy:", accuracy)
