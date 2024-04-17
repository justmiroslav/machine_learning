import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline

data = pd.read_csv("data/large_reviews.csv")

X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Rating'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

model = make_pipeline(LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
random_indices = data.sample(5).index
random_reviews = data.loc[random_indices, 'Review']
random_ratings = model.predict(tfidf_vectorizer.transform(random_reviews))
print("Random Reviews and Predicted Ratings:")
for review, rating in zip(random_reviews, random_ratings):
    print(f"Review: {review}\nPredicted Rating: {rating}")
