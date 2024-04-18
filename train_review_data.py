import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_pipeline
import seaborn as sns
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/large_reviews.csv")

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(data['Review'], data['Rating'], test_size=0.2, random_state=42)

# Initialize a TF-IDF Vectorizer and transform the datasets
tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)

# Create a pipeline with Logistic Regression
model = make_pipeline(LogisticRegression(max_iter=1000))
model.fit(X_train_tfidf, y_train)

# Predict and evaluate the model
y_pred = model.predict(X_test_tfidf)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Print a classification report
print("Classification Report:\n", classification_report(y_test, y_pred))

# Print the confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(cm, annot=True, fmt='d')
plt.title('Confusion Matrix')
plt.ylabel('True label')
plt.xlabel('Predicted label')
plt.show()

# Display random reviews with actual and predicted ratings
random_indices = data.sample(10).index
random_reviews = data.loc[random_indices, 'Review']
random_actual_ratings = data.loc[random_indices, 'Rating']
random_predicted_ratings = model.predict(tfidf_vectorizer.transform(random_reviews))

print("Random Reviews with Predicted Ratings:")
for review, actual, predicted in zip(random_reviews, random_actual_ratings, random_predicted_ratings):
    print(f"Review: {review}\nPredicted Rating: {predicted}\nActual Rating: {actual}\n")


# Display the feature names and idf scores
features_df = pd.DataFrame(tfidf_vectorizer.get_feature_names_out(), columns=['Feature Name'])
features_df['IDF Score'] = tfidf_vectorizer.idf_
print(features_df.sort_values(by='IDF Score', ascending=False).head(20))