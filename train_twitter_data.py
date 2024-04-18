import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("data/project_dataset.csv")
new_data = pd.read_csv("data/reviews.csv")

# Ensure correct data type and map labels
data['target'] = data['target'].astype(int).replace({4: 1})
new_data['Sentiment'] = new_data['Sentiment'].astype(int).replace({4: 1})

X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, random_state=42)

tfidf_vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2), max_features=10000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(new_data['Text'])

model = LogisticRegression()
model.fit(X_train_tfidf, y_train)

y_pred = model.predict(X_test_tfidf)
y_scores = model.decision_function(X_test_tfidf)

accuracy = accuracy_score(new_data['Sentiment'], y_pred)
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", confusion_matrix(new_data['Sentiment'], y_pred))
print("\nClassification Report:\n", classification_report(new_data['Sentiment'], y_pred))

# Displaying feature names and idf scores
features_df = pd.DataFrame(tfidf_vectorizer.get_feature_names_out(), columns=['Feature'])
features_df['IDF Score'] = tfidf_vectorizer.idf_
print(features_df.sort_values(by='IDF Score', ascending=False).head(20))

# Multinomial NB test
nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)


nb_y_pred = nb_model.predict(X_test_tfidf)
nb_y_scores = nb_model.predict_proba(X_test_tfidf)[:, 1]


nb_accuracy = accuracy_score(new_data['Sentiment'], nb_y_pred)
print("\n\nMultinomialNB accuracy:", nb_accuracy)
print("Confusion Matrix:\n", confusion_matrix(new_data['Sentiment'], nb_y_pred))
print("\nClassification Report:\n", classification_report(new_data['Sentiment'], nb_y_pred))


fpr_nb, tpr_nb, threshold_nb = roc_curve(new_data['Sentiment'], nb_y_scores)
roc_auc_nb = auc(fpr_nb, tpr_nb)


plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='Logistic Regression ROC curve (area = %0.2f)' % roc_auc)
plt.plot(fpr_nb, tpr_nb, color='blue', lw=2, label='Multinomial Naive Bayes ROC curve (area = %0.2f)' % roc_auc_nb)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()