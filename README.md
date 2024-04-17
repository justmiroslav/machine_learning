## Datasets

**project_dataset.csv** This dataset is used in the ```train_twitter_data.py``` script. It contains Twitter data that has been preprocessed and formatted for sentiment analysis. The dataset includes the following columns:  
  + ```target```: This is the sentiment of the tweet. It's an integer value where 0 represents negative sentiment and 1 represents positive sentiment.
  + ```date```: The date and time when the tweet was posted.
  + ```user```: The username of the person who posted the tweet.
  + ```text```: The text of the tweet.

**reviews.csv** This dataset is also used in the ```train_twitter_data.py``` script. It contains reviews data that is used to test the trained model. The dataset includes the following columns:  
  + ```Sentiment```: This is the sentiment of the review. It's an integer value where 0 represents negative sentiment and 1 represents positive sentiment.
  + ```Text```: The text of the review.

**large_reviews.csv** This dataset is used in the ```train_review_data.py``` script. It contains a large number of reviews that are used to train a Logistic Regression model. The dataset includes the following columns:
  + ```Review```: The text of the review.
  + ```Rating```: The rating of the review.

## Model

The model used in both scripts is a Logistic Regression model. In ```train_twitter_data.py```, the model is trained on the **project_dataset.csv** and tested on the **reviews.csv**. In ```train_review_data.py```, the model is trained and tested on the **large_reviews.csv**.  The model uses a TfidfVectorizer to convert the text data into a matrix of TF-IDF features. The TfidfVectorizer is configured to analyze words, use a n-gram range of (1, 2), and limit the number of features to 10000.

## Results

The *accuracy* of the model is printed at the end of each script and is calculated as the number of correct predictions divided by the total number of predictions. In ```train_twitter_data.py```, the *accuracy* is ```0.8958333333333334``` that is a very good performance. In ```train_review_data.py```, it equals ```0.8824166666666666```, which is also very nice. In addition to the *accuracy*, ```train_review_data.py``` also prints out the predicted ratings for 5 random reviews from the **large_reviews.csv** dataset.
