# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import nltk
# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
# Sample dataset with more examples
data = [
    ("I love this product, it's amazing!", "positive"),
    ("This is the best thing I have ever bought.", "positive"),
    ("Absolutely fantastic! Worth every penny.", "positive"),
    ("I hate this item, it's terrible.", "negative"),
    ("This is the worst experience I had.", "negative"),
    ("Not recommended, very disappointing.", "negative"),
    ("I am very satisfied with this purchase.", "positive"),
    ("This product did not meet my expectations.", "negative"),
    ("I would buy this again, it's great!", "positive"),
    ("It broke after one use, very bad quality.", "negative")
]
# Separate the dataset into text and labels
texts, labels = zip(*data)

# Preprocess the text data
vectorizer = CountVectorizer(stop_words=stop_words)
X = vectorizer.fit_transform(texts)
# Train a simple Naive Bayes classifier using Cross-Validation for small datasets
classifier = MultinomialNB()
scores = cross_val_score(classifier, X, labels, cv=5)  # 5-fold cross-validation
print(f"Cross-Validation Accuracy: {scores.mean() * 100:.2f}%")

# Train on all data for demonstration
classifier.fit(X, labels)

# Test with new data
new_texts = ["I love it!", "This is awful.", "Quite good, I am happy with it.", "Not worth the price."]
new_texts_vectorized = vectorizer.transform(new_texts)
predictions = classifier.predict(new_texts_vectorized)

# Display predictions
for text, prediction in zip(new_texts, predictions):
    print(f"Text: {text} -> Sentiment: {prediction}")
    
