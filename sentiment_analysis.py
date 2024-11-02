import nltk
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# Download necessary NLTK data
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('movie_reviews')

# Load positive and negative movie reviews
positive_reviews = [(word_tokenize(nltk.corpus.movie_reviews.raw(fileid)), 'pos')
                    for fileid in nltk.corpus.movie_reviews.fileids('pos')]
negative_reviews = [(word_tokenize(nltk.corpus.movie_reviews.raw(fileid)), 'neg')
                    for fileid in nltk.corpus.movie_reviews.fileids('neg')]

# Combine positive and negative reviews
reviews = positive_reviews + negative_reviews

# Define a feature extractor to extract features from the reviews
def extract_features(review):
    words = set(review)
    features = {}
    for word in words:
        features[word] = True
    return features

# Train the Naive Bayes classifier
train_data = [(extract_features(review), sentiment) for review, sentiment in reviews]
classifier = NaiveBayesClassifier.train(train_data)

# Input sentences to classify
sentences = [
    "The movie was terrible, I hated it.",
    "The food was delicious, I loved it.",
    "The weather is cloudy today.",
    "The concert was amazing, I had a great time.",
    "The book was boring, I didn't enjoy it.",
    # "the food is good, but it taste horrible!"
]

# Classify each sentence and print the result
for sentence in sentences:
    words = word_tokenize(sentence)
    features = extract_features(words)
    sentiment = classifier.classify(features)
    print(sentence, "->", sentiment)
