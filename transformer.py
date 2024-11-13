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
