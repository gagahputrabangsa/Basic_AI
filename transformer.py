# Import libraries
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from nltk.corpus import stopwords
import nltk
# Download stopwords
nltk.download('stopwords')
stop_words = stopwords.words('english')
