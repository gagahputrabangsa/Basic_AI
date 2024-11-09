import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline


# Load training data (replace with your own data)
ham_emails = [
    "This is a legitimate email from a friend.",
    "Meeting tomorrow at 2 PM. Please confirm.",
    "Order confirmation for your recent purchase."
]
spam_emails = [
    "You've won a million dollars! Click here to claim.",
    "Buy cheap Viagra now!",
    "Urgent: Please send money to this account."
]

# Preprocess data
def preprocess_email(email):
    # Convert to lowercase
    email = email.lower()

    # Remove stop words
    stop_words = set(stopwords.words('english'))
    words = [word for word in word_tokenize(email) if word not in stop_words]

    # Join words back into a string
    return ' '.join(words)

