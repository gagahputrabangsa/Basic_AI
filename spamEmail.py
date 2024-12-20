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

# Create training data
X_train = [preprocess_email(email) for email in ham_emails + spam_emails]
y_train = [0] * len(ham_emails) + [1] * len(spam_emails)

# Create Naive Bayes classifier pipeline
classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),
    ('nb', MultinomialNB())
])

# Train the classifier
classifier.fit(X_train, y_train)

# Function to classify a new email
def classify_email(email):
    preprocessed_email = preprocess_email(email)
    prediction = classifier.predict([preprocessed_email])[0]
    if prediction == 0:
        return "not spam"
    else:
        return "spam"

# Test the classifier
new_emails = [
    "You won a free iPhone! Click here to claim.",
    "Meeting rescheduled for tomorrow at 3 PM.",
    "Urgent: Please wire me $1000.",
    "Claim Now! Free Iphone"
]

for email in new_emails:
    print(email, "->", classify_email(email))
