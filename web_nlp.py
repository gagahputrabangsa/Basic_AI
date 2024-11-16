import requests
from bs4 import BeautifulSoup
import re
from transformers import pipeline
def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # search for content with paragraph tag
    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])

    return content


def clean_text(text):
    text = re.sub(r'\s+', ' ', text)  # remove extra space  
    text = re.sub(r'\[.*?\]', '', text)  # remove ()
    return text.strip()



# Creating QnA Model
qa_pipeline = pipeline("question-answering")

def answer_question(article_content, question):
    # using pipeline to answer the Q
    result = qa_pipeline(question=question, context=article_content)
    return result['answer']
def main(url, question):
    # Mengambil konten dari link
    article_content = get_article_content(url)
    # Membersihkan konten artikel
    clean_content = clean_text(article_content)
    # Menjawab pertanyaan
    answer = answer_question(clean_content, question)

    return answer

