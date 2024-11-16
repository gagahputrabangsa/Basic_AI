import requests
from bs4 import BeautifulSoup
import re
def get_article_content(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')

    # search for content with paragraph tag
    paragraphs = soup.find_all('p')
    content = ' '.join([para.get_text() for para in paragraphs])

    return content

