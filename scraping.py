# https://news.itmo.ru/module/sitemap.php

import bs4
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import store


def download_sitemap(url):
    response = requests.get(url)
    return response.text

def parse_sitemap(sitemap_xml):
    sitemap = BeautifulSoup(sitemap_xml, "xml")
    urls = []
    for loc in sitemap.find_all('loc', limit=10):
        loc = loc.get_text()
        if '/news/'in loc:
            urls.append(loc)
    return urls

def fetch_and_parse_news_page(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')
    title = soup.find('h1', class_='article-title').text.strip()
    # Extract content
    content_div = soup.find('div', class_='article-content js-mediator-article')
    content = content_div.text.strip() if content_div else ""
    return {'title': title, 'content': content}

def get_embeddings(texts):
    loader = WebBaseLoader()
    return [loader.load_text(text) for text in texts]

def main():
    # Download and parse sitemap
    sitemap_url = "https://news.itmo.ru/module/sitemap.php"
    sitemap_xml = download_sitemap(sitemap_url)
    urls = parse_sitemap(sitemap_xml)

    # Fetch and parse news pages
    loader =WebBaseLoader(
        web_paths=urls,
        bs_kwargs=dict(
            parse_only=bs4.SoupStrainer(
                class_=("lead", "post-content")
            )
        ),
    )
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(docs)
    store.add_docs(all_splits)

    print(f"Scraping completed. News articles stored in FAISS index")

if __name__ == "__main__":
    main()