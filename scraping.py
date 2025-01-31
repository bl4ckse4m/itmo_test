# https://news.itmo.ru/module/sitemap.php
import logging
from datetime import datetime
from itertools import batched

import bs4
import requests
from bs4 import BeautifulSoup
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

import store
from utils.logger import setup_logging

log = logging.getLogger(__name__)


def download_sitemap(url):
    log.info(f'Downloading sitemap from {url}...')
    response = requests.get(url)
    return response.text


def parse_sitemap(sitemap_xml):
    log.info(f'Parsing sitemap...')
    sitemap = BeautifulSoup(sitemap_xml, "xml")
    urls = []
    for loc in sitemap.find_all('url'):
        url = loc.find('loc').get_text()
        if '/news/' in url:
            date = loc.find('lastmod').get_text()
            ts = datetime.strptime(date, "%Y-%m-%d").timestamp()
            urls.append((url, ts))
    urls.sort(key=lambda x: x[1], reverse=True)
    urls = [u[0] for u in urls[:10000]]
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
    log.info(f'Loading {len(urls)} urls...')

    cnt = 0
    all_splits = []
    for batch in list(batched(urls, 100)):
        log.info(f'Loading chunk {cnt}...')
        # Fetch and parse news pages
        loader = WebBaseLoader(
            requests_per_second=20,
            web_paths=batch,
            bs_kwargs=dict(
                parse_only=bs4.SoupStrainer(
                    class_=("lead", "post-content")
                )
            ),
        )
        docs = loader.load()
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        all_splits += text_splitter.split_documents(docs)
        cnt += 1

    store.add_docs(all_splits)
    print(f"Scraping completed. News articles stored in FAISS index")


if __name__ == "__main__":
    setup_logging()
    main()
