from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from bs4 import BeautifulSoup
import pandas as pd
import time
import os

class GMKScraper:
    def __init__(self, url, target_date):
        """
        Initializes the GMKScraper class.

        Args:
        - url: The URL of the website to scrape.
        - target_date: The desired date to search for in the scraped data.
        """
        self.data_path = os.path.dirname(os.path.abspath(__file__)) + '/data'
        self.url = url
        self.target_date = target_date
        self.driver = webdriver.Chrome()
        self.data = []
        self.visited_links = set()
        self.found_desired_date = False

    def scrape(self):
        """
        Scrapes the website for articles until the desired date is found.
        """
        self.driver.get(self.url)

        while not self.found_desired_date:
            soup = BeautifulSoup(self.driver.page_source, 'html.parser')
            articles = soup.find_all('li')

            for article in articles:
                link_element = article.find('a')['href']

                if link_element not in self.visited_links:
                    title_element = article.find('span', class_='title-post')
                    date_element = article.find_previous('div', class_='day-date')

                    if title_element and date_element:
                        title = title_element.text
                        date = date_element.text

                        self.driver.get(link_element)
                        time.sleep(2)

                        page_source = self.driver.page_source
                        link_soup = BeautifulSoup(page_source, 'html.parser')

                        article_info = link_soup.find('div', class_='article-info')
                        date_in_body = article_info.find('span', class_='article-date').text.strip() if article_info else None

                        article_body = link_soup.find('div', itemprop='articleBody', class_='post-body')
                        if article_body:
                            headline_element = article_body.find('h2', itemprop='headline')
                            headline = headline_element.text if headline_element else None

                            published_time = article_body.find('meta', property='article:published_time')['content']
                            content_paragraphs = article_body.find_all('p')
                            content = '\n'.join([p.text for p in content_paragraphs])

                            self.data.append({'Date': date if date else date_in_body, 'Date_body': date_in_body if date_in_body else None,
                                              'Title': title, 'Link': link_element, 'Headline': headline,
                                              'Published Time': published_time, 'Content': content})

                            if date_in_body == self.target_date:
                                self.found_desired_date = True
                                break

                        self.driver.back()
                        time.sleep(2)

                        self.visited_links.add(link_element)

            if not self.found_desired_date:
                try:
                    load_more_button = self.driver.find_element(By.CSS_SELECTOR, 'div.loadmore.btn')
                    load_more_button.click()
                    time.sleep(3)
                except:
                    break

        self.driver.quit()

    def save_to_csv(self):
        """
        Saves the scraped data to a CSV file.
        """
        df = pd.DataFrame(self.data)
        df.to_csv(f'{self.data_path}/gmk/gmk.csv', index=False)