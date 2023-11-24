import requests
from bs4 import BeautifulSoup
import datetime
import pandas as pd
from time import sleep

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By

path = "./app/data/wine_pages_to_mine.csv"


def print_pages(page_num, pages_to_mine):
    if page_num % 25 == 0 or pages_to_mine % 50 == 0:
        print(f"Pages:  {page_num} and {pages_to_mine}")


def press_button(browser, button_selector):
    try:
        button = browser.find_element(By.CSS_SELECTOR, button_selector)
        browser.execute_script("arguments[0].click()", button)
    except Exception:
        raise Exception("Failed to parse")


def scrape_wine_links(base_url, min_page_number=1, max_page_number=5):
    wine_pages_to_mine = []
    current_len = 0
    historic_len = 1
    epoch = 1
    browser.get(base_url)
    browser.implicitly_wait(1)
    while current_len != historic_len:
        print_pages(epoch, current_len)
        try:
            all_wine_links = browser.find_elements(
                By.CSS_SELECTOR, "a.ratings-block__cta"
            )
            all_wine_links = [
                wine_link.get_attribute("href") for wine_link in all_wine_links
            ]
            wine_pages_to_mine.extend(all_wine_links)

            historic_len = current_len
            current_len = len(wine_pages_to_mine)

            epoch += 1
            press_button(browser, 'a[aria-label="Next"]')
        except:
            continue

    series_wine_pages = pd.Series(wine_pages_to_mine)
    series_wine_pages.to_csv(path, index=False)
    print(f"Pages:  {epoch} and {current_len}")
    return wine_pages_to_mine


class WineInfoScraper:
    def __init__(self, link, browser):
        self.page = link
        self.browser = browser

    def get_wine_name(self, browser, selector):
        wine_name_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_name_clean = wine_name_raw.text
        return wine_name_clean

    def get_vintage(self, wine_name_clean):
        name_strings = wine_name_clean.split(" ")
        number_strings = [i for i in name_strings if (i.isnumeric())]
        for n in number_strings:
            if 1900 < int(n) < datetime.datetime.now().year:
                return n
            else:
                continue

    def get_wine_description(self, browser, selector):
        wine_description_clean = (
            browser.find_element(By.CSS_SELECTOR, selector).text.strip().split("— ")
        )

        if len(wine_description_clean) > 1:
            wine_reviewer = wine_description_clean[1]
        else:
            wine_reviewer = None
        return wine_description_clean[0], wine_reviewer

    def get_wine_info(self, browser, selector):
        wine_info_dict = {}
        wine_info_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_info_list = wine_info_raw.text.split("\n")
        for index, property in enumerate(wine_info_list):
            if index % 2 == 0:
                value = wine_info_list[index + 1]
                wine_info_dict[property.capitalize()] = (
                    value if value != "N/A" else None
                )
        return wine_info_dict

    def get_wine_location(self, browser, selector):
        wine_info_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_info_list = wine_info_raw.text.split(", ")
        return {
            "Subregion": wine_info_list[0],
            "Country": wine_info_list[-1],
        }

    def get_food_pairings_link(self, browser, selector):
        food_pairing_link_raw = browser.find_element(By.CSS_SELECTOR, selector)
        food_pairing_link = food_pairing_link_raw.get_attribute("href")
        return food_pairing_link

    # def get_food_pairing(self, browser, selector, link):
    #     try:
    #         if "gopjn" not in link:
    #             browser.get(link + "#t2")
    #             food_pairing_raw = browser.find_element(By.CSS_SELECTOR, selector)
    #             food_pairing_clean = food_pairing_raw.text.split(" ")
    #             return food_pairing_clean
    #         return ""

    #     except:
    #         pass

    def scrape_all_info(self):
        wine_info_dict = {}
        self.browser.get(self.page)

        wine_info_dict["Name"] = self.get_wine_name(self.browser, "h1.review-title")
        wine_info_dict["Vintage"] = self.get_vintage(wine_info_dict["Name"])
        (
            wine_info_dict["Description"],
            wine_info_dict["Reviwer"],
        ) = self.get_wine_description(self.browser, "div.row.review__row")
        wine_info_dict.update(self.get_wine_info(self.browser, "div.meta"))
        wine_info_dict.update(
            self.get_wine_location(self.browser, "div.winery-location")
        )
        wine_info_dict["Food Pairing Link"] = self.get_food_pairings_link(
            self.browser, "a.btn.btn-secondary"
        )
        wine_info_dict["Link"] = self.page

        return wine_info_dict


def mine_all_wine_info(browser):
    # all_wine_links = pd.read_csv(path)
    all_wine_links = scrape_wine_links(
        base_url="https://www.wineenthusiast.com/ratings/",
        min_page_number=1,
        max_page_number=700,
    )

    all_wine_info = []
    for link in all_wine_links:
        try:
            scraper = WineInfoScraper(link, browser)
            wine_info = scraper.scrape_all_info()
            all_wine_info.append(wine_info)
        except:
            continue
        # sleep(5)

    full_wine_info_dataframe = pd.DataFrame(all_wine_info)

    full_wine_info_dataframe.to_csv(
        "./app/data/all_scraped_wine_info_wine_enthusiast.csv"
    )


if __name__ == "__main__":
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    browser = webdriver.Chrome(
        options=options, service=Service(ChromeDriverManager().install())
    )
    mine_all_wine_info(browser)
