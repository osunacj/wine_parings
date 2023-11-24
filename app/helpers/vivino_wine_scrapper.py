import datetime
import pandas as pd

import requests
import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager
from selenium.webdriver.common.by import By
from selenium.webdriver.support.wait import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

path = "./app/data/vivino_pages_to_mine.csv"


def print_pages(page_num, pages_to_mine, grape):
    if page_num % 25 == 0 or pages_to_mine % 75 == 0:
        print(f"Pages:  {page_num} and {pages_to_mine} and grape: {grape}")


def press_button(browser, button_selector="button.fc-cta-do-not-consent"):
    try:
        button_do_not_consent = browser.find_element(By.CSS_SELECTOR, button_selector)
        browser.execute_script("arguments[0].click()", button_do_not_consent)
    except Exception:
        button_do_not_consent = browser.find_element(By.CSS_SELECTOR, button_selector)
        browser.execute_script("arguments[0].click()", button_do_not_consent)


def scrape_wine_links(base_url, min_page_number=1, max_page_number=5):
    wine_pages_to_mine = []
    grapes = [
        "malbec",
        "merlot",
        "tempranillo",
        "cabernet sauvignon",
        "chardonnay",
        "grenache",
        "syrah",
        "shiraz",
        "pinot noir",
        "bordeaux",
        "rose",
        "porto",
        "sauvignon blanc",
        "pinot grigo",
        "zinfandel",
        "sangiovese",
        "sparkling",
        "champagne",
        "gewurztraminer",
        "gruner veltliner",
        "red",
        "white",
    ]
    historic_len = 1
    current_len = 0
    for grape in grapes:
        for page_number in range(min_page_number, max_page_number):
            url_to_mine = base_url + f"{grape}&start={page_number}"
            print_pages(page_number, current_len, grape)
            try:
                browser.get(url_to_mine)
                browser.implicitly_wait(1)

                if page_number == min_page_number and grape == grapes[0]:
                    press_button(browser)

                all_wine_links_raw = browser.find_elements(
                    By.CSS_SELECTOR, "span.wine-card__name"
                )

                for wine_link in all_wine_links_raw:
                    wine_link = wine_link.find_element(
                        By.CSS_SELECTOR, "a.link-color-alt-grey"
                    )
                    wine_pages_to_mine.append(
                        wine_link.get_attribute("href").replace("nl", "en")
                    )

                current_len = len(wine_pages_to_mine)
                if current_len == historic_len:
                    break

                historic_len = current_len
            except:
                continue

    series_wine_pages = pd.Series(wine_pages_to_mine)
    series_wine_pages.to_csv(path, index=False)
    print(f"Pages:  {max_page_number} and {len(wine_pages_to_mine)}")
    return wine_pages_to_mine


class WineInfoScraper:
    def __init__(self, link, browser):
        self.page = link
        self.browser = browser

    def get_wine_name(self, browser, selector):
        wine_name_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_name_clean = wine_name_raw.text
        return wine_name_clean.replace("\n", " ")

    def get_vintage(self, wine_name_clean):
        name_strings = wine_name_clean.split(" ")
        number_strings = [i for i in name_strings if (i.isnumeric())]
        for n in number_strings:
            if 1900 < int(n) < datetime.datetime.now().year:
                return n
            else:
                continue

    def get_price(self, browser, selector):
        wine_price_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_price_clean = wine_price_raw.text
        return wine_price_clean

    def get_wine_property(self, browser, selector):
        wine_text_raw = WebDriverWait(browser, 5).until(
            lambda driver: driver.find_element(By.CSS_SELECTOR, selector)
        )
        wine_text = wine_text_raw.text.split("\n")
        wine_dict = {
            wine_text[-2]: wine_text[-1],
            wine_text[-4]: wine_text[-3],
            wine_text[-6]: wine_text[-5],
        }
        return wine_dict

    def get_wine_taste(self, browser, selector):
        wine_taste_raw = WebDriverWait(browser, 5).until(
            lambda driver: driver.find_elements(By.CSS_SELECTOR, selector)
        )
        food_taste = " ".join(taste.text for taste in wine_taste_raw)
        return food_taste.strip()

    def get_wine_info(self, browser, selector):
        wine_info_dict = {}
        wine_info_raw = browser.find_elements(By.CSS_SELECTOR, selector)

        for property in wine_info_raw:
            attribute = property.get_attribute("data-cy")
            attribute = attribute.split("-")[1].capitalize()
            wine_info_dict[attribute] = property.text
        return wine_info_dict

    def get_wine_rating(self, browser, selector):
        wine_info_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_info_list = wine_info_raw.text.split("\n")
        return wine_info_list[1], wine_info_list[2].replace("ratings", "")

    def get_wine_review(self, browser, selector):
        wine_ratings_raw = browser.find_elements(By.CSS_SELECTOR, selector)
        wine_review = "\n".join(review.text for review in wine_ratings_raw)
        return wine_review

    def get_wine_location(self, browser, selector):
        wine_info_raw = browser.find_element(By.CSS_SELECTOR, selector)
        wine_info_list = wine_info_raw.text.split(", ")
        return {
            "Subregion": wine_info_list[0],
            "Country": wine_info_list[-1],
        }

    def get_food_pairings_link(self, browser, selector):
        food_pairing_section = browser.find_elements(By.CSS_SELECTOR, selector)

        food_pairing = ""
        for food in food_pairing_section:
            food_pairing = food_pairing + " " + food.text
        return food_pairing

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
        self.browser.implicitly_wait(3)

        wine_info_dict["Link"] = self.page
        wine_info_dict["Id"] = self.page().split("/")[-1]

        press_button(self.browser)
        page_height = self.browser.execute_script("return document.body.offsetHeight")
        browser_window_height = self.browser.get_window_size(windowHandle="current")[
            "height"
        ]
        current_position = self.browser.execute_script("return window.pageYOffset")
        while current_position - 600 < page_height + browser_window_height:
            self.browser.execute_script(
                f"window.scrollTo({current_position}, {browser_window_height + current_position});"
            )
            current_position = self.browser.execute_script("return window.pageYOffset")
            time.sleep(0.3)

        wine_info_dict["Name"] = self.get_wine_name(self.browser, "div.row.header")
        wine_info_dict["Vintage"] = self.get_vintage(wine_info_dict["Name"])
        wine_info_dict["Price"] = self.get_price(
            self.browser, "span.purchaseAvailability__currentPrice--3mO4u"
        )

        wine_info_dict["Score"], wine_info_dict["Num Ratings"] = self.get_wine_rating(
            self.browser, "div.row.location"
        )

        wine_info_dict.update(
            self.get_wine_info(
                self.browser, "a.anchor_anchor__m8Qi-.breadCrumbs__link--1TY6b"
            )
        )

        wine_info_dict["Food Pairing"] = self.get_food_pairings_link(
            self.browser, "a.foodPairing__imageContainer--2CtYR"
        )

        wine_info_dict.update(
            self.get_wine_property(self.browser, "table.wineFacts__wineFacts--2Ih8B")
        )

        wine_info_dict["Food Taste"] = self.get_wine_taste(
            self.browser, "div.tasteNote__popularKeywords--1gIa2"
        )

        wine_info_dict["Reviews"] = self.get_wine_review(
            self.browser, "span.communityReview__reviewText--2bfLj"
        )

        return wine_info_dict


def mine_all_wine_info(browser):
    # all_wine_links = pd.read_csv(path)
    all_wine_links = scrape_wine_links(
        base_url=f"https://www.vivino.com/search/wines?q=",
        min_page_number=1,
        max_page_number=51,
    )

    # all_wine_info = []
    # for link in all_wine_links["0"]:
    #     try:
    #         scraper = WineInfoScraper(link, browser)
    #         wine_info = scraper.scrape_all_info()
    #         all_wine_info.append(wine_info)
    #     except:
    #         continue
    #     # sleep(5)

    # full_wine_info_dataframe = pd.DataFrame(all_wine_info)

    # full_wine_info_dataframe.to_csv(
    #     "./app/data/vivno_scraped.csv"
    # )


if __name__ == "__main__":
    options = webdriver.ChromeOptions()
    options.add_argument("--headless")
    custom_user_agent = "Mozilla/5.0 (Linux; Android 11; 100011886A Build/RP1A.200720.011) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.69 Safari/537.36"
    options.add_argument(f"user-agent={custom_user_agent}")
    options.experimental_options["prefs"] = {
        "profile.managed_default_content_settings.images": 2
    }

    # proxy_server_url = "47.236.103.190"

    # options.add_argument(f"--proxy-server={proxy_server_url}")
    browser = webdriver.Chrome(
        options=options, service=Service(ChromeDriverManager().install())
    )
    mine_all_wine_info(browser)
