import requests
import pandas as pd
import numpy as np
import time

BASE_URL = "https://www.vivino.com/api/explore/explore"
PATH = "./app/data/vivino_wines.csv"
headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0",
    "Accept-Language": "en-US,en;q=0.9",
}


def generate_params():
    ratings_range = np.arange(3, 5, 0.2)
    price_range = range(5, 705, 100)
    # countries = ["", "it", "fr", "pt", "es", "au", "us", "ar", "cl", "at"]
    grapes = range(0, 20, 1)
    params = []
    for rating in ratings_range:
        for j, price in enumerate(price_range):
            # for country in countries:
                for grape in grapes:
                    params.append(
                        {
                            "price_range_max": price_range[j + 1]
                            if price < 500
                            else 500,
                            "price_range_min": price,
                            "min_rating": rating,
                            "grape_ids[]": grape if grape != 0 else '',
                            "country_codes[]": '',
                        }
                    )
    return params


def taste_structure(structure: dict):
    del structure["user_structure_count"]
    del structure["calculated_structure_count"]
    for key, value in structure.items():
        if value is None:
            structure[key] = 0
            continue
        structure[key] = round(value, 3)
    clean_structure = str(structure).replace("'", "")
    clean_structure = clean_structure.replace("{", "")
    clean_structure = clean_structure.replace("}", "")
    return clean_structure


def get_flavor(flavors: list):
    clean_flavors = []
    for flavor in flavors:
        if "primary_keywords" not in flavor.keys():
            continue
        for primary_word in flavor["primary_keywords"]:
            if primary_word["count"] > 3:
                clean_flavors.append(primary_word["name"])
    return ", ".join(clean_flavors)


def get_style(style) -> dict:
    wine_dict = {}
    if style is None:
        return {}

    wine_dict["foods"] = get_food(style["food"])
    wine_dict["grapes"] = get_grapes(style["grapes"])
    wine_dict["varietal"] = style["varietal_name"]
    wine_dict["description"] = style["description"]
    wine_dict["fast_description"] = (
        style["body_description"]
        + " "
        + str(style["body"])
        + ", Acidity "
        + style["acidity_description"]
        + " "
        + str(style["acidity"])
    )
    return wine_dict

def get_grapes(grapes: list):
    clean_grapes = ""
    if grapes:
        for grape in grapes:
            clean_grapes = grape["name"] + ", " + clean_grapes
        return clean_grapes.strip()


def get_food(foods: list):
    clean_foods = ""
    for food in foods:
        clean_foods = food["name"] + ", " + clean_foods
    return clean_foods.strip()


def get_reviews(id, year) -> str:
    params = {"page": 1, "year": year, "per_page": 4, "language": "en"}
    wine_review = ""
    try:
        response = requests.get(
            f"https://www.vivino.com/api/wines/{id}/reviews",
            params=params,
            headers=headers,
        )

        if response.status_code != 200:
            return ""

        response = response.json()["reviews"]

        for review in response:
            wine_review = review["note"] + "\n " + wine_review
        return wine_review
    except:
        return ""


def parse_wine(wine):
    wine_dict = {}
    wine_dict["name"] = wine["vintage"]["wine"]["name"]
    wine_dict["year"] = wine["vintage"]["year"] if wine["vintage"]["year"] else None
    wine_dict["vintage_id"] = wine["vintage"]["id"]
    wine_dict['id'] = wine["vintage"]['wine']["id"]
    wine_dict["ratings"] = wine["vintage"]["statistics"]["wine_ratings_count"]
    wine_dict["score"] = wine["vintage"]["statistics"]["wine_ratings_average"]
    wine_dict["status"] = wine["vintage"]["statistics"]["status"]
    wine_dict["region"] = wine["vintage"]["wine"]["region"]["name"]
    wine_dict["country"] = wine["vintage"]["wine"]["region"]["country"]["name"]
    wine_dict["winery"] = wine["vintage"]["wine"]["winery"]["name"]
    wine_dict["flavors"] = get_flavor(wine["vintage"]["wine"]["taste"]["flavor"])
    wine_dict.update(get_style(wine["vintage"]["wine"]["style"]))

    if wine["vintage"]["wine"]["taste"]["structure"]:
        wine_dict["taste_structure"] = taste_structure(
            wine["vintage"]["wine"]["taste"]["structure"]
        )
    if wine["vintage"]["wine"]["has_valid_ratings"]:
        wine_dict["reviews"] = get_reviews(wine_dict["id"], wine_dict["year"])

    wine_dict["price"] = wine["price"]["amount"]

    return wine_dict


def save_data_csv(data: list):
    new_wine_data = pd.DataFrame(data)
    new_wine_data.to_csv(PATH, mode = 'a', index=False)
    


def extract_wine(wine_records) -> list:
    all_wine_info = []
    for wine_record in wine_records:
        all_wine_info.append(parse_wine(wine_record))
    return all_wine_info


def main():
    const = {
        "country_code": "NL",
        "currency_code": "EUR",
        "grape_filter": "varietal",
        "order_by": "ratings_count",
        "order": "desc",
        "per_page": 50,
        "language": "en",
    }
    params = generate_params()

    all_wine_info = []
    wine_ids = []
    indx = 432
 
 

    for param in params[434::2]:
        try:
            indx += 2
            payload = {**param, **const, "page": 1}
            response = requests.get(BASE_URL, params=payload, headers=headers)

            if response.status_code != 200:
                print(response.status_code, response.reason)
                continue

            records = response.json()["explore_vintage"]["records_matched"]
            wine_records = response.json()["explore_vintage"]["matches"]
            all_wine_info.extend(extract_wine(wine_records=wine_records))
            pages = round(records / 50)
            current_count= len(all_wine_info)
            if current_count % 5 == 0:
                print(f"Wine count: {current_count} and page: 1 / {pages}, idx: {indx}")
 

            for page in range(2, pages + 1):
                payload['page'] = page
                response = requests.get(BASE_URL, params=payload, headers=headers)
                wine_records = response.json()["explore_vintage"]["matches"]
                
                try:
                    all_wine_info.extend(extract_wine(wine_records=wine_records))
                    current_count= len(all_wine_info)

                    if current_count % 5 == 0 or current_count % 2 == 0:
                        print(f"Wine count: {current_count} and page: {page} / {pages}, idx: {indx}")

                    if current_count % 15 == 0 or current_count % 10 == 0:
                        save_data_csv(all_wine_info)
                        all_wine_info = []
                except:
                    continue

            if current_count % 15 == 0:
                print(f"Wine count: {current_count} and idx: {indx}") 
                save_data_csv(all_wine_info)
                all_wine_info = []
               
        except:
            continue
        
    save_data_csv(all_wine_info)
    data = pd.read_csv(PATH, index_col='index').drop_duplicates()
    data.to_csv(PATH, mode='w', index=False)
    print('History data: ', len(data))
    del data 
        

if __name__ == "__main__":
    main()
