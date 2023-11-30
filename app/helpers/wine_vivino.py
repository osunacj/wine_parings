import requests
import pandas as pd
import numpy as np
import time

BASE_URL =  "https://www.vivino.com/api/explore/explore"
PATH = "./app/data/vivino_wines.csv"

def generate_params():
    ratings_range = np.arange(1,5,0.2)
    price_range = range(5,705,50)
    countries = ['it', 'fr', 'pt', 'es', 'au', 'us', 'ar', 'cl']
    grapes = range(1, 10, 1)
    params = []
    for rating in ratings_range:
        for j, price in enumerate(price_range):
            for country in countries:
                for grape in grapes:
                    params.append({
                        "price_range_max":str(price_range[j+1]) if price < 500 else '500',
                        "price_range_min":str(price),
                        "min_rating":str(rating),
                        "grape_ids[]": str(grape),
                        "country_codes[]": country,
                    })
    return params

def parse_wine(wine):
    wine_dict = {}
    wine_dict['Name'] = wine['vintage']["wine"]["name"]
    wine_dict['Id'] = wine['vintage']["wine"]["id"]
    wine_dict['Region'] = wine['vintage']['wine']['region']['name']
    wine_dict['Country'] = wine['vintage']['wine']['region']['country']['name']
    wine_dict['Winery'] = wine['vintage']['wine']['winery']['name']
    wine_dict['Taste Structure'] = wine['vintage']['wine']['taste']['structure']
    wine_dict['Ratings'] = wine['vintage']['wine']["statistics"]["ratings_count"]
    wine_dict['Score'] = wine['vintage']['wine']['statistics']['ratings_average']
    wine_dict['Status'] = wine['vintage']['wine']['statistics']['status']
    wine_dict['Varietal'] = wine['vintage']['wine']['style']['varietal_name']



    return wine_dict

def save_data_csv(data: list):
    historic_data = pd.read_csv(PATH)
    new_wine_data = pd.DataFrame(data)
    wine_data = pd.concat(
        [historic_data, new_wine_data]
    ).drop_duplicates()
    wine_data.to_csv(PATH)

def main():
    const = {
        "country_code": "NL",
        "currency_code":"EUR",
        "grape_filter":"varietal",
        "order_by":"price",
        "order":"asc"
    }
    headers= {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:66.0) Gecko/20100101 Firefox/66.0"
    }
    params = generate_params()
  

    for param in params[:5]:
        all_wine_info = []
        has_wines = True
        page = 1
        while has_wines and page < 5:
            try:
                response = requests.get(
                    BASE_URL,
                    params={**param, **const, "page": page},
                    headers=headers
                )

                if response.status_code != 200:
                    break

                response = response.json()['explore_vintage']
                records = response['records_matched']

                wine_records = response['matches']

                if len(wine_records) < 100 or records < 100:
                    break

                for wine_record in wine_records:
                    all_wine_info.append(
                        parse_wine(
                            wine_record
                        )
                    )
                page += 1
                
            except:
                continue



if __name__ == '__main__':
    main()