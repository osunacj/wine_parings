from requests import request
import time
import pandas as pd
import json


BASE_URL = "https://hzoxaduqy8-dsn.algolia.net/1/indexes/*/queries?x-algolia-agent=Algolia for JavaScript (4.17.2); Browser (lite); instantsearch.js (4.56.3); JS Helper (3.13.2)&x-algolia-api-key=e9abd7ddf7b59423aceea6146888507c&x-algolia-application-id=HZOXADUQY8"


def define_payload():
    tex = 'facetFilters=[["wine_type:Red"],["varietal_label:Pinot Noir"]]&hitsPerPage=1000&maxValuesPerFacet=100&numericFilters=["rating>=90"]&page=0'
    return {
        "requests": [
            {
            "indexName": "PROD_WINEENTHUSIAST_REVIEWS",
            "params": tex
            }
        ]
    }




def main():
    headers = {
    'Content-Type': 'text/plain'
    }
    payload = define_payload()

    response = request('POST', BASE_URL, headers=headers, json=payload)
    data = response.json()
    print(data)

if __name__ == '__main__':
    main()
