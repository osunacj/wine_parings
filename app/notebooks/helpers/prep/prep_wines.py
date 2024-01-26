import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

BASE_PATH = "./app/data/wine_extracts"


def  main():

    i = 0
    for file in os.listdir(BASE_PATH):
        file_location = BASE_PATH + '/' + str(file)
        if i==0:
            wine_dataframe = pd.read_csv(file_location, encoding='latin-1')
            i+=1
        else:
            df_to_append = pd.read_csv(file_location, encoding='latin-1', low_memory=False)
            wine_dataframe = pd.concat([wine_dataframe, df_to_append], axis=0)

    wine_dataframe.drop_duplicates(subset=['Name'], inplace=True)

    geographies = ['Subregion', 'Region', 'Province', 'Country']

    for geo in geographies:
        wine_dataframe[geo] = wine_dataframe[geo].apply(lambda x : str(x).strip())

    print(wine_dataframe.shape)




if __name__ == '__main':
    main()