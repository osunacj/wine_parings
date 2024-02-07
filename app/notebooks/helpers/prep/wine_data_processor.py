import pandas as pd
import numpy as np
import os

from .variety_mappings import variety_mapping


BASE_PATH = "./app/data/wine_extracts"
geographies = ["Subregion", "Region", "Province", "Country"]


def merge_all_wine_files(write=False):
    i = 0
    wine_dataframe = pd.DataFrame()
    for file in os.listdir(BASE_PATH):
        file_location = BASE_PATH + "/" + str(file)
        if i == 0:
            wine_dataframe = pd.read_csv(file_location)
            i += 1
        else:
            df_to_append = pd.read_csv(
                file_location, low_memory=False, encoding="latin-1"
            )
            wine_dataframe = pd.concat(
                [wine_dataframe, df_to_append], axis=0, ignore_index=False
            )

    wine_dataframe.drop_duplicates(subset=["Name"], inplace=True)

    for geo in geographies:
        wine_dataframe[geo] = wine_dataframe[geo].apply(lambda x: str(x).strip())

    if write:
        wine_dataframe.drop(["Unnamed: 0"], inplace=True, axis=1)
        wine_dataframe.to_csv("./app/data/produce/wine_data.csv", index_label=False)

    print(wine_dataframe.shape)
    return wine_dataframe


def consolidate_varieties(variety_name):
    if variety_name in variety_mapping:
        return variety_mapping[variety_name]
    else:
        return variety_name


# replace any nan values in the geography columns with the word none
def replace_nan_for_zero(value):
    if str(value) == "0" or str(value) == "nan":
        return "none"
    else:
        return value


def preprocess_wine_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    variety_geo_df = pd.read_csv(
        "./app/data/produce/varieties_all_geos_normalized.csv",
        index_col=0,
        usecols=[
            "Variety",
            "Country",
            "Province",
            "Region",
            "Subregion",
            "geo_normalized",
        ],
    )

    df.loc[:, "Variety"] = df.loc[:, "Variety"].apply(consolidate_varieties)

    for geography in geographies:
        df.loc[:, geography] = df.loc[:, geography].apply(replace_nan_for_zero)

    df.loc[:, geographies].fillna("none", inplace=True)

    variety_geo = (
        df.groupby(["Variety", "Country", "Province", "Region", "Subregion"])
        .size()
        .reset_index()
        .rename(columns={0: "count"})
    )

    vgeos_df = pd.merge(
        left=pd.DataFrame(
            variety_geo.loc[variety_geo["count"] > 1],
            columns=["Variety", "Country", "Province", "Region", "Subregion", "count"],
        ),
        right=variety_geo_df,
        left_on=["Variety", "Country", "Province", "Region", "Subregion"],
        right_on=["Variety", "Country", "Province", "Region", "Subregion"],
    )

    df = pd.merge(
        left=df,
        right=vgeos_df,
        left_on=["Variety", "Country", "Province", "Region", "Subregion"],
        right_on=["Variety", "Country", "Province", "Region", "Subregion"],
    )

    df.drop(
        [
            "Appellation",
            "Bottle Size",
            "Date Published",
            "Designation",
            "Importer",
            "Rating",
            "Reviewer",
            "Reviewer Twitter Handle",
            "Subregion",
            "User Avg Rating",
            "count",
        ],
        axis=1,
        inplace=True,
    )

    variety_geos = df.groupby(["Variety", "geo_normalized"]).size()
    at_least_n_types = variety_geos[variety_geos > 30].reset_index()

    df = pd.merge(
        df,
        at_least_n_types,
        left_on=["Variety", "geo_normalized"],
        right_on=["Variety", "geo_normalized"],
    )

    print(df.shape)

    df.to_csv("./app/data/production/wines.csv", index_label=False)
    return df
