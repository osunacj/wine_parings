import numpy as np
from scipy import spatial
import re
import ast


def weight_rule(df, food_weight):
    # Rule 1: the wine should have at least the same body as the food
    df = df.loc[(df["weight"] >= food_weight[1] - 1) & (df["weight"] <= food_weight[1])]
    return df


def acidity_rule(df, food_attributes):
    # Rule 2: the wine should be at least as acidic as the food
    df = df.loc[df["acid"] >= food_attributes["acid"][1]]
    return df


def sweetness_rule(df, food_attributes):
    # Rule 3: the wine should be at least as sweet as the food
    df = df.loc[df["sweet"] >= food_attributes["sweet"][1]]
    return df


def bitterness_rule(df, food_attributes):
    # Rule 4: bitter wines do not pair well with bitter foods
    if food_attributes["bitter"][1] == 4:
        df = df.loc[df["bitter"] <= 2]
    return df


def bitter_salt_rule(df, food_attributes):
    # Rule 5: bitter and salt do not go well together
    if food_attributes["bitter"][1] == 4:
        df = df.loc[(df["salt"] <= 2)]
    if food_attributes["salt"] == 4:
        df = df.loc[(df["bitter"][1] <= 2)]
    return df


def acid_bitter_rule(df, food_attributes):
    # Rule 6: acid and bitterness do not go well together
    if food_attributes["acid"][1] == 4:
        df = df.loc[(df["bitter"] <= 2)]
    if food_attributes["bitter"][1] == 4:
        df = df.loc[(df["acid"] <= 2)]
    return df


def acid_piquant_rule(df, food_attributes):
    # Rule 7: acid and piquant do not go well together
    if food_attributes["acid"][1] == 4:
        df = df.loc[(df["piquant"] <= 2)]
    if food_attributes["piquant"][1] == 4:
        df = df.loc[(df["acid"] <= 2)]
    return df


def sweet_pairing(df, food_attributes):
    # Rule 1: sweet food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_attributes["sweet"][1] == 4:
        df["pairing_type"] = np.where(
            (
                (df.bitter == 4)
                | (df.fat == 4)
                | (df.piquant == 4)
                | (df.salt == 4)
                | (df.acid == 4)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def acid_pairing(df, food_attributes):
    # Rule 2: acidic food goes well with highly sweet, fat, or salt wine
    if food_attributes["acid"][1] == 4:
        df["pairing_type"] = np.where(
            ((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)),
            "contrasting",
            df.pairing_type,
        )
    return df


def salt_pairing(df, food_attributes):
    # Rule 3: salt food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_attributes["salt"][1] == 4:
        df["pairing_type"] = np.where(
            (
                (df.bitter == 4)
                | (df.sweet == 4)
                | (df.piquant == 4)
                | (df.fat == 4)
                | (df.acid == 4)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def piquant_pairing(df, food_attributes):
    # Rule 4: piquant food goes well with highly sweet, fat, or salt wine
    if food_attributes["piquant"][1] == 4:
        df["pairing_type"] = np.where(
            ((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)),
            "contrasting",
            df.pairing_type,
        )
    return df


def fat_pairing(df, food_attributes):
    # Rule 5: fatty food goes well with highly bitter, fat, piquant, salt or acid wine
    if food_attributes["fat"][1] == 4:
        df["pairing_type"] = np.where(
            (
                (df.bitter == 4)
                | (df.sweet == 4)
                | (df.piquant == 4)
                | (df.salt == 4)
                | (df.acid == 4)
            ),
            "contrasting",
            df.pairing_type,
        )
    return df


def bitter_pairing(df, food_attributes):
    # Rule 6: bitter food goes well with highly sweet, fat, or salt wine
    if food_attributes["bitter"][1] == 4:
        df["pairing_type"] = np.where(
            ((df.sweet == 4) | (df.fat == 4) | (df.salt == 4)),
            "contrasting",
            df.pairing_type,
        )
    return df


def congruent_pairing(pairing_type, max_food_nonaroma_val, wine_nonaroma_val):
    if pairing_type == "congruent":
        return "congruent"
    elif wine_nonaroma_val >= max_food_nonaroma_val:
        return "congruent"
    else:
        return "contrasting"


def nonaroma_rules(wine_df, food_attributes):
    df = weight_rule(wine_df, food_attributes["weight"])
    list_of_rules = [
        acidity_rule,
        sweetness_rule,
        bitterness_rule,
        bitter_salt_rule,
        acid_bitter_rule,
        acid_piquant_rule,
    ]
    for rule in list_of_rules:
        # only apply the rule if it retains a sufficient number of wines in the selection.
        df_test = rule(df, food_attributes)
        if df_test.shape[0] > 3:
            df = rule(df, food_attributes)
    #         print(df.shape)
    return df


def congruent_or_contrasting(df, food_attributes):
    food_attributes = {
        taste: value for taste, value in food_attributes.items() if taste != "weight"
    }

    # first, look for a congruent match
    max_nonaroma_val = max([i[1] for i in list(food_attributes.values())])
    most_defining_tastes = [
        key for key, val in food_attributes.items() if val[1] == max_nonaroma_val
    ]
    df["pairing_type"] = ""
    for taste in most_defining_tastes:
        df["pairing_type"] = df.apply(
            lambda x: congruent_pairing(
                x["pairing_type"], food_attributes[taste][1], x[taste]
            ),
            axis=1,
        )

    # then, look for any contrasting matches
    list_of_tests = [
        sweet_pairing,
        acid_pairing,
        salt_pairing,
        piquant_pairing,
        fat_pairing,
        bitter_pairing,
    ]
    for test in list_of_tests:
        df = test(df, food_attributes)
    return df


def sort_by_aroma_similarity(df, food_aroma):
    df["aroma_distance"] = df["aroma"].apply(
        lambda x: spatial.distance.cosine(x, food_aroma["aroma"])
    )
    df["flavor_distance"] = df["flavor"].apply(
        lambda x: spatial.distance.cosine(x, food_aroma["flavor"])
    )
    df.sort_values(
        by=["aroma_distance", "flavor_distance"], ascending=True, inplace=True
    )
    return df


def retrieve_pairing_type_info(wine_recommendations, pairing_type, top_n, wine_df):
    pairings = wine_recommendations.loc[
        wine_recommendations["pairing_type"] == pairing_type
    ].head(top_n)
    wine_names = list(pairings.index)
    recommendation_nonaromas = wine_df.loc[wine_names, :]
    pairing_nonaromas = recommendation_nonaromas[
        ["sweet", "acid", "salt", "piquant", "fat", "bitter"]
    ].to_dict("records")
    pairing_body = list(recommendation_nonaromas["weight"])
    descriptors = pairings["descriptors"].tolist()
    descriptors = [eval(descriptor) for descriptor in descriptors]
    return wine_names, pairing_nonaromas, pairing_body, descriptors
