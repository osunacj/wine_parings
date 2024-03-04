# a file containing the 50 most frequently appearing descriptors for each wine
descriptor_frequencies = pd.read_csv("wine_variety_descriptors.csv", index_col="index")

# our word2vec model for all wine and food terms
wine_word2vec_model = Word2Vec.load("food_word2vec_model.bin")
word_vectors = wine_word2vec_model.wv

# a file with the average wine nonaroma vectors for each nonaroma
food_nonaroma_infos = pd.read_csv(
    "average_nonaroma_vectors.csv", index_col="Unnamed: 0"
)


# this function scales each nonaroma between 0 and 1
def minmax_scaler(val, minval, maxval):
    val = max(min(val, maxval), minval)
    normalized_val = (val - minval) / (maxval - minval)
    return normalized_val


# this function makes sure that a scaled value (between 0 and 1) is returned for a food nonaroma
def check_in_range(label_range_dict, value):
    for label, value_range_tuple in label_range_dict.items():
        lower_end = value_range_tuple[0]
        upper_end = value_range_tuple[1]
        if value >= lower_end and value <= upper_end:
            return label
        else:
            continue


# this function calculates the average embedding of all foods supplied as input
def calculate_avg_food_vec(sample_foods):
    sample_food_vecs = []
    for s in sample_foods:
        sample_food_vec = word_vectors[s]
        sample_food_vecs.append(sample_food_vec)
    sample_food_vecs_avg = np.average(sample_food_vecs, axis=0)
    return sample_food_vecs_avg


# this function returns two things: a score (between 0 and 1) and a normalized value (integer between 1 and 4) for a given nonaroma
def nonaroma_values(nonaroma, average_food_embedding):
    average_taste_vec = food_nonaroma_infos.at[nonaroma, "average_vec"]
    average_taste_vec = re.sub("\s+", ",", average_taste_vec)
    average_taste_vec = average_taste_vec.replace("[,", "[")
    average_taste_vec = np.array(ast.literal_eval(average_taste_vec))

    similarity = 1 - spatial.distance.cosine(average_taste_vec, average_food_embedding)
    # scale the similarity using our minmax scaler
    scaled_similarity = minmax_scaler(
        similarity,
        food_nonaroma_infos.at[nonaroma, "farthest"],
        food_nonaroma_infos.at[nonaroma, "closest"],
    )
    standardized_similarity = check_in_range(food_weights[nonaroma], scaled_similarity)
    similarity_and_scalar = (scaled_similarity, standardized_similarity)
    return similarity_and_scalar


# this function loops through the various nonaromas, returning the nonaroma scores & normalized values, the body/weight of the food and the average food embedding
def return_all_food_values(sample_foods):
    food_nonaromas = dict()
    average_food_embedding = calculate_avg_food_vec(sample_foods)
    for nonaroma in ["sweet", "acid", "salt", "piquant", "fat", "bitter"]:
        food_nonaromas[nonaroma] = nonaroma_values(nonaroma, average_food_embedding)
    food_weight = nonaroma_values("weight", average_food_embedding)
    return food_nonaromas, food_weight, average_food_embedding


# these functions return the wine descriptors that most closely match the wine aromas of the selected recommendations. This will help give additional context and justification to the recommendations.


def find_descriptor_distance(word, foodvec):
    descriptor_wordvec = word_vectors[word]
    similarity = 1 - spatial.distance.cosine(descriptor_wordvec, foodvec)
    return similarity


def most_impactful_descriptors(recommendation):
    recommendation_frequencies = descriptor_frequencies.filter(
        like=recommendation, axis=0
    )
    recommendation_frequencies["similarity"] = recommendation_frequencies[
        "descriptors"
    ].apply(lambda x: find_descriptor_distance(x, aroma_embedding))
    recommendation_frequencies.sort_values(
        ["similarity", "relative_frequency"], ascending=False, inplace=True
    )
    recommendation_frequencies = recommendation_frequencies.head(5)
    most_impactful_descriptors = list(recommendation_frequencies["descriptors"])
    return most_impactful_descriptors


def retrieve_pairing_type_info(wine_recommendations, pairing_type):
    pairings = wine_recommendations.loc[
        wine_recommendations["pairing_type"] == pairing_type
    ].head(4)
    wine_names = list(pairings.index)
    recommendation_nonaromas = wine_recommendations.loc[wine_names, :]
    pairing_nonaromas = recommendation_nonaromas[
        ["sweet", "acid", "salt", "piquant", "fat", "bitter"]
    ].to_dict("records")
    pairing_body = list(recommendation_nonaromas["weight"])
    impactful_descriptors = list(pairings["most_impactful_descriptors"])
    return wine_names, pairing_nonaromas, pairing_body, impactful_descriptors
