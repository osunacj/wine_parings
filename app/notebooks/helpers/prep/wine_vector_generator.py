import pandas as pd
import numpy as np

core_tastes = [
        "aroma",
        "weight",
        "sweet",
        "acid",
        "salt",
        "piquant",
        "fat",
        "bitter",
    ]


def extract_descriptors_from_review():
    





def return_descriptor_from_mapping(descriptor_mapping, word, core_taste):
    # if the common word is in the index (normalized word) of the dataframe of core taste c
    if word in list(descriptor_mapping.index):
        # return the descriptor in "combined that matches the taset and the common word"
        descriptor_to_return = descriptor_mapping["combined"][word]
        return descriptor_to_return
    else:
        # return None if the word is not in the dataframe
        return None


def main():

    descriptor_mappings = dict()
    for c in core_tastes:
        if c == "aroma":
            descriptor_mapping_filtered = descriptor_mapping.loc[
                descriptor_mapping["type"] == "aroma"
            ]
        else:
            descriptor_mapping_filtered = descriptor_mapping.loc[
                descriptor_mapping["primary taste"] == c
            ]
        # dict containing dataframes of descriptors of certain taste
        descriptor_mappings[c] = descriptor_mapping_filtered

    review_descriptors = []
    # iterate through every review
    for review in wine_reviews:
        taste_descriptors = []
        # returns a list of lists being the sentences[words[]]
        normalized_review = normalize_text(review)
        # a list with the most common words matched by _
        phrased_review = wine_trigram_model[normalized_review]
        #     print(phrased_review)

        for c in core_tastes:
            descriptors_only = [
                return_descriptor_from_mapping(descriptor_mappings[c], word, c)
                # word is every word in the common words in phrases
                for word in phrased_review
            ]
            # remove NONE, only keep the descriptors for the taste
            no_nones = [str(d).strip() for d in descriptors_only if d is not None]
            # build a string of the descriptors
            descriptorized_review = " ".join(no_nones)
            # list of strings that represent the descriptors for a taste
            taste_descriptors.append(descriptorized_review)
        # list of lists where each list is a review containing the list of descriptors by taste  [[desciptor aroma, descriptor weight, ...], [review 2] ...]
        review_descriptors.append(taste_descriptors)


def vectors():
    taste_descriptors = []
    taste_vectors = []

    for n, taste in enumerate(core_tastes):
        print(taste)
        # lsit of strings of all descriptors for taste n
        taste_words = [r[n] for r in review_descriptors]

        vectorizer = TfidfVectorizer()
        X = vectorizer.fit(taste_words)
        dict_of_tfidf_weightings = dict(zip(X.get_feature_names(), X.idf_))

        wine_review_descriptors = []
        wine_review_vectors = []
        # d is the string of descriptor words
        for d in taste_words:
            descriptor_count = 0
            weighted_review_terms = []
            terms = d.split(" ")
            # descriptor of term in the string of descriptors d
            for term in terms:
                if term in dict_of_tfidf_weightings.keys():
                    tfidf_weighting = dict_of_tfidf_weightings[term]
                    try:
                        word_vector = wine_word2vec_model.wv.get_vector(term).reshape(
                            1, 300
                        )
                        # Vector of the word times the tfidf vector for the word
                        weighted_word_vector = tfidf_weighting * word_vector
                        weighted_review_terms.append(weighted_word_vector)
                        descriptor_count += 1
                    except:
                        continue
                else:
                    continue
            try:
                # average of the vectors of each descriptor(term) of a given taste [oak, plum, ]
                review_vector = sum(weighted_review_terms) / len(weighted_review_terms)
                review_vector = review_vector[0]
            except:
                review_vector = np.nan
            #         terms_and_vec = [terms, review_vector]
            wine_review_vectors.append(review_vector)
            wine_review_descriptors.append(terms)

        taste_vectors.append(wine_review_vectors)
        # [aroma[[vector of first review of aromas], [vector of second review of aromas], ...], biter[], ...]
        taste_descriptors.append(wine_review_descriptors)

    taste_vectors_t = list(map(list, zip(*taste_vectors)))
    taste_descriptors_t = list(map(list, zip(*taste_descriptors)))

    review_vecs_df = pd.DataFrame(taste_vectors_t, columns=core_tastes)

    columns_taste_descriptors = [a + "_descriptors" for a in core_tastes]
    review_descriptors_df = pd.DataFrame(
        taste_descriptors_t, columns=columns_taste_descriptors
    )

    wine_df_vecs = pd.concat(
        [wine_df_merged_filtered, review_descriptors_df, review_vecs_df], axis=1
    )
    wine_df_vecs.head(5)


def average():
    # pull the average embedding for the wine attribute across all wines. 
    avg_taste_vecs = dict()
    for t in core_tastes:
        # look at the average embedding for a taste, across all wines that have descriptors for that taste 
        review_arrays = wine_df_vecs[t].dropna()
        average_taste_vec = np.average(review_arrays)
        avg_taste_vecs[t] = average_taste_vec

# lsit of tuple of (variety, geo_normalized)
normalized_geos = list(set(zip(wine_df_vecs['Variety'], wine_df_vecs['geo_normalized'])))

def subset_wine_vectors(list_of_varieties, wine_attribute):
    wine_variety_vectors = []
    for v in list_of_varieties:

        one_var_only = wine_df_vecs.loc[(wine_df_vecs['Variety'] == v[0]) & 
                                                (wine_df_vecs['geo_normalized'] == v[1])]
        if len(list(one_var_only.index)) < 1 or str(v[1][-1]) == '0':
            continue
        else:
            taste_vecs = list(one_var_only[wine_attribute])
            # if vector exits palce existent vector otherwise place average of taste (wine attribute)
            taste_vecs = [avg_taste_vecs[wine_attribute] if 'numpy' not in str(type(x)) else x for x in taste_vecs]
            # get the average for the variety and geo normalization
            average_variety_vec = np.average(taste_vecs, axis=0)
            
            descriptor_colname = wine_attribute + '_descriptors'
            # write the descriptors for each variety, geo, taste -> all descritors of aroma
            all_descriptors = [i[0] for i in list(one_var_only[descriptor_colname])]
            word_freqs = Counter(all_descriptors)
            # Get most common word frequencies
            most_common_words = word_freqs.most_common(50)
            top_n_words = [(i[0], "{:.2f}".format(i[1]/len(taste_vecs))) for i in most_common_words]
            top_n_words = [i for i in top_n_words if len(i[0])>2]
            # str, vector with average, list of 50 most common words
            wine_variety_vector = [v, average_variety_vec, top_n_words]
                
            wine_variety_vectors.append(wine_variety_vector)
            
    return wine_variety_vectors


def pca_wine_variety(list_of_varieties, wine_attribute, pca=True):
    wine_var_vectors = subset_wine_vectors(normalized_geos, wine_attribute)
    
    wine_varieties = [str(w[0]).replace('(', '').replace(')', '').replace("'", '').replace('"', '') for w in wine_var_vectors]
    wine_var_vec = [w[1] for w in wine_var_vectors]
    if pca:
        pca = PCA(1)
        wine_var_vec = pca.fit_transform(wine_var_vec)
        wine_var_vec = pd.DataFrame(wine_var_vec, index=wine_varieties)
    else:
        wine_var_vec = pd.Series(wine_var_vec, index=wine_varieties)
    wine_var_vec.sort_index(inplace=True)
    
    wine_descriptors = pd.DataFrame([w[2] for w in wine_var_vectors], index=wine_varieties)
    wine_descriptors = pd.melt(wine_descriptors.reset_index(), id_vars='index')
    wine_descriptors.sort_index(inplace=True)
    
    return wine_var_vec, wine_descriptors

taste_dataframes = []
# generate the dataframe of aromas vectors as output, 
aroma_vec, aroma_descriptors = pca_wine_variety(normalized_geos, 'aroma', pca=False)
taste_dataframes.append(aroma_vec)

# generate the dataframes of nonaroma scalars
for tw in core_tastes[1:]:
    pca_w_dataframe, nonaroma_descriptors = pca_wine_variety(normalized_geos, tw, pca=True)
    taste_dataframes.append(pca_w_dataframe)
    
# combine all the dataframes created above into one 
all_nonaromas = pd.concat(taste_dataframes, axis=1)
all_nonaromas.columns = core_tastes


if __name__ == "__main__":
    main()
