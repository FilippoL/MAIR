import json

import pandas as pd
from Levenshtein import distance
from nltk import word_tokenize

from TextClassification import program

current_state = next_state = None


def read_sections_from_json(s=[], p=""):
    sections_content = []

    with open('ontology_dstc2.json') as file:
        ontology = json.load(file)
        if s:
            for section in s:
                sections_content += ontology[p][section]
        else:
            sections_content += ontology[p]
    return sections_content


def check_presence(full_sentence, looking_for):
    for word in full_sentence:
        if word in looking_for:
            return word


def check_spelling(utterance, list_right_words):
    for i in range(len(utterance)):
        for right_word in list_right_words:
            if distance(right_word, utterance[i]) == 0:
                break
            elif distance(right_word, utterance[i]) < 2:
                utterance[i] = right_word
                break


# These are capital cause they will be class instances
def StartState():
    IdleState()
    pass


def IdleState():
    sections = ["food", "pricerange", "area"]

    dialog_acts, stripped_utterances = program.read_target_and_labels_from_file(
        "utterance_dialog_act_only_shuffled.txt")
    classifier, x, y, x_test, y_test, labels, vectorizer = program.get_fitted_model(dialog_acts, stripped_utterances)

    right_words = read_sections_from_json(sections, "informable")
    food_types = read_sections_from_json(["food"], "informable")
    price_range = read_sections_from_json(["pricerange"], "informable")
    area = read_sections_from_json(["area"], "informable")

    while True:
        user_utterance = input("\n> ")
        check_spelling(user_utterance.split(), right_words)

        predictions = classifier.predict(vectorizer.transform([user_utterance]))
        tokenized_utterance = word_tokenize(user_utterance)
        print(tokenized_utterance)

        restaurant_info = pd.read_csv("restaurantinfo.csv")
        # with pd.option_context('display.max_rows', None, 'display.max_columns',
        #                        None):
        #     print(restaurant_info)

        try:
            label_pred = labels[predictions[0].tolist().index(1)]
        except ValueError:
            label_pred = "null"
        print(label_pred)
        if label_pred == "hello":
            print("Hi, how can I help you?")
        elif label_pred == "inform":
            user_food_type = check_presence(tokenized_utterance, food_types)
            user_area = check_presence(tokenized_utterance, area)
            user_price_range = check_presence(tokenized_utterance, price_range)

            if not user_food_type:
                tokenized_utterance = word_tokenize(input("Please state what kind of restaurant you want.\n>"))
                check_spelling(tokenized_utterance, right_words)
                user_food_type = check_presence(tokenized_utterance, food_types)

            elif user_food_type and not user_area:
                tokenized_utterance = word_tokenize(input("Please state the area you want the restaurant to be in.\n>"))
                check_spelling(tokenized_utterance, right_words)
                user_area = check_presence(tokenized_utterance, area)

            restaurant_count = len(restaurant_info.loc[
                                       (user_food_type == restaurant_info["food"]) & (
                                               user_area == restaurant_info["area"])])

            available_restaurants = restaurant_info.loc[
                (user_food_type == restaurant_info["food"]) & (user_area == restaurant_info["area"])]

            if restaurant_count == 0:
                print("There are no restaurants available.")

            print("There are {0} restaurants available.".format(restaurant_count))

            print(
                "So you want a(n) {0} restaurant in {1} part of town, with {2} price.".format(user_food_type, user_area,
                                                                                              user_price_range))


def EndState():
    pass


def QuestioningState():
    pass


if __name__ == '__main__':
    StartState()
