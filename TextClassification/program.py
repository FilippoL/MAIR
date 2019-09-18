import json
import os
from numpy.random import choice


def write_dialogue_to_file(utterances, dialogue_index, filename):
    """
    Function used to write a dialogue to a specified file line by line
    :param utterances: dialogues extracted from the json files
    :param dialogue_index: index of the dialogue considered
    :param filename: name of the file to which the dialogue will be written
    """
    with open(filename, 'a') as file:
        for sentence_index in range(len(utterances[dialogue_index][0])):
            file.write('{0}     {1}\n'.format(utterances[dialogue_index][0][sentence_index],
                                              utterances[dialogue_index][1][sentence_index]))


def read_utterances_from_files(session_folder, voice_sample_folder):
    """
    Function used to extract dialogues from json files
    :param session_folder: Name of the folder in which the voice_sample_folder can be found
    :param voice_sample_folder: Name of the folder from which the dialogue is to be extracted
    :return: array containing the classification, the content of the dialogue and the session id
    """
    utterance_content = []
    dialog_act = []

    with open('../test/data/' + session_folder + '/' + voice_sample_folder + '/label.json') as label_data:
        label = json.load(label_data)

        for j in range(len(label['turns'])):
            utterance_content += [label['turns'][j]['transcription']]
            dialog_act += [label['turns'][j]['semantics']['cam']]

        session_id = label['session-id']
    return [dialog_act, utterance_content, session_id]


def keyword_classifier(utterance):
    """
    Function used to classify a sentence based on a keyword classifier
    :param utterance: sentence to be classified
    :return: array containing all classified categories
    """
    categories = {
        'hello': ['hi', 'greetings', 'hello', 'what\'s up', 'hey', 'how are you?', 'good morning', 'good night',
                  'good evening', 'good day', 'howdy', 'hi-ya', 'hey ya'],
        'bye': ['bye', 'cheerio', 'adios', 'sayonara', 'peace out', 'see ya', 'see you', 'c ya', 'c you', 'ciao'],
        'ack': ['okay', 'um', 'whatever', 'ok', 'o.k.', 'kay ', 'fine ', 'good '],
        'confirm': ['is it', 'is that', 'make sure', 'confirm', 'double check', 'check again', 'does it'],
        'deny': ['dont want', 'don\'t want', 'wrong', 'dont like', 'don\'t like'],
        'inform': ['dont care', 'don\'t care', 'whatever', 'restaurants', 'Bakery cafÈs', 'Barbecue restaurants',
                   'Coffeehouse chains', 'Doughnut shops', 'Fast-food', 'chicken restaurants',
                   'Fish and chip restaurants', 'Frozen yogurt companies', 'Hamburger restaurants',
                   'Hot dog restaurants', 'Ice cream parlor', 'Noodle restaurants', 'Ramen shops', 'Oyster bars',
                   'Pancake houses', 'Pizza chains', 'Pizza franchises', 'Seafood restaurants', 'Steakhouses',
                   'Submarine sandwichrestaurants', 'Sushi restaurants', 'Vegetarian restaurants', 'spanish', 'mexican',
                   'thai', 'indonesian', 'japanese', 'west', 'north', 'south', 'east', 'polynesian', 'italian',
                   'portuguese', 'moderate', 'expensive', 'cheap', 'vietnamese', 'any', 'priced'],
        'negate': ['no', 'false', 'nope'],
        'repeat': ['repeat', 'say again', 'what was that'],
        'reqalts': ['how about', 'what about', 'anything else'],
        'reqmore': ['more', 'additional information'],
        'request': ['what', 'whats' 'what\'s', 'why', 'where', 'when', 'how much', 'may', 'address', 'phone number',
                    'area'],
        'restart': ['reset', 'start over', 'restart'],
        'thankyou': ['thank you', 'cheers', 'thanks'],
        'affirm': ['ye', 'yes', 'right'],
    }

    classification = []
    sentence_to_classify = utterance.lower()
    for category, keywords in categories.items():
        keywords_found = [keyword for keyword in keywords if keyword in sentence_to_classify]
        if len(keywords_found) > 0: classification.append(category)

    return classification if len(classification) > 0 else None


def random_keyword_classifier():
    """
    Function used to classify a sentence based on a pseudo random value
    :param utterance: sentence to be classified
    :return: array containing all classified categories
    """
    categories = ['hello', 'bye', 'ack', 'confirm', 'deny', 'inform', 'negate', 'repeat', 'reqalts', 'reqmore',
                  'request', 'restart', 'thankyou', 'affirm', 'null']

    my_choice = choice(categories, replace=False,
                       p=[0.00324, 0.11755, 0.00107, 0.006, 0.00094, 0.35505, 0.01518,
                          0.00115, 0.06124, 0.00017, 0.22733, 0.00048, 0.11396, 0.04036,
                          0.05628])
    return my_choice


def get_user_input():
    """
    Function used to prompt user for input to classify
    """
    while True:
        utterance = input('Please make your request and then hit \'enter\'. Type \'exit\' to end the programm.\n')
        if utterance == 'exit':
            break

        # result = keyword_classifier(utterance)
        result = random_keyword_classifier()
        if result is not None:
            print('The following category(ies) have been identified using the keyword classifier:')
            # print(*result, sep="\n")
            print(result)
        else:
            print(
                'Unfortunately, no categories were detected for the sentence you entered using the keyword classifier.')


if __name__ == '__main__':
    utterances = []
    for session_folder in os.listdir('../test/data/'):
        for voice_sample_folder in os.listdir('../test/data/' + session_folder):
            utterances.append(read_utterances_from_files(session_folder, voice_sample_folder))

    if not os.path.isfile('utterance_dialog_act.txt'): [
        write_dialogue_to_file(utterances, dialogue_index, 'utterance_dialog_act.txt') for dialogue_index in
        range(len(utterances))]
    get_user_input()

#   10169 = inform = 35.505%
#   29 = ack = 0.101%
#   1156 = affirm = 4.036%
#   3367 = bye = 11.755%
#   172 = confirm = 0.6%
#   27 = deny = 0.094%
#   93 = hello = 0.324%
#   435 = negate = 1.518%
#   33 = repeat = 0.115%
#   1754 = reqalts = 6.124%
#   5 = reqmore = 0.017%
#   6511 = request = 22.733%
#   14 = restart = 0.048%
#   3264 = thankyou = 11.396%
#   1612 = null = 5.628%
