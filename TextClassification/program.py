import json
import os
from collections import Counter
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
        'hello': ['hi ', 'greetings', 'hello', 'what\'s up', 'hey ', 'how are you?', 'good morning', 'good night',
                  'good evening', 'good day', 'howdy', 'hi-ya', 'hey ya'],
        'bye': ['bye', 'cheerio', 'adios', 'sayonara', 'peace out', 'see ya', 'see you', 'c ya', 'c you', 'ciao'],
        'ack': ['okay', 'whatever', 'ok ', 'o.k. ', 'kay ', 'fine '],
        'confirm': ['is it', 'is that', 'make sure', 'confirm', 'double check', 'check again', 'does it'],
        'deny': ['dont want', 'don\'t want', 'wrong', 'dont like', 'don\'t like'],
        'inform': ['dont care', 'don\'t care', 'whatever', 'bakery', 'bar', 'cafe', 'coffeeshop', 'pub', 'restaurants',
                   'roadhouse', 'african',
                   'american', 'arabian', 'asian', 'international', 'european', 'central american', 'middle eastern',
                   'world', 'vegan', 'vegetarian', 'free', 'kosher', 'traditional', 'fusion', 'modern', 'afghan',
                   'algerian', 'angolan', 'argentine',
                   'austrian', 'australian', 'bangladeshi', 'belarusian', 'belgian', 'bolivian', 'bosnian',
                   'herzegovinian', 'brazilian', 'british', 'bulgarian', 'cambodian',
                   'cameroonian', 'canadian', 'cantonese', 'catalan', 'caribbean', 'chadian', 'chinese', 'colombian',
                   'costa rican', 'czech', 'congolese', 'cuban', 'danish', 'ecuadorian', 'salvadoran', 'emirati',
                   'english', 'eritrean',
                   'estonian',
                   'ethiopian', 'finnish', 'french', 'german', 'ghanaian', 'greek', 'guatemalan', 'dutch', 'honduran',
                   'hungarian', 'icelandic',
                   'indian', 'indonesian', 'iranian', 'iraqi', 'irish', 'israeli', 'italian', 'ivorian', 'jamaican',
                   'japanese',
                   'jordanian', 'kazakh', 'kenyan', 'korean', 'lao', 'latvian', 'lebanese', 'libyan', 'lithuanian',
                   'malagasy', 'malaysian',
                   'mali', 'mauritanian', 'mediterranean', 'mexican', 'moroccan', 'namibian', 'new zealand',
                   'nicaraguan',
                   'nigerien', 'nigerian', 'norwegian', 'omani', 'pakistani', 'panamanian', 'paraguayan', 'peruvian',
                   'persian', 'philippine', 'polynesian', 'polish', 'portuguese', 'romanian', 'russian', 'scottish',
                   'senegalese', 'serbian',
                   'singaporean', 'slovak', 'somalian', 'spanish', 'sudanese', 'swedish', 'swiss', 'syrian', 'thai',
                   'tunisian', 'turkish',
                   'ukranian', 'uruguayan', 'vietnamese', 'welsh', 'zambian', 'zimbabwean', 'west', 'north', 'south',
                   'east', 'part of town', 'moderate', 'expensive', 'cheap', 'any ', 'priced', 'barbecue', 'burger',
                   'chicken',
                   'doughnut', 'fast food',
                   'fish and chips', 'hamburger', 'hot dog', 'ice cream', 'noodles', 'pasta', 'pancake', 'pizza',
                   'ramen', 'restaurant', 'seafood', 'steak',
                   'sandwich', 'sushi'],
        'negate': ['no ', 'false', 'nope'],
        'repeat': ['repeat', 'say again', 'what was that'],
        'reqalts': ['how about', 'what about', 'anything else'],
        'reqmore': ['more', 'additional information'],
        'request': ['what', 'whats' 'what\'s', 'why', 'where', 'when', 'how much', 'may', 'address', 'post code',
                    'location', 'phone number'],
        'restart': ['reset', 'start over', 'restart'],
        'thankyou': ['thank you', 'cheers', 'thanks'],
        'affirm': ['ye ', 'yes', 'right ']
    }

    classification = []
    sentence_to_classify = utterance.lower()
    for category, keywords in categories.items():
        keywords_found = [keyword for keyword in keywords if keyword in sentence_to_classify]
        if len(keywords_found) > 0: classification.append(category)

    return classification if len(classification) > 0 else ['null']


def random_keyword_classifier(utterance):
    """
    Function used to classify a sentence based on a pseudo random value
    :return: The chosen category of utterance.
    """
    dialog_act_counter = Counter()

    with open(os.path.abspath("train_dialogact.txt"), 'r') as file:
        dialog_act_counter.update(file.read().split())
        file.seek(0)
        tot_dialog_act = len(file.readlines())

    categories = [cat for cat in dialog_act_counter.keys()]
    probabilities = [prob / tot_dialog_act for prob in dialog_act_counter.values()]

    my_choice = choice(categories, p=probabilities)
    return [my_choice]


def get_user_input():
    """
    Function used to prompt user for input to classify
    """
    while True:
        utterance = input('Please make your request and then hit \'enter\'. Type \'exit\' to end the program.\n')
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


def test_classifier(utterances, dialogue_index, sentence_index, classifier):
    keyword_classification = classifier(utterances[dialogue_index][1][sentence_index])
    ground_truths = [ground_truth.split('(')[0] for ground_truth in
                     utterances[dialogue_index][0][sentence_index].split('|')]

    ground_truths.sort()
    keyword_classification.sort()
    if ground_truths != keyword_classification:
        print(ground_truths, keyword_classification, utterances[dialogue_index][1][sentence_index])
        return False, ground_truths, keyword_classification
    else:
        return True, ground_truths, keyword_classification


def evaluate_classifier(utterances, test_data, classifier):
    correctly_classified = 0
    incorrectly_classified = 0
    true_positives = {'hello': 0, 'bye': 0, 'ack': 0, 'confirm': 0, 'deny': 0, 'inform': 0,
                      'negate': 0, 'repeat': 0, 'reqalts': 0, 'reqmore': 0, 'request': 0, 'restart': 0,
                      'thankyou': 0, 'affirm': 0, 'null': 0}
    false_negatives = {'hello': 0, 'bye': 0, 'ack': 0, 'confirm': 0, 'deny': 0, 'inform': 0,
                       'negate': 0, 'repeat': 0, 'reqalts': 0, 'reqmore': 0, 'request': 0, 'restart': 0,
                       'thankyou': 0, 'affirm': 0, 'null': 0}
    false_positives = {'hello': 0, 'bye': 0, 'ack': 0, 'confirm': 0, 'deny': 0, 'inform': 0,
                       'negate': 0, 'repeat': 0, 'reqalts': 0, 'reqmore': 0, 'request': 0, 'restart': 0,
                       'thankyou': 0, 'affirm': 0, 'null': 0}
    for dialogue_index in range(test_data):
        for sentence_index in range(len(utterances[dialogue_index][0])):
            test_result, ground_truths, keyword_classification = test_classifier(utterances, dialogue_index,
                                                                                 sentence_index, classifier)
            if test_result == True:
                correctly_classified += 1
                for i in range(len(ground_truths)):
                    true_positives[ground_truths[i]] += 1
            else:
                incorrectly_classified += 1
                for i in range(len(ground_truths)):
                    if ground_truths[i] not in keyword_classification:
                        false_negatives[ground_truths[i]] += 1
                    else:
                        true_positives[ground_truths[i]] += 1
                for i in range(len(keyword_classification)):
                    if keyword_classification[i] not in ground_truths:
                        false_positives[keyword_classification[i]] += 1

    return correctly_classified, incorrectly_classified, true_positives, false_negatives, false_positives


def get_recall(true_positives, false_negatives):
    recall = {}
    for category in true_positives:
        if true_positives[category] == 0 and false_negatives[category] == 0:
            recall[category] = 0
        else:
            recall[category] = true_positives[category] / (true_positives[category] + false_negatives[category])

    return recall


def get_precision(true_positives, false_positives):
    precision = {}
    for category in true_positives:
        if true_positives[category] == 0 and false_positives[category] == 0:
            precision[category] = 0
        else:
            precision[category] = true_positives[category] / (true_positives[category] + false_positives[category])

    return precision


def get_average(values_per_category):
    sum = 0
    num_categories = 0
    for category in values_per_category:
        num_categories += 1
        sum += values_per_category[category]
    return sum / num_categories


if __name__ == '__main__':
    utterances = []
    for session_folder in os.listdir('../test/data/'):
        for voice_sample_folder in os.listdir('../test/data/' + session_folder):
            utterances.append(read_utterances_from_files(session_folder, voice_sample_folder))

    if not os.path.isfile('utterance_dialog_act.txt'): [
        write_dialogue_to_file(utterances, dialogue_index, 'utterance_dialog_act.txt') for dialogue_index in
        range(len(utterances))]

    training_data = int(len(utterances) * .8)
    test_data = int(len(utterances) * .2)
    correctly_classified, incorrectly_classified, true_positives, false_negatives, false_positives = evaluate_classifier(
        utterances, test_data, random_keyword_classifier)
    recall = get_recall(true_positives, false_negatives)
    precision = get_precision(true_positives, false_positives)
    average_recall = get_average(recall)
    average_precision = get_average(precision)
    print('Accuracy: ', correctly_classified / (correctly_classified + incorrectly_classified))
    print('Recall per category: ', recall)
    print('Average Recall: ', average_recall)
    print('Precision per category: ', precision)
    print('Average Precision: ', average_precision)

    get_user_input()