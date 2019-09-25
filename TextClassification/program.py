import json
import os
from collections import Counter
from numpy.random import choice
from numpy import array
import pickle
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import scikitplot as skplt
from sklearn.feature_extraction.text import CountVectorizer
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score, log_loss, jaccard_score
import warnings

warnings.filterwarnings('ignore')


def write_dialogue_to_file(utterances, dialogue_index, filename):
    """
    Function used to write a dialogue to a specified file line by line.
    :param utterances: dialogues extracted from the json files.
    :param dialogue_index: index of the dialogue considered.
    :param filename: name of the file to which the dialogue will be written.
    """
    with open(filename, 'a') as file:
        for sentence_index in range(len(utterances[dialogue_index][0])):
            file.write('{0}     {1}\n'.format(utterances[dialogue_index][0][sentence_index],
                                              utterances[dialogue_index][1][sentence_index]))


def write_to_file(content, filename):
    """
    Function checks if file already exists, if it doesn't will create a text file and
    fill it with the data provided in the format data[0]  data[1].
    :param content: data to be written.
    :param filename: file into which data should be allocated.
    """
    if not os.path.isfile(filename):  # Checking if file already exists, don't append data if it does.
        for j in range(len(content)):  # For each dialog in dialogues array.
            with open(filename, 'a') as file:  # Open a text file in append mode and write data into it.
                for k in range(len(content[j][0])):
                    file.write('{0}     {1}\n'.format(str(content[j][0][k]).lower().split("(")[0],
                                                      str(content[j][1][k])).lower())


def get_fitted_model(dialog_acts, utterances):
    """
    This function will take labels and targets, vectorise and encode the data
    and fit the a Random Forest Classifier with the processed label and target.
    :param dialog_acts: targets to be encoded.
    :param utterances: labels to be vectorised.
    :return: seven elements to be unpacked, in order: classifier, x, y, x_test, y_test, labels name, vectorised used.
    """
    vectorizer = CountVectorizer()
    x = vectorizer.fit_transform(utterances).toarray()

    array(dialog_acts)

    label_encoder = LabelEncoder()
    one_hot_encoder = OneHotEncoder(sparse=False)

    integer_encoded = label_encoder.fit_transform(dialog_acts)
    integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
    y = one_hot_encoder.fit_transform(integer_encoded)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=0)

    if not os.path.isfile("model_binary_RFC.sav"):
        with open("model_binary_RFC.sav", "wb") as model:
            classifier = RandomForestClassifier(n_estimators=500)
            classifier.fit(x_train, y_train)
            try:
                pickle.dump(classifier, model)

            except MemoryError:
                print("I failed dumping.\n\n")
        model.close()

    else:
        with open('model_binary_RFC.sav', 'rb') as training_model:
            classifier = pickle.load(training_model)
            training_model.close()
    return classifier, x, y, x_test, y_test, label_encoder.classes_, vectorizer


def read_utterances_from_files(session_folder, voice_sample_folder):
    """
    Function used to extract dialogues from json files.
    :param session_folder: Name of the folder in which the voice_sample_folder can be found.
    :param voice_sample_folder: Name of the folder from which the dialogue is to be extracted.
    :return: array containing the classification, the content of the dialogue and the session id.
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


def read_target_and_labels_from_file(file_path):
    """
    Function used to read target and labels from their source file.
    In here stemming for each label is applied as well.
    :param file_path: file to read data from.
    :return: two lists to be unpacked, first is target, second labels.
    """
    stemmer = SnowballStemmer("english")
    dialog_acts = []
    utterances = []
    with open(file_path, "r") as file:
        for line in file.readlines():
            dialog_acts.append(line.split()[0])
            utterances.append(stemmer.stem(line[line.find(" "):].strip()))
    return dialog_acts, utterances


def keyword_classifier(utterance):
    """
    Function used to classify a sentence based on a keyword classifier.
    :param utterance: sentence to be classified.
    :return: array containing all classified categories.
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


def test_classifier(utterances, dialogue_index, sentence_index, classifier):
    """
    Tests a given classifier by trying each and single test label against
    each test target and evaluating whether the classifier was correct or not.
    :param utterances: labels
    :param dialogue_index:
    :param sentence_index:
    :param classifier: the classifier to be tested.
    :return: boolean indicating the presence of true positive, array of ground truths, classifier classified keywords.
    """
    keyword_classification = classifier(utterances[dialogue_index][1][sentence_index])
    ground_truths = [ground_truth.split('(')[0] for ground_truth in
                     utterances[dialogue_index][0][sentence_index].split('|')]

    ground_truths.sort()
    keyword_classification.sort()
    if ground_truths != keyword_classification:
        return False, ground_truths, keyword_classification
    else:
        return True, ground_truths, keyword_classification


def evaluate_classifier(utterances, test_data, classifier):
    """

    :param utterances:
    :param test_data:
    :param classifier:
    :return:
    """
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
    """
    Function to compute recall of a given classification.
    :param true_positives: true positives values.
    :param false_negatives: false negative values.
    :return:
    """
    recall = {}
    for category in true_positives:
        if true_positives[category] == 0 and false_negatives[category] == 0:
            recall[category] = 0
        else:
            recall[category] = true_positives[category] / (true_positives[category] + false_negatives[category])

    return recall


def get_precision(true_positives, false_positives):
    """
    Function to compute precision of a given classification.
    :param true_positives: true positives values.
    :param false_positives: false positive values.
    :return:
    """
    precision = {}
    for category in true_positives:
        if true_positives[category] == 0 and false_positives[category] == 0:
            precision[category] = 0
        else:
            precision[category] = true_positives[category] / (true_positives[category] + false_positives[category])

    return precision


def get_average(values_per_category):
    """
    Function to compute average of a given set of values.
    :param values_per_category: value of each classified category.
    :return: average of classified category.
    """
    sum = 0
    num_categories = 0
    for category in values_per_category:
        num_categories += 1
        sum += values_per_category[category]
    return sum / num_categories


def get_stats(full_utterances, x_test, y_test, x, y):
    """
    Function computing and printing metrics for each classifier shown in the program,
    metrics represented is: accuracy, precision, recall and for the RFC also F-Score,
    Loss error and Jaccard index.
    :param full_utterances: list of all the labels.
    :param x_test: x_test/x_true set.
    :param y_test: y_test/y_true set.
    :param x: original X values.
    :param y: original Y values.
    """
    test_data = int(len(full_utterances) * .2)

    print("\nThis might take minutes to be executed.\n")
    print("\n###################++RANDOM KEYWORD CLASSIFIER++#########################")
    correctly_classified, incorrectly_classified, true_positives, false_negatives, false_positives = evaluate_classifier(
        full_utterances, test_data, random_keyword_classifier)
    recall = get_recall(true_positives, false_negatives)
    precision = get_precision(true_positives, false_positives)
    average_recall = get_average(recall)
    average_precision = get_average(precision)
    print('Accuracy: ', correctly_classified / (correctly_classified + incorrectly_classified))
    print('Recall per category: ', recall)
    print('Average Recall: ', average_recall)
    print('Precision per category: ', precision)
    print('Average Precision: ', average_precision)

    print("\n\n###################++KEYWORD CLASSIFIER++#########################")
    correctly_classified, incorrectly_classified, true_positives, false_negatives, false_positives = evaluate_classifier(
        full_utterances, test_data, keyword_classifier)
    recall = get_recall(true_positives, false_negatives)
    precision = get_precision(true_positives, false_positives)
    average_recall = get_average(recall)
    average_precision = get_average(precision)
    print('Accuracy: ', correctly_classified / (correctly_classified + incorrectly_classified))
    print('Recall per category: ', recall)
    print('Average Recall: ', average_recall)
    print('Precision per category: ', precision)
    print('Average Precision: ', average_precision)

    print("\n\n###################++RANDOM FOREST CLASSIFIER++#########################")
    y_pred = classifier.predict(x_test)
    print("Accuracy of train set: {0}".format(classifier.score(x, y)))
    print("Accuracy of test set: {0}".format(accuracy_score(y_test, y_pred)))
    print("Logarithmic loss: {0}".format(log_loss(y_test, y_pred)))
    print("Macro avg Jaccard index: {0}\n".format(jaccard_score(y_test, y_pred, average="macro")))
    print(classification_report(y_test, y_pred, target_names=labels))


def use_random_forest_classifier(classifier, vectorizer, labels):
    """
    Function that uses the RFC to classify a given new utterance.
    """
    while True:
        utterance = input("\n\nEnter utterance you want to classify, \ntype menu or exit to go back:\n-> ").lower()
        if utterance == "menu" or utterance == "exit":
            break
        else:
            y_pred = classifier.predict(vectorizer.transform([utterance]))
            try:
                label_pred = labels[y_pred[0].tolist().index(1)]
                print("Prediction: {0}".format(label_pred))
            except ValueError:
                print("Prediction: {0}".format("null"))


def use_random_keyword_classifier():
    """
    Function that uses the random classifier to classify a given new utterance.
    """
    while True:
        utterance = input("\n\nEnter utterance you want to classify, \ntype menu or exit to go back:\n-> ").lower()
        if utterance == "menu" or utterance == "exit":
            break
        else:
            try:
                label_pred = random_keyword_classifier(utterance)
                print("Prediction: {0}".format(*label_pred))
            except ValueError:
                print("Prediction: {0}".format("null"))


def use_keyword_classifier():
    """
    Function that uses the keyword classifier to classify a given new utterance.
    """
    while True:
        utterance = input("\n\nEnter utterance you want to classify, \ntype menu or exit to go back:\n-> ").lower()
        if utterance == "menu" or utterance == "exit":
            break
        else:
            try:
                label_pred = keyword_classifier(utterance)
                print("Prediction: {0}".format(*label_pred))
            except ValueError:
                print("Prediction: {0}".format("null"))


if __name__ == '__main__':
    full_utterances = []
    stripped_utterances = []
    for session_folder in os.listdir('../test/data/'):
        for voice_sample_folder in os.listdir('../test/data/' + session_folder):
            full_utterances.append(read_utterances_from_files(session_folder, voice_sample_folder))

    write_to_file(stripped_utterances, "utterance_dialog_act.txt")

    dialog_acts, stripped_utterances = read_target_and_labels_from_file("utterance_dialog_act_only_shuffled.txt")
    classifier, x, y, x_test, y_test, labels, vectorizer = get_fitted_model(dialog_acts, stripped_utterances)
    y_pred = classifier.predict(x_test)

    while True:
        user_input = input('\nPlease make your request and then hit [enter]. Type \'exit\' to end the program.\n' +
                           'Select: \n[1] - To utilise the random keyword classifier.\n' +
                           '[2] - To utilise the keyword classifier.\n' +
                           '[3] - To utilise the random forest classifier.\n' +
                           '[4] - To display metrics of all the above classifiers.\n-> ')
        if user_input == 'exit':
            break
        elif user_input == '1':
            use_random_keyword_classifier()
        elif user_input == '2':
            use_keyword_classifier()
        elif user_input == '3':
            use_random_forest_classifier(classifier, vectorizer, labels)
        elif user_input == '4':
            get_stats(full_utterances, x_test, y_test, x, y)
