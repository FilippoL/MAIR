import json
import os
import keyboard

'''Script that recursively searches into folder, when 
    finds a json files (which structure was given to us),
    it reads content in order to select user utterance 
    and its value as dialog act. 
    Data is retrieved both from label.json,
    located in test/data/* directory. 
'''


#  Function writing to file from given parameters, in this case the array with all the utterance and dialog acts.
def write_dialogue_to_file(content, dialogue_index, filename):
    with open(filename, 'a') as file:
        for sentence_index in range(len(content[dialogue_index][0])):
            file.write('{0}     {1}\n'.format(content[dialogue_index][0][sentence_index],
                                              content[dialogue_index][1][sentence_index]))


def read_utterances_from_files(file, voice_sample):
    utterance_content = []
    dialog_act = []

    with open('./test/data/' + file + '/' + voice_sample + '/label.json') as label_data:
        label = json.load(label_data)

        for j in range(len(label['turns'])):
            utterance_content += [label['turns'][j]['transcription']]
            dialog_act += [label['turns'][j]['semantics']['cam']]

        session_id = label['session-id']
    return [dialog_act, utterance_content, session_id]


def keyword_classifier(utterance):
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
        'thankyou': ['thank you', 'cheers', 'thanks']
    }

    classification = []
    sentence_to_classify = utterance.lower()
    for category, keywords in categories.items():
        keywords_found = [keyword for keyword in keywords if keyword in sentence_to_classify]
        if len(keywords_found) > 0: classification.append(category)

    return classification if len(classification) > 0 else None

def run_keyword_classifier():
    while True:
        utterance = input('Please make your request and then hit \'enter\'. Type \'exit\' to end the programm.\n')
        if utterance == 'exit':
            break

        result = keyword_classifier(utterance)
        if result is not None:
            print('The following category(ies) have been identified:')
            print(*result, sep="\n")
        else:
            print('Unfortunately, no categories were detected for the sentence you entered.')


if __name__ == '__main__':
    utterances = []
    for file in os.listdir('./test/data/'):
        for voice_sample in os.listdir('./test/data/' + file):
            utterances.append(read_utterances_from_files(file, voice_sample))

    if not os.path.isfile('utterance_dialog_act.txt'): [
        write_dialogue_to_file(utterances, dialogue_index, 'utterance_dialog_act.txt') for dialogue_index in
        range(len(utterances))]

    #  85%  =  8406
    #  15%  =  1484

    run_keyword_classifier()


