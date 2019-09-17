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


#  Function writing to file from given parameters, in this
#  case the array with all the utterance and dialog acts.
def write_to_file(content, filename):
    if not os.path.isfile(filename):  # Checking if file already exists, don't append data if it does.
        for j in range(len(content)):  # For each dialog in dialogues array.
            with open(filename, 'a') as file:  # Open a text file in append mode and write data into it.
                for k in range(len(content[j][0])):
                    file.write('{0}     {1}\n'.format(content[j][0][k], content[j][1][k]))


def main():
    utterances = []  # Array that will hold all our dialogues.
    for session in os.listdir('./test/data/'):  # For each file in the directory.
        for i in range(1):
            for voice_sample in os.listdir('./test/data/' + session):  # Append latter file to
                # directory.
                utterance_content = []  # Empty array to hold utterance.
                dialog_act = []  # Empty array to hold dialog act.
                t_shuffle_array = []

                with open('./test/data/' + session + '/' + voice_sample + '/label.json') as label_data:
                    label = json.load(label_data)  # This wil hold our json file as a dictionary for user.

                    for j in range(len(label['turns'])):  # Looking for turns in the dictionary.
                        # For each turn, look at output and transcript value and append to user dialog.
                        utterance_content += [label['turns'][j]['transcription']]
                        dialog_act += [label['turns'][j]['semantics']['cam']]

                    # Now store few more information we want to show the final user.
                    session_id = label['session-id']
                utterances.append([dialog_act, utterance_content, session_id])

    write_to_file(utterances, 'utterance_dialog_act.txt')  # Function call

    # Displaying everything.
    for i in range(len(utterances)):
        print('\n')
        print('Session ID: {0}'.format(utterances[i][2]))
        for k in range(len(utterances[i][0])):
            print('{0}     {1}'.format(utterances[i][0][k], utterances[i][1][k]))


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


if __name__ == '__main__':
    main()

    #  85%  =  8406
    #  15%  =  1484

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
