import json
import os
from random import shuffle
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
                    file.write('{0}     {1}\n'.format(str(content[j][0][k]).lower().split("(")[0], str(content[j][1][k])).lower())


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
    shuffle(utterances)
    write_to_file(utterances, 'utterance_dialog_act_only_shuffled.txt')  # Function call

    # Displaying everything.
    for i in range(len(utterances)):
        print('\n')
        print('Session ID: {0}'.format(utterances[i][2]))
        for k in range(len(utterances[i][0])):
            print('{0}     {1}'.format(utterances[i][0][k], utterances[i][1][k]))


if __name__ == '__main__':
    main()

