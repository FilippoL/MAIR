import json
import os
import keyboard

'''Script that recursively searches into folder, when 
    finds a json files (which structure was given to us),
    it reads content in order to form a dialogue. 
    Data is retrieved both from label.json (user) and
    from log.json (system) files, located in test/data/* directory. 
'''


#  Function writing to file from given parameters, in this
#  case the array with all the dialogs and the file name with
#  extension type.
def write_to_file(content, filename):
    if not os.path.isfile(filename):  # Checking if file already exists, don't append data if it does.
        for j in range(len(content)):  # For each dialog in dialogues array.
            with open(filename, 'a') as file:  # Open a text file in append mode and write data into it.
                file.write('Session ID: {0}\n'.format(content[j][2]))
                file.write('{0}\n'.format(content[j][3]))
                for k in range(len(content[j][0])):
                    file.write('System: {0}\n'.format(content[j][0][k]))
                    if content[j][1]:
                        file.write('User: {0}\n'.format(content[j][1][k]))
                file.write('\n')


def main():
    dialogs = []  # Array that will hold all our dialogues.
    for session in os.listdir('./test/data/'):  # For each file in the directory.
        for i in range(1):
            for voice_sample in os.listdir('./test/data/' + session):  # Append latter file to
                # directory.
                dialog_system = []  # Empty array to hold system utterance.
                dialog_user = []  # Empty array to hold user utterance.
                with open('./test/data/' + session + '/' + voice_sample + '/log.json') as log_data:
                    log = json.load(log_data)  # This wil hold our json file as a dictionary for system.

                    for k in range(len(log['turns'])):  # Looking for turns in the dictionary.
                        # For each turn, look at output and transcript value and append to system dialog.
                        dialog_system += [log['turns'][k]['output']['transcript']]

                with open('./test/data/' + session + '/' + voice_sample + '/label.json') as label_data:
                    label = json.load(label_data)  # This wil hold our json file as a dictionary for user.

                    for j in range(len(label['turns'])):  # Looking for turns in the dictionary.
                        # For each turn, look at output and transcript value and append to user dialog.
                        dialog_user += [label['turns'][j]['transcription']]

                    # Now store few more information we want to show the final user.
                    session_id = label['session-id']
                    task_info = label['task-information']['goal']['text']
                dialogs.append([dialog_system, dialog_user, session_id, task_info])

    write_to_file(dialogs, 'DataDomainModelling/dialogs.txt')  # Function call

    print('I am ready to display dialogs - press enter to start.\n')  # Acknowledge user that program is ready on input.

    # Displaying everything.
    for i in range(len(dialogs)):
        waiting = True
        while waiting:
            if keyboard.is_pressed('\n'):
                print('\n')
                print('Session ID: {0}'.format(dialogs[i][2]))
                print(dialogs[i][3])
                for k in range(len(dialogs[i][0])):
                    print('System: {0}'.format(dialogs[i][0][k]))
                    if dialogs[i][1]:
                        print('User: {0}'.format(dialogs[i][1][k]))
                waiting = False


if __name__ == '__main__':
    main()
