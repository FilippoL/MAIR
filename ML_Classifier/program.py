import pickle
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, LabelBinarizer, OneHotEncoder
from sklearn.metrics import classification_report, accuracy_score
from numpy import array
import warnings
import os

warnings.filterwarnings('ignore')

stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")

dialog_acts = []
utterances = []

with open("utterance_dialog_act_only_shuffled.txt", "r") as file:
    for line in file.readlines():
        dialog_acts.append(line.split()[0])
        utterances.append(stemmer.stem(line[line.find(" "):].strip()))

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(utterances).toarray()

array(dialog_acts)

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(dialog_acts)

one_hot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)

y = one_hot_encoder.fit_transform(integer_encoded)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=0)

if not os.path.isfile("model_binary_RFC.sav"):
    with open("model_binary_RFC.sav", "wb") as model:
        classifier = RandomForestClassifier(n_estimators=500)
        classifier.fit(X_train, y_train)
        try:
            pickle.dump(classifier, model)
            y_pred = classifier.predict(X_test)
            print("\nAccuracy of train set: {0}".format(classifier.score(X, y)))
            print("Accuracy of test set: {0}\n\n".format(accuracy_score(y_test, y_pred)))
        except MemoryError:
            print("I failed dumping.\n\n")
            y_pred = classifier.predict(X_test)
            print("\nAccuracy of train set: {0}".format(classifier.score(X, y)))
            print("Accuracy of test set: {0}\n\n".format(accuracy_score(y_test, y_pred)))
else:
    with open('model_binary_RFC.sav', 'rb') as training_model:
        model = pickle.load(training_model)
        y_pred = model.predict(X_test)
        print("\nAccuracy of train set: {0}".format(model.score(X, y)))
        print("Accuracy of test set: {0}\n\n".format(accuracy_score(y_test, y_pred)))

print(classification_report(y_test, y_pred, target_names=label_encoder.classes_))

running = True
while running:
    new_y = []
    utterance = input("\n\nEnter utterance you want to classify:\n> ").lower()
    if utterance == "exit" or "":
        running = False
    else:
        if not os.path.isfile("model_binary_RFC.sav"):
            y_pred = classifier.predict(vectorizer.transform([utterance]))
        else:
            y_pred = model.predict(vectorizer.transform([utterance]))
            try:
                label_pred = label_encoder.classes_[y_pred[0].tolist().index(1)]
                print("X= %s, Prediction = %s" % (y_pred, label_pred))
            except ValueError:
                print("X= %s, Prediction = %s" % (y_pred, y_pred))
