import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

stemmer = SnowballStemmer("english")
stop_words = stopwords.words("english")

dialog_acts = []
utterances = []

with open("utterance_dialog_act_only_shuffled.txt", "r") as file:
    for line in file.readlines():
        dialog_acts.append(line.split()[0])
        utterances.append(stemmer.stem(line[line.find(" "):].strip()))

vectorizer = CountVectorizer(max_features=1500, stop_words=stop_words)
X = vectorizer.fit_transform(utterances).toarray()

lb = LabelBinarizer()
y = lb.fit_transform(dialog_acts)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = RandomForestClassifier(n_estimators=1000, random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

print(y_pred)


#print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
print(accuracy_score(y_test, y_pred))
