import nltk
from nltk.stem.lancaster import LancasterStemmer
import numpy
from nltk.corpus import stopwords
import tflearn
import tensorflow
import random
import json
from tensorflow.python.framework import ops

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

stopwords = set(stopwords.words('german'))
stopwords.add("?")


def remove_(msg):
    result = [i for i in msg if not i in stopwords]
    return result


words = []
labels = []
docs_x = []
docs_y = []
s = " "
for intent in data["intents"]:
    for pattern in intent["patterns"]:
        wrd = stemmer.stem(pattern.lower())
        wo = nltk.word_tokenize(wrd)
        words.extend(wo)
        print(words)
        docs_x.append(wo)
        print(docs_x)
        docs_y.append(intent["tag"])
        print(docs_y)

    if intent["tag"] not in labels:
        labels.append(intent["tag"])

words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    wrds = [stemmer.stem(w.lower()) for w in doc]

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.fit(training, output, n_epoch=2000, batch_size=10, show_metric=True)
model.save("model.tflearn")