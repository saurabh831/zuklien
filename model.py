import nltk
from nltk.stem.lancaster import LancasterStemmer
from tensorflow.python.framework import ops
import numpy
import tflearn
from nltk.corpus import stopwords
import tensorflow
import random
import json
import pickle

stemmer = LancasterStemmer()

with open("intents.json") as file:
    data = json.load(file)

stopwords = set(stopwords.words('german'))
stopwords.add("?")
def remove_(msg):
    result = [i for i in msg if not i in stopwords]
    return result

try:
    with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
except:
    words = []
    labels = []
    docs_x = []
    docs_y = []

    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

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

    with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)

ops.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

try:
    model.load("model.tflearn")
except:
    model.fit(training, output, n_epoch=1000, batch_size=8, show_metric=True)
    model.save("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    s_words = [stemmer.stem(word.lower()) for word in s_words]

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1

    return numpy.array(bag)


def chat():
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        if inp.lower() == "quit":
            break

        results = model.predict([bag_of_words(inp, words)])
        results_index = numpy.argmax(results)
        tag = labels[results_index]

        for tg in data["intents"]:
            if tg['tag'] == tag:
                responses = tg['responses']
                print(random.choice(responses))

def get_contxt(we, t):
    wrd = stemmer.stem(we.lower())
    wrdsg = nltk.word_tokenize(wrd)
    sw = remove_(wrdsg)
    wq = sw
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            c = 0
            wrd = stemmer.stem(pattern.lower())
            wrds = nltk.word_tokenize(wrd)
            for j in sw:
                for i in wrds:
                    if j == i:
                        c = c + 1
            if c == len(wq):
                for tg in intent["tag"]:
                    for tr in t:
                        if tr == tg:
                            a = intent["context_set"]
                            return a
    return ['none']

def get_resp(we, t):
    wrd = stemmer.stem(we.lower())
    wrdsg = nltk.word_tokenize(wrd)
    sw = remove_(wrdsg)
    wq = sw
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            c = 0
            wrd = stemmer.stem(pattern.lower())
            wrds = nltk.word_tokenize(wrd)
            for j in sw:
                for i in wrds:
                    if j == i:
                        c = c + 1
            if c == len(wq):
                for tg in intent["tag"]:
                    #wrds = nltk.word_tokenize(tg)
                    for tr in t:
                        if tr == tg:
                            a = intent["responses"]
                            return random.choice(a)
    return "I do not understand wrong pattern..."


def prediction_(inp, tg):
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tr in tg:
        if tr == tag[0]:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    responses = tg['responses']
                    return random.choice(responses)
    return "nothing"

def context__(inp, tg):
    results = model.predict([bag_of_words(inp, words)])
    results_index = numpy.argmax(results)
    tag = labels[results_index]
    for tr in tg:
        if tr == tag[0]:
            for tg in data["intents"]:
                if tg['tag'] == tag:
                    contxt = tg['context_set']
                    return contxt
    return "nothing"

