# ------- importing all the required libraries ---------

import collections as coll
import math
import string

import matplotlib.pyplot as plt
import nltk
import numpy as np
import scipy as sc
from matplotlib import style
from nltk.corpus import cmudict
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

nltk.download('cmudict')
nltk.download('stopwords')

style.use("ggplot")
cmuDictionary = None

# ------ This method returns the average no of words ------
def sentence_length_by_word_avg(text):
    tokens = sent_tokenize(text)
    return np.average([len(token.split()) for token in tokens])
    
    
# ------ This method removes stop words -----
def word_length_avg(str):
    str.translate(string.punctuation)
    tokens = word_tokenize(str, language='english')
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    stop = stopwords.words('english') + st
    words = []
    i = 0
    while i < len(tokens):
        if tokens[i] not in stop:
            words.append(tokens[i])
        i += 1
    return np.average([len(word) for word in words])
    
# ------ This method counts special characters ------
def special_character_count(text):
    st = ["#", "$", "%", "&", "(", ")", "*", "+", "-", "/", "<", "=", '>',
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']
    count = 0
    i = 0
    while i < len(text):
        if text[i] in st:
            count = count + 1
        i += 1

    return count / len(text)

# ------ This method counts punctuation --------
def punctuation_count(text):
    st = [",", ".", "'", "!", '"', ";", "?", ":", ";"]
    count = 0
    i = 0
    while i < len(text):
        if text[i] in st:
            count = count + 1
        i += 1
    return float(count) / float(len(text))
    
# ----- This method counts syllables manually ------
def manual_count_syllable(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if word[0] in vowels:
        count += 1

    index = 1
    while index < len(word):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
            if word.endswith("e"):
                count -= 1
        index += 1

    if count == 0:
        count += 1
    return count


# ----- This method counts the syllables ------
def count_syllable(word):
    global cmuDictionary
    d = cmuDictionary
    try:
        syl = [len(list(y for y in x if y[-1].isdigit())) for x in d[word.lower()]][0]
    except:
        syl = manual_count_syllable(word)
    return syl
    

# ------- This method divides the document into chunks --------
def slidingWindow(data, window_Size, step_Size=1):
    try:
        it = iter(data)
    except TypeError:
        raise Exception("-------- Text data must be iterable -----------")
    if not ((type(window_Size) == type(0)) and (type(step_Size) == type(0))):
        raise Exception("--------- type(window_Size) and type(step_Size) must be int ----------")
    if step_Size > window_Size:
        raise Exception("--------- step_Size must not be larger than window_Size ---------")
    if window_Size > len(data):
        raise Exception("--------- window_Size must not be larger than text data length ---------")

    data = sent_tokenize(data)

    # Find the chunks
    numOfChunks = int(((len(data) - window_Size) / step_Size) + 1)

    value = []
    # Remove the chunks
    for i in range(0, numOfChunks * step_Size, step_Size):
        value.append(" ".join(data[i:i + window_Size]))

    return value

# -------- This method removes special characters --------
def special_characters_removal(text):
    text = word_tokenize(text)
    st = [",", ".", "'", "!", '"', "#", "$", "%", "&", "(", ")", "*", "+", "-", ".", "/", ":", ";", "<", "=", '>', "?",
          "@", "[", "\\", "]", "^", "_", '`', "{", "|", "}", '~', '\t', '\n']

    words = []
    i = 0
    while i < len(text):
        if text[i] not in st:
            words.append(text[i])
        i += 1
    return words

# ------ This method demonstrates shannon entropy ---------
# -1*sigma(pi*lnpi)
def shannon_entropy(text):
    words = special_characters_removal(text)
    length = len(words)

    freqs = coll.Counter()
    freqs.update(words)

    arr = np.array(list(freqs.values()))

    distribution = 1. * arr
    distribution /= max(1, length)

    H = sc.stats.entropy(distribution, base=2)
    return H

# -------- This method demonstrates Yules Characteristic ---------
# K = 10,000 * (M - N) / N**2 where M  Sigma i**2 * Vi.
def yules_characteristic(text):
    words = special_characters_removal(text)
    N = len(words)

    freqs = coll.Counter()
    freqs.update(words)

    vi = coll.Counter()
    vi.update(freqs.values())

    M = sum([(value * value) * vi[value] for key, value in freqs.items()])

    K = 10000 * (M - N) / math.pow(N, 2)
    return K


# -------- This method demonstrates Simsons index ---------
# 1 - (sigma(n(n - 1))/N(N-1)
# here N is total number of words
# here n is the number of each type of word
def simpsons_index(text):
    words = special_characters_removal(text)

    freqs = coll.Counter()
    freqs.update(words)

    N = len(words)

    n = sum([1.0 * i * (i - 1) for i in freqs.values()])

    D = 1 - (n / (N * (N - 1)))

    return D

# -------- This method demonstrates gunning fox index ---------
def gunning_fox_index(text, NoOfSentences):
    words = special_characters_removal(text)

    NoOFWords = float(len(words))
    complexWords = 0

    for word in words:
        if (count_syllable(word) > 2):
            complexWords += 1

    gun = 0.4 * ((NoOFWords / NoOfSentences) + 100 * (complexWords / NoOFWords))

    return gun


# --------- This method returns a feature vector with all the captured features -------------
def ExtractingFeatures(data, window_Size, step_Size):
    global cmuDictionary
    cmuDictionary = cmudict.dict()

    chunks_of_data = slidingWindow(data, window_Size, step_Size)
    vector_data = []
    for chunk in chunks_of_data:
        features = []

        # LEXICAL FEATURES

        # Calling avg word per length method
        avg_wl = (word_length_avg(chunk))
        features.append(avg_wl)

        # Calling avg sentence length by word method
        avg_sl = (sentence_length_by_word_avg(chunk))
        features.append(avg_sl)

        # Calling count special character method
        special = special_character_count(chunk)
        features.append(special)

        # Calling punctuation count method
        punc = punctuation_count(chunk)
        features.append(punc)

        # VOCABULARY RICHNESS FEATURES

        # Calling Yules Characteristic method
        yules_k = yules_characteristic(chunk)
        features.append(yules_k)

        # Calling Simpsons Index method
        simp = simpsons_index(chunk)
        features.append(simp)

        # Calling Shannon Entropy method
        shannon = shannon_entropy(data)
        features.append(shannon)

        # READIBILTY FEATURES

        # Calling Gunning fox index method
        gunning = gunning_fox_index(chunk, window_Size)
        features.append(gunning)

        vector_data.append(features)

    return vector_data


# Finding optimal value of k using elbow method
def ElbowRoutine(data):
    X = data
    distorsions = []

    for k in range(1, 10):
        kmeans_algo = KMeans(n_clusters=k)
        kmeans_algo.fit(X.reshape(-1, 1))
        distorsions.append(kmeans_algo.inertia_)

    fig = plt.figure(figsize=(15, 5))
    plt.plot(range(1, 10), distorsions, 'bo-')
    plt.grid(True)
    plt.ylabel("SSE")
    plt.xlabel("No. of clusters")
    plt.title('Elbow curve')
    plt.savefig("ElbowCurve.png")
    plt.show()


# Finding k value using the elbow graph above.
def DataAnalysis(vector, K=2):
    arr = (np.array(vector))

    # Normalization of data
    sc = StandardScaler()
    x = sc.fit_transform(arr)

    # Principal Component Analysis
    pca = PCA(n_components=2)
    pc_components = (pca.fit_transform(x))

    # Using unsupervised K-means to determine centroids
    kmeans_algo = KMeans(n_clusters=K)
    kmeans_algo.fit_transform(pc_components)
    print("Labels : ", kmeans_algo.labels_)
    centers = kmeans_algo.cluster_centers_

    # Assigning the labels
    labels = kmeans_algo.labels_
    colors = ["r.", "g.", "b.", "y.", "c."]
    colors = colors[:K + 1]

    for i in range(len(pc_components)):
        plt.plot(pc_components[i][0], pc_components[i][1], colors[labels[i]], markersize=10)

    plt.scatter(centers[:, 0], centers[:, 1], marker="x", s=150, linewidths=10, zorder=15)
    plt.xlabel("1st PC")
    plt.ylabel("2nd PC")
    title = "Styles Clusters"
    plt.title(title)
    plt.savefig("PC_Results" + ".png")
    plt.show()


if __name__ == "__main__":
    data = open("Test.txt").read()
    vector = ExtractingFeatures(data, window_Size=10, step_Size=10)
    ElbowRoutine(np.array(vector))
    DataAnalysis(vector)
