import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.tokenize import word_tokenize
import random
import PorterStemmer

pos_reviews = []
neg_reviews = []
stemmer = PorterStemmer.PorterStemmer()

with open('pos_reviews', 'r') as p:
    lines = p.readlines()
    random.shuffle(lines)
    pos_reviews = lines[0:12500]

with open('neg_reviews', 'r') as n:
    lines = n.readlines()
    random.shuffle(lines)
    neg_reviews = lines[0:12500]


def word_feats(words):
    word_dict = {}
    words = word_tokenize(words)
    words = nltk.pos_tag(words)
    for word in words:
        if word[1] == 'JJ':
            word_dict[word[0]] = True
    return word_dict


negfeats = [(word_feats(f), 'neg') for f in neg_reviews]
posfeats = [(word_feats(f), 'pos') for f in pos_reviews]

negcutoff = len(negfeats)*3/4
poscutoff = len(posfeats)*3/4

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print 'train on %d instances, test on %d instances' % (len(trainfeats), len(testfeats))

classifier = NaiveBayesClassifier.train(trainfeats)
print 'accuracy:', nltk.classify.util.accuracy(classifier, testfeats)
classifier.show_most_informative_features()

