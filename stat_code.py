#STAT 471 Project Appendix
#Python Code

# David Zbarsky: dzbarsky@wharton.upenn.edu
# Yaou Wang: yaouwang@wharton.upenn.edu

from nltk.corpus import PlaintextCorpusReader
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.stem.porter import PorterStemmer
from xml.etree import ElementTree
import math
import string
import random
import fileinput
import os
import itertools
import subprocess
import xml.etree.ElementTree as ET
import matplotlib.pyplot as Plot
import numpy as np
import statsmodels.api as sm
import scipy.stats as stats
from sklearn import linear_model
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import CCA
from sklearn.naive_bayes import MultinomialNB

def get_all_files(directory):
    # We assume that a filename with a . is always a file rather than a directory
    # IF we were passed a file, just return that file.  This simplifies representing documents because they need to handle single files.
    if directory.find('.') < 0:
        return PlaintextCorpusReader(directory, '.*').fileids()
    #if directory is a file return the file in a list
    return [directory]

def load_file_sentences(filepath):
    index = filepath.rfind('/')
    if index < 0:
        sents = sent_tokenize(PlaintextCorpusReader('.', filepath).raw())
    else:
        sents = sent_tokenize(PlaintextCorpusReader(filepath[:index], filepath[index+1:]).raw())
    return sents

def load_file_tokens(filepath):
    tokens = []
    for sentence in load_file_sentences(filepath):
        tokens.extend(word_tokenize(sentence))
    return tokens

def load_collection_tokens(directory):
    tokens = []
    for file in get_all_files(directory):
        tokens.extend(load_file_tokens(directory + '/' + file))
    return tokens

#returns a list of all words occurring >= 5 times in the directory
def extract_top_words(directory):
    tokens = load_collection_tokens(directory)
    freq = dict()
    top_words = []
    for token in tokens:
        if token in freq.keys():
            freq[token] += 1
        else:
            freq[token] = 1

    a = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    words = [r[1] for r in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
    indices = [n for n in range(len(words))]
    Plot.plot([math.log1p(i) for i in indices], [math.log1p(w) for w in words])
    Plot.title('Zipf Distribution')
    Plot.xlabel('Log of Index')
    Plot.ylabel('Log of Word Frequency')
    Plot.savefig('zipf.png')
    Plot.show()

# Produces a histogram of article lengths
def print_file_length_hist(directory):
    map = dict()
    for file in get_all_files(directory):
        map[file] = len(load_file_tokens(directory + '/' + file))
    Plot.hist(map.values(), 20)
    Plot.title('Histogram of Article Length')
    Plot.xlabel('Length in Words')
    Plot.ylabel('Frequency')
    Plot.savefig('histogram.png')

    print "Mean: " + str(numpy.mean(map.values()))
    print "Standard Deviation: " + str(numpy.std(map.values()))

    Plot.show()

#uses filelist to group and divide the appropriate files
#in corpusroot
def extract_returns(filelist):
    returns = []
    index = filelist.rfind('/')
    if index < 0:
        tokens = word_tokenize(PlaintextCorpusReader('.', filelist).raw())
    else:
        tokens = word_tokenize(PlaintextCorpusReader(filelist[:index], filelist[index+1:]).raw())
    i = 0
    while i < len(tokens):
        returns.append(float(tokens[i+1]))
        i += 2

    print "Mean: " + str(np.mean(returns))
    print "Standard Deviation: " + str(np.std(returns))

    Plot.hist(returns, 100)
    Plot.title('Distribution of Returns')
    Plot.xlabel('Percentage Return')
    Plot.ylabel('Frequency')
    Plot.savefig('returns.png')
    Plot.show()

    return sorted(returns)

def normalize(word):
    try:
        word = float(word.replace(',', ''))
        if word < 0:
            return "<NEGATIVE>"
        if word > 0:
            return "<POSITIVE>"
        return "<ZERO>"
    except ValueError:
        return word

def generate_bag_of_words(xret_tails):
    returns = dict()

    fileDicts = []

    allWords = set()

    for line in open(xret_tails):
        file = line[:line.find('\t')]
        line = line[line.find('\t')+1:]
        returns[file] = line.strip()

    filenames = returns.keys()

    # Delete the files that are in the test set.
    filenames2 = []
    for file in filenames:
        try:
            with open('data/' + file):
                filenames2.append(file)
        except IOError:
            pass
    filenames = filenames2

    for file in filenames:
        map = dict()
        for word in load_file_tokens('data/' + file):
            word = normalize(word)
            if word not in map:
                map[word] = 1
            else:
                map[word] += 1
            allWords.add(word)
        fileDicts.append(map)

    allBagsofWords = []
    for i in range(len(filenames)):
        file = filenames[i]
        map = fileDicts[i]
        arr = []
        for word in allWords:
            if word in map:
                arr.append(map[word])
            else:
                arr.append(0)
        allBagsofWords.append(arr)

    returnsVector = []
    for file in filenames:
        returnsVector.append(float(returns[file]))

    return (allBagsofWords, returnsVector,
            [i/math.fabs(i) for i in returnsVector], list(allWords))
    #print "Returns: " + str(returnsVector)
    #print "Matrix: " + str(allBagsofWords)

def generate_pred(xret_tails):
    returns = dict()

    fileDicts = []

    allWords = set()

    for line in open(xret_tails):
        file = line[:line.find('\t')]
        line = line[line.find('\t')+1:]
        returns[file] = line.strip()

    filenames = returns.keys()

    # Delete the files that are in the train set.
    filenames2 = []
    for file in filenames:
        try:
            with open('test_data/' + file):
                filenames2.append(file)
        except IOError:
            pass
    filenames = filenames2

    for file in filenames:
        map = dict()
        for word in load_file_tokens('test_data/' + file):
            word = normalize(word)
            if word not in map:
                map[word] = 1
            else:
                map[word] += 1
            allWords.add(word)
        fileDicts.append(map)

    allBagsofWords = []
    for i in range(len(filenames)):
        file = filenames[i]
        map = fileDicts[i]
        arr = []
        for word in allWords:
            if word in map:
                arr.append(map[word])
            else:
                arr.append(0)
        allBagsofWords.append(arr)

    returnsVector = []
    for file in filenames:
        returnsVector.append(float(returns[file]))

    return (allBagsofWords, returnsVector,
            [i/math.fabs(i) for i in returnsVector], list(allWords))

def generate_bigram(xret_tails):


def generate_bigram_pred(xret_tails):
	
    
def bonferroni_regression(y, matrix):
    matrix = np.array(matrix)
    betas = []
    currentArray = np.ones((matrix.shape[0], 1), dtype=float)
    while True:
        max_model = None
        max_sig = 0
        max_column = 0
        max_beta = 0
        max_pvalue = 0
        prev_aic = 0
        for i in range(len(matrix[0])):
            if i not in betas:
                X = matrix[:,i:i+1]
                X = np.append(currentArray, X, 1)
                model = sm.OLS(y, X)
                results = model.fit()
                sig = results.tvalues[len(results.params)-1]
                if sig > max_sig:
                    max_sig = sig 
                    max_column = i
                    max_beta = results.params[len(results.params)-1]
                    max_pvalue = results.pvalues[len(results.params)-1]
                    max_model = results
                    max_aic = results.rsquared_adj
        
        if max_pvalue > 0.05/len(matrix[0]):
            prev_aic = max_aic
            betas.append(max_column)
            currentArray = np.append(currentArray, matrix[:,max_column:max_column+1], 1)
            print max_model.summary()
            print betas 
        else:
            break
    
    return betas

def aic_regression(y, matrix):
    matrix = np.array(matrix)
    betas = []
    currentArray = np.ones((matrix.shape[0], 1), dtype=float)
    while True:
        max_model = None
        max_sig = 0
        max_column = 0
        max_beta = 0
        max_pvalue = 0
        prev_aic = 0
        for i in range(len(matrix[0])):
            if i not in betas:
                X = matrix[:,i:i+1]
                X = np.append(currentArray, X, 1)
                model = sm.OLS(y, X)
                results = model.fit()
                sig = results.tvalues[len(results.params)-1]
                if sig > max_sig:
                    max_sig = sig 
                    max_column = i
                    max_beta = results.params[len(results.params)-1]
                    max_pvalue = results.pvalues[len(results.params)-1]
                    max_model = results
                    max_aic = results.rsquared_adj
        
        if max_aic > prev_aic:
            prev_aic = max_aic
            betas.append(max_column)
            currentArray = np.append(currentArray, matrix[:,max_column:max_column+1], 1)
            print max_model.summary()
            print betas 
        else:
            break
    
    return betas

def main():
    #print_file_length_hist('data')
    #extract_top_words('data')
    '''
	betas = bonferroni_regression(y, matrix)
	
    #code ran in the console
    
    matrix, y, y2, wordlist = generate_bag_of_words('xret_tails.txt')
    matrix = np.array(matrix)
    y2 = np.array(y2)
    x_pred, y_pred = generate_pred('xret_tails.txt')
	bigram = generate_bigram('xret_tails.txt')
	bigram_pred = generate_bigram_pred('xret_tails.txt')
    
    #stepwise regression on bag of words
    clf = linear_model.Lars()
    clf.fit(matrix, y2)
    for i in range(len(clf.coef_)):
        if clf.coef_[i] != 0:
            print wordlist[i]
            print clf.coef_[i]
            print clf.get_params(i)

    #predictive accuracy
    clf.score(x_pred, y_pred)

    #pca
    pca = PCA()
    pca.fit(matrix)
    pca.explained_variance_ratio_
    pcam = pca.components_
    clf.fit(pcam, y2)
    for i in range(150):
        print clf.coef_[i]
        print clf.get_params(i)
		
    #cca
    clf = CCA()
	clf.fit(bigram, y2)
	for i in range(len(clf.coef_)):
        if clf.coef_[i] != 0:
            print wordlist[i]
            print clf.coef_[i]
            print clf.get_params(i)
	clf.score(bigram_pred, y_pred)
    
    #naive bayes
    clf = MultinomialNB()
	clf.fit (matrix, y2)
	for i in range(len(clf.coef_[0])):
    	if clf.coef_[0][i] != 0:
                print wordlist[i]
                print clf.coef_[0][i]
    clf.score(x_pred, y_pred) 

    
    

    #returns = extract_returns('xret_tails.txt')
    #print returns
    '''
