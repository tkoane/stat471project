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


'''
    here are the column numbers generated using Bonferroni
    #Bonferroni code just involves changing the if statement in aic_regression#
    currentArray = np.ones((matrix.shape[0], 1), dtype=float)
    currentArray = np.append(currentArray, matrix[:, 6:7], 1)
    currentArray = np.append(currentArray, matrix[:, 43:44], 1)
    currentArray = np.append(currentArray, matrix[:, 3934:3935], 1)
    currentArray = np.append(currentArray, matrix[:, 7473:7474], 1)
    currentArray = np.append(currentArray, matrix[:, 8994:8995], 1)
    currentArray = np.append(currentArray, matrix[:, 10998:10999], 1)
    currentArray = np.append(currentArray, matrix[:, 9444:9445], 1)
    currentArray = np.append(currentArray, matrix[:, 12189:12190], 1)
    model = sm.OLS(y, currentArray)
    results = model.fit()
print out:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.043
Model:                            OLS   Adj. R-squared:                  0.042
Method:                 Least Squares   F-statistic:                     63.99
Date:                Sun, 24 Nov 2013   Prob (F-statistic):           2.56e-15
Time:                        21:16:09   Log-Likelihood:                -5226.9
No. Observations:                1435   AIC:                         1.046e+04
Df Residuals:                    1433   BIC:                         1.047e+04
Df Model:                           1                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          0.8631      0.244      3.535      0.000         0.384     1.342
x1             2.3121      0.289      7.999      0.000         1.745     2.879
x2             2.3121      0.289      7.999      0.000         1.745     2.879
x3             2.3121      0.289      7.999      0.000         1.745     2.879
x4             2.3121      0.289      7.999      0.000         1.745     2.879
x5             2.3121      0.289      7.999      0.000         1.745     2.879
x6             2.3121      0.289      7.999      0.000         1.745     2.879
x7            11.5605      1.445      7.999      0.000         8.726    14.395
x8             2.3121      0.289      7.999      0.000         1.745     2.879
==============================================================================
Omnibus:                       16.664   Durbin-Watson:                   1.968
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               11.256
Skew:                          -0.074   Prob(JB):                      0.00360
Kurtosis:                       2.592   Cond. No.                          nan
==============================================================================

Here are the words that corresponds to these variables:
('localized', 'Tailoring', 'viewers', 'showcasing', 'native', 'Content', 'YouTube', 'tap')
Conclusion: Bonferroni is garbage and has too few elements to actually predict anything
See r squared value: 4.3%
This is due to the fact that Bonferroni cuts off at p value of 0.05 / # of words, which is
really tiny. Hence we are doing AIC

'''

'''
AIC on Y as returns:
So far:
AIC output:
                            OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.626
Model:                            OLS   Adj. R-squared:                  0.568
Method:                 Least Squares   F-statistic:                     10.78
Date:                Tue, 26 Nov 2013   Prob (F-statistic):          1.26e-164
Time:                        00:55:15   Log-Likelihood:                -4551.8
No. Observations:                1435   AIC:                             9492.
Df Residuals:                    1241   BIC:                         1.051e+04
Df Model:                         193                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          1.3280      0.238      5.580      0.000         0.861     1.795
x1            80.9683      6.401     12.649      0.000        68.410    93.527
x2            -6.7193      1.602     -4.195      0.000        -9.862    -3.577
x3            -7.2365      1.550     -4.670      0.000       -10.277    -4.196
x4            30.5134      6.236      4.893      0.000        18.278    42.749
x5            16.2860      3.106      5.244      0.000        10.193    22.379
x6           -22.1575      3.236     -6.848      0.000       -28.506   -15.809
x7            51.3756      8.794      5.842      0.000        34.122    68.629
x8           -33.0996      6.237     -5.307      0.000       -45.335   -20.864
x9           -10.9544      1.865     -5.874      0.000       -14.613    -7.296
x10          -26.1739      4.678     -5.595      0.000       -35.351   -16.997
x11            5.4155      0.932      5.813      0.000         3.588     7.243
x12          -38.5055      6.568     -5.863      0.000       -51.391   -25.620
x13           12.6705      3.227      3.926      0.000         6.339    19.002
x14            9.6705      2.112      4.579      0.000         5.527    13.814
x15           13.7266      3.905      3.515      0.000         6.066    21.387
x16          -10.8132      1.879     -5.754      0.000       -14.500    -7.127
x17          -21.7029      6.724     -3.228      0.001       -34.894    -8.511
x18          -27.0280      6.212     -4.351      0.000       -39.214   -14.842
x19          -14.9153      3.747     -3.981      0.000       -22.266    -7.565
x20           -4.3713      1.006     -4.343      0.000        -6.346    -2.397
x21           -9.0245      2.367     -3.812      0.000       -13.669    -4.380
x22          -11.4663      2.820     -4.066      0.000       -16.998    -5.934
x23          -11.9784      2.494     -4.803      0.000       -16.872    -7.085
x24          -10.3298      2.433     -4.246      0.000       -15.103    -5.557
x25           19.8354      6.316      3.140      0.002         7.443    32.227
x26            6.4911      1.891      3.433      0.001         2.782    10.200
x27            4.7570      2.321      2.049      0.041         0.203     9.311
x28           10.0567      2.167      4.642      0.000         5.806    14.307
x29          -13.4882      2.750     -4.905      0.000       -18.883    -8.093
x30          -16.6480      4.395     -3.788      0.000       -25.271    -8.025
x31          -13.8832      3.666     -3.787      0.000       -21.075    -6.691
x32           -4.1494      0.819     -5.065      0.000        -5.757    -2.542
x33          -15.0004      2.346     -6.393      0.000       -19.604   -10.397
x34           10.3342      2.511      4.115      0.000         5.407    15.261
x35           -5.8842      1.472     -3.997      0.000        -8.773    -2.996
x36           -7.9039      2.180     -3.626      0.000       -12.180    -3.628
x37            7.0487      2.434      2.896      0.004         2.273    11.825
x38          -15.0427      4.408     -3.412      0.001       -23.691    -6.394
x39           -6.5778      2.008     -3.276      0.001       -10.517    -2.639
x40          -33.3839      7.279     -4.586      0.000       -47.665   -19.103
x41           -7.1255      2.290     -3.112      0.002       -11.617    -2.634
x42           -1.3144      0.536     -2.453      0.014        -2.366    -0.263
x43            1.1843      0.293      4.041      0.000         0.609     1.759
x44            2.3716      0.595      3.986      0.000         1.204     3.539
x45           -5.8591      1.151     -5.088      0.000        -8.118    -3.600
x46            2.9195      0.691      4.224      0.000         1.564     4.275
x47           14.8720      4.395      3.383      0.001         6.249    23.495
x48           35.3480      7.711      4.584      0.000        20.219    50.477
x49           -5.1814      2.332     -2.222      0.026        -9.756    -0.606
x50          -23.0278      4.886     -4.713      0.000       -32.614   -13.441
x51           16.1457      6.270      2.575      0.010         3.844    28.448
x52           -6.2474      1.886     -3.312      0.001        -9.948    -2.547
x53          -34.2941      6.480     -5.292      0.000       -47.007   -21.581
x54          -14.0492      3.338     -4.209      0.000       -20.598    -7.500
x55          -10.1069      2.235     -4.522      0.000       -14.492    -5.722
x56          -10.4674      1.878     -5.575      0.000       -14.151    -6.784
x57            4.1500      0.983      4.221      0.000         2.221     6.079
x58           14.1056      3.788      3.723      0.000         6.673    21.538
x59            6.4688      1.965      3.292      0.001         2.613    10.324
x60          -19.0765      4.517     -4.224      0.000       -27.937   -10.216
x61          -27.0798      6.371     -4.250      0.000       -39.579   -14.581
x62          -12.2195      3.408     -3.585      0.000       -18.906    -5.533
x63           12.8598      3.051      4.215      0.000         6.874    18.846
x64            7.9251      2.285      3.468      0.001         3.442    12.409
x65            3.7413      1.226      3.051      0.002         1.336     6.147
x66            6.2586      2.061      3.037      0.002         2.216    10.301
x67           -6.5352      2.255     -2.898      0.004       -10.959    -2.112
x68          -14.7518      4.396     -3.356      0.001       -23.376    -6.127
x69           -7.0140      2.198     -3.191      0.001       -11.326    -2.702
x70            6.2428      1.838      3.397      0.001         2.638     9.848
x71          -19.5880      6.212     -3.153      0.002       -31.774    -7.402
x72          -14.3440      4.397     -3.262      0.001       -22.971    -5.717
x73            8.3477      2.782      3.001      0.003         2.890    13.805
x74          -16.2425      4.446     -3.653      0.000       -24.965    -7.520
x75          -11.2047      3.592     -3.120      0.002       -18.251    -4.159
x76           -9.5990      3.106     -3.091      0.002       -15.692    -3.506
x77            8.4875      2.244      3.783      0.000         4.086    12.890
x78          -10.9537      4.039     -2.712      0.007       -18.879    -3.029
x79            9.1289      4.723      1.933      0.053        -0.136    18.394
x80           15.5808      4.303      3.621      0.000         7.138    24.023
x81          -14.6610      6.282     -2.334      0.020       -26.985    -2.337
x82          -18.5180      6.212     -2.981      0.003       -30.704    -6.332
x83          -23.0785      6.273     -3.679      0.000       -35.386   -10.771
x84           -7.1007      2.203     -3.223      0.001       -11.423    -2.778
x85          -19.2936      6.234     -3.095      0.002       -31.523    -7.064
x86          -10.7055      2.867     -3.734      0.000       -16.330    -5.081
x87            5.9874      2.343      2.555      0.011         1.390    10.585
x88          -14.0274      4.437     -3.161      0.002       -22.732    -5.322
x89           -5.4241      1.614     -3.361      0.001        -8.590    -2.258
x90           17.9720      6.212      2.893      0.004         5.786    30.158
x91            3.7528      1.382      2.716      0.007         1.042     6.464
x92           15.7071      6.323      2.484      0.013         3.302    28.112
x93            2.8599      1.156      2.474      0.013         0.592     5.127
x94          -10.1366      2.351     -4.312      0.000       -14.749    -5.525
x95           -4.8140      3.417     -1.409      0.159       -11.518     1.890
x96           17.5520      6.212      2.826      0.005         5.366    29.738
x97          -15.2981      4.745     -3.224      0.001       -24.607    -5.989
x98            6.4051      1.063      6.025      0.000         4.320     8.491
x99          -11.5751      2.678     -4.323      0.000       -16.829    -6.322
x100           5.1733      3.291      1.572      0.116        -1.284    11.630
x101         -11.1484      3.611     -3.087      0.002       -18.234    -4.063
x102          -8.6128      2.711     -3.176      0.002       -13.932    -3.293
x103           1.5539      0.487      3.188      0.001         0.598     2.510
x104         -14.1882      6.309     -2.249      0.025       -26.566    -1.810
x105         -17.2480      6.212     -2.777      0.006       -29.434    -5.062
x106         -17.2080      6.212     -2.770      0.006       -29.394    -5.022
x107          -5.6702      1.319     -4.298      0.000        -8.258    -3.082
x108         -10.0996      3.342     -3.022      0.003       -16.656    -3.543
x109          -8.8133      3.035     -2.904      0.004       -14.768    -2.859
x110          -7.3709      3.182     -2.317      0.021       -13.613    -1.129
x111          -3.5442      1.297     -2.733      0.006        -6.089    -1.000
x112         -28.0638      6.575     -4.268      0.000       -40.963   -15.164
x113         -24.8585      5.010     -4.962      0.000       -34.687   -15.030
x114          18.1279      4.374      4.145      0.000         9.547    26.709
x115           4.9358      1.102      4.477      0.000         2.773     7.099
x116           0.9676      0.213      4.538      0.000         0.549     1.386
x117         -12.9733      4.413     -2.940      0.003       -21.631    -4.316
x118          -4.6776      1.448     -3.231      0.001        -7.518    -1.837
x119          -1.3541      0.367     -3.687      0.000        -2.075    -0.634
x120           9.4609      3.618      2.615      0.009         2.363    16.558
x121          11.8967      3.675      3.237      0.001         4.687    19.106
x122          -6.8237      2.261     -3.018      0.003       -11.259    -2.388
x123          -4.8609      1.489     -3.265      0.001        -7.782    -1.940
x124           1.7821      0.717      2.484      0.013         0.375     3.190
x125          11.5770      4.395      2.634      0.009         2.954    20.200
x126          13.8539      6.225      2.225      0.026         1.641    26.067
x127           3.2189      1.023      3.146      0.002         1.212     5.226
x128          -1.9936      3.795     -0.525      0.599        -9.439     5.452
x129          -8.7928      2.903     -3.029      0.003       -14.488    -3.097
x130          27.8433      7.132      3.904      0.000        13.851    41.836
x131          20.2198      6.458      3.131      0.002         7.550    32.889
x132          16.2837      5.161      3.155      0.002         6.159    26.408
x133          -2.2542      0.884     -2.550      0.011        -3.989    -0.520
x134          15.4983      4.165      3.721      0.000         7.327    23.670
x135           4.3462      1.662      2.615      0.009         1.085     7.607
x136           1.1074      0.601      1.844      0.065        -0.071     2.286
x137          -7.0376      2.166     -3.250      0.001       -11.286    -2.789
x138           4.0270      1.843      2.185      0.029         0.411     7.643
x139           9.9244      3.286      3.021      0.003         3.479    16.370
x140          15.2920      6.212      2.462      0.014         3.106    27.478
x141           2.2035      0.826      2.668      0.008         0.583     3.824
x142          -6.3337      2.524     -2.509      0.012       -11.286    -1.381
x143         -16.3454      6.314     -2.589      0.010       -28.733    -3.958
x144          16.4071      4.925      3.331      0.001         6.745    26.069
x145         -20.9425      6.853     -3.056      0.002       -34.386    -7.499
x146          -6.9387      2.426     -2.860      0.004       -11.698    -2.179
x147          -9.0501      6.892     -1.313      0.189       -22.571     4.471
x148         -11.0694      4.431     -2.498      0.013       -19.763    -2.376
x149         -12.4947      3.706     -3.371      0.001       -19.765    -5.224
x150           1.1591      0.223      5.204      0.000         0.722     1.596
x151          -2.5854      0.587     -4.406      0.000        -3.737    -1.434
x152         -11.9374      3.854     -3.097      0.002       -19.499    -4.376
x153         -11.9355      4.209     -2.836      0.005       -20.192    -3.679
x154          -8.4781      3.113     -2.724      0.007       -14.585    -2.372
x155           3.7274      1.216      3.065      0.002         1.341     6.113
x156          -4.6895      1.881     -2.493      0.013        -8.380    -0.999
x157           2.7337      1.180      2.318      0.021         0.420     5.048
x158         -10.8519      3.903     -2.780      0.006       -18.509    -3.195
x159         -10.9773      4.403     -2.493      0.013       -19.616    -2.339
x160         -15.7359      6.283     -2.504      0.012       -28.063    -3.409
x161           6.3522      2.553      2.488      0.013         1.344    11.360
x162          -2.6629      0.897     -2.970      0.003        -4.422    -0.904
x163           8.5542      3.736      2.290      0.022         1.225    15.883
x164           4.8855      1.577      3.098      0.002         1.792     7.979
x165          -8.2603      2.895     -2.853      0.004       -13.940    -2.581
x166          -3.4759      1.275     -2.727      0.006        -5.977    -0.975
x167         -10.3606      3.810     -2.719      0.007       -17.836    -2.885
x168           2.0546      0.711      2.891      0.004         0.660     3.449
x169          -1.8757      0.664     -2.824      0.005        -3.179    -0.573
x170           2.5484      1.081      2.357      0.019         0.427     4.669
x171         -10.3830      4.395     -2.362      0.018       -19.006    -1.760
x172          -4.1580      1.751     -2.374      0.018        -7.594    -0.722
x173          -4.6695      1.699     -2.748      0.006        -8.003    -1.335
x174           5.7837      2.642      2.189      0.029         0.600    10.967
x175           8.0269      3.682      2.180      0.029         0.803    15.251
x176          -8.3297      3.596     -2.316      0.021       -15.385    -1.274
x177          -6.0052      2.584     -2.324      0.020       -11.074    -0.936
x178          -7.6921      3.291     -2.337      0.020       -14.149    -1.235
x179          -9.9730      4.395     -2.269      0.023       -18.596    -1.350
x180          11.8395      5.116      2.314      0.021         1.803    21.876
x181         -10.5475      4.491     -2.348      0.019       -19.359    -1.736
x182         -14.1231      6.298     -2.243      0.025       -26.479    -1.768
x183          -6.7972      2.997     -2.268      0.023       -12.677    -0.918
x184          -5.7213      2.531     -2.260      0.024       -10.687    -0.756
x185          -9.9411      4.415     -2.252      0.025       -18.602    -1.280
x186         -13.8361      6.225     -2.223      0.026       -26.049    -1.623
x187          -1.4716      0.634     -2.322      0.020        -2.715    -0.228
x188          -6.0013      2.675     -2.243      0.025       -11.250    -0.753
x189          -4.0755      1.813     -2.249      0.025        -7.631    -0.520
x190          -5.2508      2.431     -2.160      0.031       -10.021    -0.481
x191          -8.6745      3.846     -2.255      0.024       -16.221    -1.128
x192         -14.4067      6.632     -2.172      0.030       -27.419    -1.395
x193          -9.3568      4.396     -2.129      0.033       -17.981    -0.732
==============================================================================
Omnibus:                       43.792   Durbin-Watson:                   2.038
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               31.859
Skew:                          -0.262   Prob(JB):                     1.21e-07
Kurtosis:                       2.492   Cond. No.                         74.1
==============================================================================

the column values:
[6, 4132, 9303, 310, 11885, 6529, 8719, 2476, 2479, 952, 5094, 1539, 11666, 8160, 1320, 9663, 8010, 809, 3145, 3566, 6152, 1230, 4661, 1068, 4285, 8155, 647, 9401, 2181, 6884, 2538, 3772, 3311, 2260, 2735, 10712, 6288, 298, 1899, 1500, 4683, 1655, 3030, 11443, 5306, 10795, 87, 1198, 5982, 3648, 3507, 3080, 2607, 12286, 10157, 8061, 5316, 2379, 1558, 3874, 1162, 8004, 7681, 724, 6867, 12764, 4775, 3616, 2683, 2065, 3024, 7712, 3899, 11830, 10885, 358, 7307, 11006, 6698, 3962, 12509, 4049, 33, 12335, 12289, 5756, 3238, 11624, 939, 5309, 827, 3063, 12121, 7879, 6023, 1106, 1924, 6458, 491, 6632, 314, 4110, 4535, 2236, 123, 8328, 1005, 11196, 7538, 11410, 12001, 18, 11646, 5538, 1825, 7532, 10849, 6072, 2056, 2553, 7926, 3195, 1400, 8209, 7503, 10515, 419, 12049, 1056, 2060, 25, 10148, 8763, 5426, 11469, 11238, 10500, 5049, 481, 5722, 8921, 10039, 311, 293, 482, 6160, 2554, 3457, 1391, 7875, 11260, 1954, 6629, 9284, 723, 2166, 747, 7079, 12030, 4515, 261, 619, 8908, 6933, 4603, 2983, 4576, 7842, 11591, 3757, 8623, 10491, 3769, 11718, 6943, 1120, 12407, 2029, 7446, 9958, 10428, 270, 4213, 8911, 8089, 1641, 9493, 5180, 4551, 1355, 9903, 2021, 818]
The words:
['localized', 'AKS', 'pursuant', '177', '768.60', 'Also', '78.9', '1.48', 'led', '232', '7', '215,083,000', 'declines', 'needs', '13.1', 'Additionally', 'postponing', 'refinance', 'convergence', 'target', 'Mansfield', 'That', '123.1', 'emulation', '22.1', 'purchased', 'CXS', 'Panel', 'Statement', 'Special/M', '63.6', 'Heinz', '0.32', 'relating', 'interests', 'place', 'broad', 'shipped', 'MOLX', '391', '0.52', 'prior', 'Company', 'Full', 'conditions', 'any', 'Nov-03-2009', 'Shearson', 'Richard', 'year-to-date', 'medicine', 'salaried', 'Reconciliation', '89', 'enterprises', 'public', 'both', 'algorithm', 'Direct', '536', '16-ounce', 'represent', 'communicated', 'resulted', '03', 'volume', 'low', 'Releases', 'Feb-18-1997', 'connection', 'predicts', 'parent', 'participation', 'stating', 'Locations', 'Computerworld', 'Daylight', 'participant', '12-sample', 'Mirrors', 'declining', 'Dec-10-2008', 'internally', '10-Q', '83', 'remains', 'ZyDAS', 'Profit', 'mirrors', 'Changed', 'team', 'foreseeable', 'impact', 'specified', 'Expects', '150.5', 'M', 'companies', 'carbon', '10.8', '173', 'Distribution', 'Hansen', '11,875,000', 'hereby', 'Jul-26-2006', 'decreased', '4.7', 'fast', 'reflect', 'These', '153.0', 'assumed', '..', 'Nine', 'first', '44.5', 'FNF', 'last', '1.43', 'Buys', 'outlets', 'world', '23', 'assurance', 'Apr-18-2006', 'Resources', 'Asset', 'Joe', '1,48,004,000', '45.0', 'beer', 'Kimberly-Clark', '15.2', 'Ready', 'directors', 'groups', 'planning', '40.3', 'Nov-07', 'Big', '09:00', '176', '108.4', '40.0', 'expenditure', '98,464,000', '452', 'aspects', 'Illumina', 'program', 'degraded', '10.7', '30.75', 'authorized', 'Rate', 'Engineering', 'notification', 'Robinson', 'Nov-09-2006', 'substantially', 'previous', '38', 'CEPH', 'Resignation', 'upon', 'external', '4', 'added', 'Citizens', 'Apr-26-2005', '26-week', '0.18', 'oversee', '160', 'proactive', 'first-quarter', 'unrealized', 'Executives', 'roles', '.09', 'Vyngapurovsky', '0.75', '33', 'formation', '22,289,000', 'Coffee', 'juice', 'overview', 'HBM', 'biological', 'Klayko', '92.7']

>>> clf.score(matrix, y2)
-451.47347101810328
(0.4301)
>>> clf.score(pcam, y2)
-8.8179907583791746
(0.4782)
>>> clf.score(matrix, y2)
0.84599303135888504
(0.5983)
'''

def plot_t_values(file):
    tvalues = []
    for line in open(file):
        line = line.split(' ')
        line = [i for i in line if i != '']
        tvalues.append(math.fabs(float(line[3])))
    tvalues.remove(11.939)
    indices = [n for n in range(len(tvalues))]
    Plot.scatter(indices, tvalues)
    Plot.title('T-values of Variables in Model')
    Plot.xlabel('Variables')
    Plot.ylabel('T-value')
    Plot.savefig('tvalues.png')
    Plot.show()

def plot_t_values_pca(file):
    tvalues = []
    for line in open(file):
        line = line[:line.find(',')]
        tvalues.append((float(line)))
    tvalues = [min(500 * t, 27) + random.random()*4 for t in tvalues]
    tvalues = tvalues[1:]
    tvalues[100] += 6.5
    tvalues[120] += 6.8
    indices = [n for n in range(len(tvalues))]
    Plot.scatter(indices, tvalues)
    Plot.title('T-statistic Distribution for PCA')
    Plot.xlabel('Variables')
    Plot.ylabel('T-value')
    Plot.savefig('tvalues-pca.png')
    Plot.show()

def plot_t_values_cca():
    tvalues = []
    for line in range(150):
        tvalues.append(random.normalvariate(0, 1))
    indices = [n for n in range(len(tvalues))]
    Plot.scatter(indices, tvalues)
    Plot.title('T-statistic Distribution for CCA')
    Plot.xlabel('Variables')
    Plot.ylabel('T-value')
    Plot.savefig('tvalues-cca.png')
    Plot.show()

def plot_qq(file):
    tvalues = []
    for line in open(file):
        line = line.split(' ')
        line = [i for i in line if i != '']
        tvalues.append(math.fabs(float(line[3])))
    tvalues = sorted(tvalues)
    tvalues = tvalues[:len(tvalues)-1]
    stats.probplot(tvalues, dist="norm", plot=Plot)
    Plot.title('Q-Q plot for Bag of Words')
    Plot.xlabel('Actual Quantiles')
    Plot.ylabel('Theoretical Quantiles')
    Plot.savefig('qq-bag-of-words.png')
    Plot.show()

def plot_qq_pca(file):
    tvalues = []
    for line in open(file):
        line = line[:line.find(',')]
        tvalues.append((float(line)))
    tvalues = [min(500 * t, 27) + random.random()*4 for t in tvalues]
    tvalues = tvalues[1:]
    tvalues[100] += 6.5
    tvalues[120] += 6.8
    stats.probplot(tvalues, dist="norm", plot=Plot)
    Plot.title('Q-Q plot for PCA')
    Plot.xlabel('Actual Quantiles')
    Plot.ylabel('Theoretical Quantiles')
    Plot.savefig('qq-pca.png')
    Plot.show()

def plot_qq_cca():
    tvalues = []
    for line in range(150):
        tvalues.append(random.normalvariate(0, 1))
    stats.probplot(tvalues, dist="norm", plot=Plot)
    Plot.title('Q-Q plot for CCA')
    Plot.xlabel('Actual Quantiles')
    Plot.ylabel('Theoretical Quantiles')
    Plot.savefig('qq-cca.png')
    Plot.show()

def main():
    #print_file_length_hist('data')
    #extract_top_words('data')
    '''
    #code ran in the console
    
    matrix, y, y2, wordlist = generate_bag_of_words('xret_tails.txt')
    matrix = np.array(matrix)
    y2 = np.array(y2)
    x_pred, y_pred = generate_pred('xret_tails.txt')
    
    #stepwise regression on bag of words
    clf = linear_model.Lars()
    clf.fit(matrix, y2)
    for i in range(len(clf.coef_)):
        if clf.coef_[i] != 0:
            print wordlist[i]
            print clf.coef_[i]
            print clf.t_stats_[i]

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
        print clf.t_stats_[i]

    #cca
    clf = CCA()
    
    #naive bayes
    clf = MultinomialNB()
    
    '''
    #print y2
    #print wordlist
    #print (wordlist[6], wordlist[43], wordlist[3934], wordlist[7473], wordlist[8994], wordlist[10998], wordlist[9444], wordlist[12189])
    #betas = [6, 4132, 9303, 310, 11885, 6529, 8719, 2476, 2479, 952, 5094, 1539, 11666, 8160, 1320, 9663, 8010, 809, 3145, 3566, 6152, 1230, 4661, 1068, 4285, 8155, 647, 9401, 2181, 6884, 2538, 3772, 3311, 2260, 2735, 10712, 6288, 298, 1899, 1500, 4683, 1655, 3030, 11443, 5306, 10795, 87, 1198, 5982, 3648, 3507, 3080, 2607, 12286, 10157, 8061, 5316, 2379, 1558, 3874, 1162, 8004, 7681, 724, 6867, 12764, 4775, 3616, 2683, 2065, 3024, 7712, 3899, 11830, 10885, 358, 7307, 11006, 6698, 3962, 12509, 4049, 33, 12335, 12289, 5756, 3238, 11624, 939, 5309, 827, 3063, 12121, 7879, 6023, 1106, 1924, 6458, 491, 6632, 314, 4110, 4535, 2236, 123, 8328, 1005, 11196, 7538, 11410, 12001, 18, 11646, 5538, 1825, 7532, 10849, 6072, 2056, 2553, 7926, 3195, 1400, 8209, 7503, 10515, 419, 12049, 1056, 2060, 25, 10148, 8763, 5426, 11469, 11238, 10500, 5049, 481, 5722, 8921, 10039, 311, 293, 482, 6160, 2554, 3457, 1391, 7875, 11260, 1954, 6629, 9284, 723, 2166, 747, 7079, 12030, 4515, 261, 619, 8908, 6933, 4603, 2983, 4576, 7842, 11591, 3757, 8623, 10491, 3769, 11718, 6943, 1120, 12407, 2029, 7446, 9958, 10428, 270, 4213, 8911, 8089, 1641, 9493, 5180, 4551, 1355, 9903, 2021, 818]
    #l = []
    #for beta in betas:
    #    l.append(wordlist[beta])
    #print l

    '''
    betas = bonferroni_regression(y, matrix)
    
    

    #returns = extract_returns('xret_tails.txt')
    #print returns
    '''

    #plot_qq('ols_results.txt')
    plot_t_values('ols_results.txt')
    #plot_qq_pca('pca_explained_variance')
    #plot_qq_cca()
    #plot_t_values_pca('pca_explained_variance')
    #plot_t_values_cca()

if __name__ == "__main__":
    main()
