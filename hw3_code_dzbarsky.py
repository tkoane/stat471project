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
#from sklearn import linear_model

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
    print a[:10]
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

    return allBagsofWords, returnsVector, list(allWords)
    #print "Returns: " + str(returnsVector)
    #print "Matrix: " + str(allBagsofWords)

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
                sig = results.rsquared_adj
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
Dep. Variable:                      y   R-squared:                       0.572
Model:                            OLS   Adj. R-squared:                  0.519
Method:                 Least Squares   F-statistic:                     10.81
Date:                Mon, 25 Nov 2013   Prob (F-statistic):          2.87e-149
Time:                        14:23:45   Log-Likelihood:                -4648.6
No. Observations:                1435   AIC:                             9615.
Df Residuals:                    1276   BIC:                         1.045e+04
Df Model:                         158                                         
==============================================================================
                 coef    std err          t      P>|t|      [95.0% Conf. Int.]
------------------------------------------------------------------------------
const          1.0221      0.239      4.283      0.000         0.554     1.490
x1            80.5606      6.748     11.939      0.000        67.322    93.799
x2            -7.2186      1.520     -4.750      0.000       -10.200    -4.237
x3            -7.0833      1.634     -4.336      0.000       -10.288    -3.878
x4            31.2399      6.577      4.750      0.000        18.338    44.142
x5            16.4390      3.276      5.017      0.000        10.011    22.867
x6           -24.7693      3.299     -7.509      0.000       -31.241   -18.298
x7            50.9578      9.276      5.493      0.000        32.760    69.156
x8           -32.6508      6.578     -4.963      0.000       -45.557   -19.745
x9            -8.4747      1.798     -4.714      0.000       -12.002    -4.947
x10          -28.6597      4.749     -6.035      0.000       -37.976   -19.343
x11            5.7516      0.977      5.886      0.000         3.835     7.669
x12          -37.8258      6.924     -5.463      0.000       -51.409   -24.242
x13           14.7658      3.370      4.382      0.000         8.155    21.376
x14            9.5212      2.182      4.363      0.000         5.240    13.802
x15           13.0910      4.104      3.190      0.001         5.040    21.142
x16          -10.6104      1.975     -5.374      0.000       -14.484    -6.737
x17          -27.4021      6.553     -4.182      0.000       -40.257   -14.547
x18          -26.7221      6.553     -4.078      0.000       -39.577   -13.867
x19          -14.3126      3.951     -3.622      0.000       -22.064    -6.561
x20           -4.4110      1.058     -4.170      0.000        -6.486    -2.336
x21           -8.7617      2.497     -3.509      0.000       -13.660    -3.863
x22          -11.3872      2.966     -3.839      0.000       -17.206    -5.569
x23          -12.4815      2.598     -4.804      0.000       -17.579    -7.384
x24          -11.4848      2.507     -4.581      0.000       -16.403    -6.566
x25           21.4951      6.650      3.232      0.001         8.449    34.541
x26            7.2197      1.992      3.625      0.000         3.312    11.127
x27            8.1160      2.184      3.716      0.000         3.831    12.401
x28           10.9481      2.278      4.806      0.000         6.479    15.417
x29          -13.5250      2.900     -4.665      0.000       -19.213    -7.837
x30          -16.3421      4.637     -3.525      0.000       -25.438    -7.246
x31          -12.1052      3.837     -3.155      0.002       -19.633    -4.578
x32           -3.5296      0.831     -4.246      0.000        -5.160    -1.899
x33          -13.2962      2.412     -5.513      0.000       -18.028    -8.565
x34           10.2156      2.639      3.870      0.000         5.037    15.394
x35           -6.1021      1.532     -3.982      0.000        -9.108    -3.096
x36           -8.2751      2.293     -3.608      0.000       -12.774    -3.776
x37            7.7272      2.565      3.012      0.003         2.695    12.760
x38          -14.6837      4.650     -3.158      0.002       -23.806    -5.561
x39           -6.4584      2.117     -3.050      0.002       -10.612    -2.304
x40          -24.3808      6.578     -3.706      0.000       -37.287   -11.475
x41           -6.7569      2.412     -2.802      0.005       -11.488    -2.025
x42           -1.7322      0.535     -3.236      0.001        -2.782    -0.682
x43            0.9740      0.298      3.266      0.001         0.389     1.559
x44            2.2287      0.620      3.596      0.000         1.013     3.445
x45           -5.2264      1.200     -4.355      0.000        -7.581    -2.872
x46            2.6144      0.722      3.619      0.000         1.197     4.032
x47           15.1779      4.637      3.274      0.001         6.082    24.274
x48           40.0473      7.933      5.048      0.000        24.485    55.610
x49           -7.4893      2.277     -3.290      0.001       -11.956    -3.023
x50          -19.9037      5.105     -3.899      0.000       -29.919    -9.888
x51           17.2827      6.612      2.614      0.009         4.312    30.253
x52           -6.1366      1.990     -3.084      0.002       -10.040    -2.233
x53          -32.8278      6.833     -4.805      0.000       -46.232   -19.423
x54          -12.6804      3.496     -3.627      0.000       -19.539    -5.822
x55           -9.4259      2.355     -4.003      0.000       -14.046    -4.806
x56           -9.3238      1.960     -4.757      0.000       -13.169    -5.478
x57            4.0007      1.021      3.919      0.000         1.998     6.003
x58           14.4501      3.996      3.617      0.000         6.611    22.289
x59            6.5912      2.073      3.180      0.002         2.524    10.658
x60          -16.9584      4.746     -3.573      0.000       -26.270    -7.647
x61          -25.0141      6.704     -3.731      0.000       -38.166   -11.862
x62          -13.4117      3.523     -3.807      0.000       -20.324    -6.500
x63           13.6806      3.205      4.269      0.000         7.393    19.968
x64            7.1759      2.395      2.996      0.003         2.477    11.874
x65            3.9597      1.292      3.066      0.002         1.426     6.494
x66            7.4778      2.162      3.459      0.001         3.237    11.719
x67           -7.9770      2.320     -3.439      0.001       -12.528    -3.426
x68          -14.3444      4.637     -3.094      0.002       -23.441    -5.248
x69           -6.8610      2.318     -2.960      0.003       -11.409    -2.313
x70            5.9126      1.925      3.071      0.002         2.136     9.689
x71          -19.2821      6.553     -2.943      0.003       -32.137    -6.427
x72          -13.8314      4.638     -2.982      0.003       -22.930    -4.733
x73            8.5153      2.934      2.902      0.004         2.759    14.272
x74          -16.1268      4.679     -3.447      0.001       -25.306    -6.947
x75          -10.8987      3.788     -2.877      0.004       -18.330    -3.467
x76           -9.4460      3.276     -2.883      0.004       -15.874    -3.018
x77            8.3742      2.313      3.621      0.000         3.837    12.912
x78          -15.3149      3.938     -3.889      0.000       -23.040    -7.590
x79           10.4237      4.974      2.096      0.036         0.666    20.181
x80           17.7124      4.459      3.972      0.000         8.965    26.460
x81          -17.0664      6.562     -2.601      0.009       -29.941    -4.192
x82          -18.2121      6.553     -2.779      0.006       -31.067    -5.357
x83          -21.7313      6.615     -3.285      0.001       -34.708    -8.755
x84           -6.7387      2.323     -2.900      0.004       -11.297    -2.180
x85          -18.5699      6.574     -2.825      0.005       -31.466    -5.674
x86          -10.6157      3.023     -3.511      0.000       -16.547    -4.685
x87            5.9047      2.470      2.391      0.017         1.059    10.750
x88          -13.6686      4.679     -2.921      0.004       -22.849    -4.488
x89           -5.0178      1.700     -2.952      0.003        -8.352    -1.683
x90           18.2779      6.553      2.789      0.005         5.423    31.133
x91            3.4686      1.436      2.416      0.016         0.652     6.286
x92           15.8531      6.667      2.378      0.018         2.773    28.933
x93            3.5798      1.213      2.951      0.003         1.200     5.959
x94           -9.2405      2.467     -3.745      0.000       -14.081    -4.400
x95           -6.3857      3.518     -1.815      0.070       -13.287     0.516
x96           17.8579      6.553      2.725      0.007         5.003    30.713
x97          -12.4204      4.934     -2.518      0.012       -22.099    -2.741
x98            5.2150      1.092      4.777      0.000         3.073     7.357
x99           -9.3160      2.779     -3.353      0.001       -14.767    -3.865
x100           6.3770      3.466      1.840      0.066        -0.422    13.176
x101         -11.3853      3.798     -2.998      0.003       -18.836    -3.935
x102         -10.2125      2.789     -3.661      0.000       -15.685    -4.740
x103           1.1574      0.469      2.466      0.014         0.237     2.078
x104         -14.5063      6.654     -2.180      0.029       -27.561    -1.452
x105         -16.9421      6.553     -2.586      0.010       -29.797    -4.087
x106         -16.9021      6.553     -2.579      0.010       -29.757    -4.047
x107          -5.3525      1.383     -3.871      0.000        -8.065    -2.640
x108          -9.5990      3.524     -2.724      0.007       -16.513    -2.685
x109          -8.5134      3.200     -2.661      0.008       -14.791    -2.236
x110          -9.8312      3.248     -3.027      0.003       -16.203    -3.459
x111          -3.5282      1.362     -2.590      0.010        -6.201    -0.855
x112         -22.6501      6.769     -3.346      0.001       -35.931    -9.370
x113         -23.2825      5.270     -4.418      0.000       -33.621   -12.944
x114          17.3472      4.587      3.781      0.000         8.347    26.347
x115           4.7247      1.155      4.091      0.000         2.459     6.991
x116           0.7646      0.218      3.512      0.000         0.338     1.192
x117         -12.2613      4.653     -2.635      0.009       -21.390    -3.132
x118          -4.1780      1.525     -2.740      0.006        -7.169    -1.187
x119          -1.3057      0.385     -3.393      0.001        -2.061    -0.551
x120           9.2416      3.804      2.429      0.015         1.778    16.705
x121          12.1174      3.876      3.126      0.002         4.513    19.721
x122          -6.4616      2.380     -2.714      0.007       -11.132    -1.792
x123          -4.4354      1.550     -2.861      0.004        -7.477    -1.394
x124           1.6980      0.744      2.282      0.023         0.238     3.158
x125          11.8829      4.637      2.563      0.010         2.787    20.979
x126          14.5753      6.566      2.220      0.027         1.694    27.457
x127           3.5076      1.069      3.283      0.001         1.411     5.604
x128          -9.1528      3.329     -2.749      0.006       -15.685    -2.621
x129          -8.6524      3.058     -2.829      0.005       -14.652    -2.653
x130          16.1692      6.578      2.458      0.014         3.263    29.075
x131          15.9992      6.578      2.432      0.015         3.093    28.905
x132          12.0816      5.063      2.386      0.017         2.150    22.013
x133          -2.1068      0.932     -2.260      0.024        -3.935    -0.278
x134          10.7012      4.188      2.555      0.011         2.486    18.917
x135           4.3434      1.753      2.478      0.013         0.904     7.783
x136           1.1874      0.625      1.899      0.058        -0.039     2.414
x137          -6.4123      2.281     -2.811      0.005       -10.888    -1.937
x138           5.1071      1.893      2.697      0.007         1.393     8.821
x139           9.6485      3.461      2.788      0.005         2.859    16.438
x140          15.5979      6.553      2.380      0.017         2.743    28.453
x141           2.3661      0.866      2.733      0.006         0.668     4.065
x142          -6.0019      2.651     -2.264      0.024       -11.203    -0.800
x143         -15.6855      6.659     -2.356      0.019       -28.749    -2.622
x144          16.4655      5.193      3.171      0.002         6.278    26.653
x145         -19.5602      7.215     -2.711      0.007       -33.716    -5.405
x146          -5.9240      2.550     -2.324      0.020       -10.926    -0.922
x147         -15.1954      6.581     -2.309      0.021       -28.105    -2.285
x148         -10.6511      4.673     -2.279      0.023       -19.818    -1.484
x149          -9.9838      3.798     -2.629      0.009       -17.434    -2.533
x150           0.9513      0.229      4.151      0.000         0.502     1.401
x151          -2.2973      0.612     -3.751      0.000        -3.499    -1.096
x152         -10.6749      4.057     -2.631      0.009       -18.633    -2.717
x153         -11.0839      4.438     -2.498      0.013       -19.790    -2.378
x154          -8.1173      3.283     -2.472      0.014       -14.558    -1.676
x155           3.3035      1.267      2.606      0.009         0.817     5.790
x156          -4.7705      1.981     -2.408      0.016        -8.657    -0.885
x157           2.9142      1.244      2.343      0.019         0.474     5.354
x158          -9.2967      4.101     -2.267      0.024       -17.341    -1.252
==============================================================================
Omnibus:                      109.743   Durbin-Watson:                   2.017
Prob(Omnibus):                  0.000   Jarque-Bera (JB):               48.325
Skew:                          -0.244   Prob(JB):                     3.21e-11
Kurtosis:                       2.244   Cond. No.                         74.0
==============================================================================
the column values:
[6, 4132, 9303, 310, 11885, 6529, 8719, 2476, 2479, 952, 5094, 1539, 11666, 8160, 1320, 9663, 8010, 809, 3145, 3566, 6152, 1230, 4661, 1068, 4285, 8155, 647, 9401, 2181, 6884, 2538, 3772, 3311, 2260, 2735, 10712, 6288, 298, 1899, 1500, 4683, 1655, 3030, 11443, 5306, 10795, 87, 1198, 5982, 3648, 3507, 3080, 2607, 12286, 10157, 8061, 5316, 2379, 1558, 3874, 1162, 8004, 7681, 724, 6867, 12764, 4775, 3616, 2683, 2065, 3024, 7712, 3899, 11830, 10885, 358, 7307, 11006, 6698, 3962, 12509, 4049, 33, 12335, 12289, 5756, 3238, 11624, 939, 5309, 827, 3063, 12121, 7879, 6023, 1106, 1924, 6458, 491, 6632, 314, 4110, 4535, 2236, 123, 8328, 1005, 11196, 7538, 11410, 12001, 18, 11646, 5538, 1825, 7532, 10849, 6072, 2056, 2553, 7926, 3195, 1400, 8209, 7503, 10515, 419, 12049, 1056, 2060, 25, 10148, 8763, 5426, 11469, 11238, 10500, 5049, 481, 5722, 8921, 10039, 311, 293, 482, 6160, 2554, 3457, 1391, 7875, 11260, 1954, 6629, 9284, 723, 2166, 747, 7079]
The words:

['localized', 'AKS', 'pursuant', '177', '768.60', 'Also', '78.9', '1.48', 'led', '232', '7', '215,083,000', 'declines', 'needs', '13.1', 'Additionally', 'postponing', 'refinance', 'convergence', 'target', 'Mansfield', 'That', '123.1', 'emulation', '22.1', 'purchased', 'CXS', 'Panel', 'Statement', 'Special/M', '63.6', 'Heinz', '0.32', 'relating', 'interests', 'place', 'broad', 'shipped', 'MOLX', '391', '0.52', 'prior', 'Company', 'Full', 'conditions', 'any', 'Nov-03-2009', 'Shearson', 'Richard', 'year-to-date', 'medicine', 'salaried', 'Reconciliation', '89', 'enterprises', 'public', 'both', 'algorithm', 'Direct', '536', '16-ounce', 'represent', 'communicated', 'resulted', '03', 'volume', 'low', 'Releases', 'Feb-18-1997', 'connection', 'predicts', 'parent', 'participation', 'stating', 'Locations', 'Computerworld', 'Daylight', 'participant', '12-sample', 'Mirrors', 'declining', 'Dec-10-2008', 'internally', '10-Q', '83', 'remains', 'ZyDAS', 'Profit', 'mirrors', 'Changed', 'team', 'foreseeable', 'impact', 'specified', 'Expects', '150.5', 'M', 'companies', 'carbon', '10.8', '173', 'Distribution', 'Hansen', '11,875,000', 'hereby', 'Jul-26-2006', 'decreased', '4.7', 'fast', 'reflect', 'These', '153.0', 'assumed', '..', 'Nine', 'first', '44.5', 'FNF', 'last', '1.43', 'Buys', 'outlets', 'world', '23', 'assurance', 'Apr-18-2006', 'Resources', 'Asset', 'Joe', '1,48,004,000', '45.0', 'beer', 'Kimberly-Clark', '15.2', 'Ready', 'directors', 'groups', 'planning', '40.3', 'Nov-07', 'Big', '09:00', '176', '108.4', '40.0', 'expenditure', '98,464,000', '452', 'aspects', 'Illumina', 'program', 'degraded', '10.7', '30.75', 'authorized', 'Rate', 'Engineering', 'notification']


'''

def main():
    #print_file_length_hist('data')
    #extract_top_words('data')

    matrix, y, wordlist = generate_bag_of_words('xret_tails.txt')
    #print (wordlist[6], wordlist[43], wordlist[3934], wordlist[7473], wordlist[8994], wordlist[10998], wordlist[9444], wordlist[12189])
    betas = [6, 4132, 9303, 310, 11885, 6529, 8719, 2476, 2479, 952, 5094, 1539, 11666, 8160, 1320, 9663, 8010, 809, 3145, 3566, 6152, 1230, 4661, 1068, 4285, 8155, 647, 9401, 2181, 6884, 2538, 3772, 3311, 2260, 2735, 10712, 6288, 298, 1899, 1500, 4683, 1655, 3030, 11443, 5306, 10795, 87, 1198, 5982, 3648, 3507, 3080, 2607, 12286, 10157, 8061, 5316, 2379, 1558, 3874, 1162, 8004, 7681, 724, 6867, 12764, 4775, 3616, 2683, 2065, 3024, 7712, 3899, 11830, 10885, 358, 7307, 11006, 6698, 3962, 12509, 4049, 33, 12335, 12289, 5756, 3238, 11624, 939, 5309, 827, 3063, 12121, 7879, 6023, 1106, 1924, 6458, 491, 6632, 314, 4110, 4535, 2236, 123, 8328, 1005, 11196, 7538, 11410, 12001, 18, 11646, 5538, 1825, 7532, 10849, 6072, 2056, 2553, 7926, 3195, 1400, 8209, 7503, 10515, 419, 12049, 1056, 2060, 25, 10148, 8763, 5426, 11469, 11238, 10500, 5049, 481, 5722, 8921, 10039, 311, 293, 482, 6160, 2554, 3457, 1391, 7875, 11260, 1954, 6629, 9284, 723, 2166, 747, 7079]
    l = []
    for beta in betas:
        l.append(wordlist[beta])
    print l

'''
    betas = aic_regression(y, matrix)
    
    

    #returns = extract_returns('xret_tails.txt')
    #print returns
'''


if __name__ == "__main__":
    main()
