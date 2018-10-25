#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

Cluster doc class
Created on Wed Oct 24 10:22:02 2018

@author: brialy
"""
import os
# changing path to load dependencies, then switching back
cwd = os.getcwd()
os.chdir("/Users/brialy/Documents/python code/")
import pandas as pd
import numpy as np
import glob
import nltk
import re
import operator
from mpl_toolkits.mplot3d import Axes3D
from sklearn.manifold import TSNE
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from ExtStops import *
# switching back to original directory 
os.chdir(cwd)


class PreProcess(object):
    """
    contains the methods related to the preprocessing step of document clustering.
    """
    def __init__(self, target, k = 1):
        self.normalize_documents = self.__normalize_documents(target)
        self.lsa_transform = self.__lsa_transform(self.normalize_documents, k)
    
    def __normalize_documents(self, target):
        """
        This function removes stopwords and performs a TF-IDF transformation on the data set.
        """
        stop_words = set(stopwords.words('english'))
        stop_words.update(extStop)
        tv = TfidfVectorizer(use_idf=True, ngram_range=(1,2), stop_words= stop_words)
        return  (tv.fit_transform(target)).toarray()

    def __lsa_transform(self, target, k):
        """
        This function performs a truncated SVD, which on text data is Lantent semantic analysis (LSA)
        a technique which groups and extracts sematic topics in documents togther.
        """  
        lsa = TruncatedSVD(n_components= k, n_iter=100)
        return lsa.fit_transform(target)


class HyperTuneMethods(object):
    """
    contains the methods related to hyper tunning values for document clustering
    """
    def __init__(self, target, k_min = 2, k_max = 10, method = 'sil'):
        if method == 'sil':
            self.sil = self.__silHyperTune(target, k_min, k_max)
        
        elif method == 'elbow':
            self.elbow = self.__elbowHyperTune(target, k_min, k_max)
            plt.figure()
            plt.plot(list(self.elbow.keys()), list(self.elbow.values()))
            plt.xlabel("Number of cluster")
            plt.ylabel("SSE")
            plt.show()
        
        else:
            assert False, "You have not selected a supported method"
        
    def __silHyperTune(self, target, k_min, k_max):
        """
        This function runs a lsa and kmeans, k_min to k_max times and stores the silhouette scores
        in order to determine the optimal number of clusters needed.
        """
        scores ={}
        for k_test in range(k_min, (k_max +1)):
            tune = PreProcess(target, k_test).lsa_transform
            kmeans = KMeans(n_clusters=k_test).fit(tune)
            label = kmeans.labels_
            sil_coeff = silhouette_score(tune, label, metric='euclidean')
            scores.update({k_test : sil_coeff})    
        return scores
    
    def __elbowHyperTune(self, target, k_min, k_max):
        """
        This function charts the sse values from k_min to k_max to visualise the 'elbow'
        in order to determine the optimal number of clusters needed.
        """
        scores = {}
        for k_test in range(target, k_min, (k_max +1)):
            tune = PreProcess(target, k_test).lsa_transform
            kmeans = KMeans(n_clusters=k_testk, max_iter=1000).fit(tune)
            scores[k] = kmeans.inertia_ 
            return scores
          
        
class DocumentCluster(object):
    """
    Contains the methods relating to document clustering and evaluating results
    """
    def __init__(self, target, raw, k, method = 'kmeans'):
        if method == 'kmeans':          
            self.result = self.__kmeansCluster(target, raw, k)
        
        elif method == 'agglomerative':
            #TODO
            #create a hierarchical clustering class method 
            pass
        
        else:
            assert False, "You have not selected a supported method"
        
            
    def __kmeansCluster(self, target, raw, k):
        """
        This function implments Kmeans to cluster the processed data
        """
        km = KMeans(n_clusters= k, random_state=0, max_iter= 1000)
        data = PreProcess(target, k).lsa_transform
        km.fit_transform(data)
        cluster_labels = pd.DataFrame(km.labels_, columns=['ClusterLabel'])
        dataframe = pd.DataFrame(raw)
        result = pd.concat([dataframe, cluster_labels], axis = 1 )
        return result
    
    def count_results(self):
        print(pd.DataFrame(Counter(self.result.ClusterLabel).most_common(),columns= ('Cluster No.', 'Count'))) 


# example of usage        
#k_num = max(HyperTuneMethods(corpus, 2,20).sil)
#output = DocumentCluster(corpus, data, k_num)
#output.count_results()
#results = output.result





















    
    
