import gensim
import jieba
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec
import re 
import numpy as np

import xlwt
jieba.load_userdict('new_words.txt')
TaggededDocument = gensim.models.doc2vec.TaggedDocument
 


model = Doc2Vec.load('test.model')



def cos_sim(vector_a, vector_b):  
    vector_a = np.mat(vector_a)
    vector_b = np.mat(vector_b)
    num = vector_a * vector_b.T
    denom = np.linalg.norm(vector_a) * np.linalg.norm(vector_b)
    cos = num / denom
   
    return cos



def sent2vec(model, words): 
    vect_list = []
    for w in words:
        try:
            vect_list.append(model.wv[w])
        except:
            x = np.zeros(300)
            vect_list.append(x)
            continue
    vect_list = np.array(vect_list)
    vect = vect_list.sum(axis=0)
    return vect / np.sqrt((vect ** 2).sum())






