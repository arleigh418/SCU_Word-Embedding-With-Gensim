import gensim
import pandas as pd
import os
from gensim.models.doc2vec import Doc2Vec
import numpy as np
TaggededDocument = gensim.models.doc2vec.TaggedDocument
 


model = Doc2Vec.load('your.model')
x = pd.read_excel('Cut_Finish_jieba.xlsx',encoding = 'utf-8')


def cos_sim(vector1, vector2):  
    cos=np.dot(vector1,vector2)/(np.linalg.norm(vector1)*(np.linalg.norm(vector2)))
    return cos

def count_vector(model, words): 
    vect_list = []
    vec = np.zeros(500) 
    for w in words:  
        try:    
           vec = vec + model.wv[w]
        except:
            continue
    return vec


for i in x:
 i = i.split(' ')
 vec = count_vector(model,i)
 cos = cos_sim(vec,model.wv['鴻海'])
 print(cos)


