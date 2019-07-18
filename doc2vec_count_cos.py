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
 

f1 = pd.read_excel('test_all.xlsx')
word = list(f1['內容'])
r="[\s+\.\!\/_,$%^*~(+\"\')]~,-|[——()?『\』；【】“”！，。？、~@#￥%……&*（）●:[\]「」=：.!',-/' ]"

def cut_sentence(word):
   stop_list = [line[:-1] for line in open("stop.txt")]
   result = []
   for each in word:
      each = re.sub(r,'',each)
      each_cut = jieba.cut(each)
      each_split = ' '.join(each_cut).split()
      each_result = [words for words in each_split if words not in stop_list]
      result.append(' '.join(each_result))
   return result



def x_train(cut_sentence):
   x_train = []
   for i,text in enumerate(cut_sentence):
      word_list = text.split(' ')
      use = len(word_list)
      word_list[use-1] = word_list[use-1].strip()
      document = TaggededDocument(word_list,tags = [i])
      x_train.append(document)
   return x_train

def train(x_train,size = 300):
   model = Doc2Vec(x_train,min_count = 5,window = 5 ,size = size,workers = 4,sample = 1e-3)
   model.train(x_train,total_examples=model.corpus_count,epochs=1000)
   model.save("test.model")
   return(model)



b = cut_sentence(word)
print('stage1')
# c = x_train(b)
# print(c)
print('stage2')
# model = train(c)
print('stage3')
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




f2 = pd.read_excel('your_file.xlsx') #中心文章處理
word_c = list(f2['內容'])

word_c = " ".join(word_c)
# print(word_c)
word_cent =cut_sentence(word_c)
# print(word_cent)
word_center = sent2vec(model,word_cent)
# print('Prepare over')



count =0
score= []
look = []

for x in b:
   # print(x)
   word_compare = sent2vec(model,x)
   score.append(word_compare)
   sim  = cos_sim(word_compare,word_center)
   # print(sim)
#    print('===============================')
   if sim >0.7:
      score.append(1)
      count=count+1
      look.append(x)
   else:
      score.append(0)
# print(score)
print(look)
print(count)
score = np.array(score)
pd_data = pd.DataFrame(score)
pd_data.to_csv('final_mark.csv')

