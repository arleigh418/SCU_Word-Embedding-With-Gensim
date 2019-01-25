import pandas as pd
from gensim.models import Word2Vec
import jieba

# with open ('PPT-movie-7200-7275.txt','r',encoding  = 'utf-8') as f:
#     f1 = f.read()


f1 = pd.read_excel('your_file.xlsx')

word = f1['內容']

words = []
for i in word:
   words.append(i)
words = str(words)


sentence = list(jieba.cut(words))
# sentence


# build a Word2Vce model
model = Word2Vec([sentence], size=300, window=5, min_count=5, workers=4, iter=100)
# save model to file
# model.save("word2vec_indexasia.model")
# load model to python
model = Word2Vec.load("word2vec_indexasia.model")









# print(model.wv['籃球'])
# print(model.wv['劇情'])
# print(model.wv['導演'])

# model.most_similar("電影", topn=5)
# model.most_similar("劇情", topn=5)
# model.most_similar("導演", topn=5)

