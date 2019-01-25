

from gensim.models import FastText

with open ('your_file.xlsx.txt','r',encoding  = 'utf-8') as f:
    f1 = f.read()

sentence = list(jieba.cut(f1))
# build a Word2Vce model
model = FastText([sentence], size=50, window=5, min_count=5, workers=4, iter=100)
# save model to file
model.save("fastText.model")
# load model to python
# model = Word2Vec.load("word2vec.model")

print(model.wv['電影'])
print(model.wv['劇情'])
print(model.wv['導演'])

model.most_similar("電影", topn=5)
model.most_similar("劇情", topn=5)
model.most_similar("導演", topn=5)
