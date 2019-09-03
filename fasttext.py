from gensim.models import FastText
import pandas as pd

article = pd.read_excel('Cut_Finish_jieba.xlsx')

sentences = article['內容'].tolist()

split_sentences = []

for i in sentences:
    split_sentences.append(i.split(' '))

print('訓練開始')
# build a Word2Vce model
model = FastText(split_sentences, size=500, window=10, min_count=5, workers=4)
# save model to file
model.save("fastText_stock.model")
# load model to python
# model = Word2Vec.load("word2vec.model")



print(model.most_similar("台積電", topn=5))
print(model.most_similar("鴻海", topn=5))
print(model.most_similar("中華電信", topn=5))
print(model.most_similar("仁寶", topn=5))
print(model.most_similar("兆豐金", topn=5))
