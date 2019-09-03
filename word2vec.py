import pandas as pd
from gensim.models import Word2Vec
import jieba




article = pd.read_excel('Cut_Finish_jieba.xlsx')

sentences = article['內容'].tolist()

split_sentences = []

for i in sentences:
    split_sentences.append(i.split(' '))

# build a Word2Vce model
model = Word2Vec(split_sentences, size=500, window=10, min_count=5, workers=4)

model.save("word2vec_stock.model")



print(model.most_similar("台積電", topn=5))
print(model.most_similar("鴻海", topn=5))
print(model.most_similar("中華電信", topn=5))
print(model.most_similar("仁寶", topn=5))
print(model.most_similar("兆豐金", topn=5))
