from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import jieba
import pandas as pd

article = pd.read_excel('Cut_Finish_jieba.xlsx')

sentences = article['內容'].tolist()

split_sentences = []

for i in sentences:
    split_sentences.append(i.split(' '))



# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(wordss)]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(split_sentences)]
print(documents)
print("stage4")
model = Doc2Vec(documents, size=500, window=5, min_count=5, workers=4, epoch=5000)
model.save("doc2vec_stock.model")
print("over")

model = Doc2Vec.load("doc2vec_stock.model")

print(model.most_similar("台積電", topn=5))
print(model.most_similar("鴻海", topn=5))
print(model.most_similar("中華電信", topn=5))
print(model.most_similar("仁寶", topn=5))
print(model.most_similar("兆豐金", topn=5))

