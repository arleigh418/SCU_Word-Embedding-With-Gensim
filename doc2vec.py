from gensim.models.doc2vec import Doc2Vec,TaggedDocument
import jieba
import pandas as pd

f1 = pd.read_excel('your_file.xlsx.xlsx')

word = f1['內容']

words = []
for i in word:
   words.append(i)
words = str(words)
print("stage1")
words = jieba.cut(words)
print("stage2")
word_store = []
for x in words:
   word_store.append(x)
# wordss = str(word_store)
print("stage3")
# wordss = word_store.split()
word_store = [word_store]

# documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(wordss)]
documents = [TaggedDocument(doc, [i]) for i, doc in enumerate(word_store)]
# documents
print("stage4")
model = Doc2Vec(documents, size=300, window=5, min_count=5, workers=4, epoch=100)
model.save("doc2vec_index_asia.model")
print("over")

# model = Doc2Vec.load("doc2vec_index_asia.model")
# print(model.wv['藍'])
print(model.wv['籃球'])








# print(model.wv['電影'])
# print(model.wv['劇情'])
# print(model.wv['導演'])
# model.most_similar("電影", topn=5)
# model.most_similar("劇情", topn=5)
# model.most_similar("導演", topn=5)

# article = "動作片 電影 劇情 導演 美國"
# art123 = article.split()
# # art_art = [art_art]
# art3 = model.infer_vector(doc_words = art123,alpha=0.025,steps=10)
# print(art3)
