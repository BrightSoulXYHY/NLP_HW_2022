import gensim

# from gensim.models import Word2Vec
from gensim.models.word2vec import LineSentence,Word2Vec


sentences = LineSentence('out/corpus_onehot_text.txt')
model = Word2Vec(sentences,vector_size=100, hs=1, min_count=1, window=5)


save_path="out/word_vec/gensim.model"
model.save(save_path)