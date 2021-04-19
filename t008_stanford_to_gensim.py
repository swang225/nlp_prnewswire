# python -m gensim.scripts.glove2word2vec --input  glove.42B.300d.txt --output glove.42B.300d.w2vformat.txt

import gensim
model_file = "glove.42B.300d.w2vformat.txt"
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, limit=500000)
model.save("wv_100000.model")

import gensim
model = gensim.models.KeyedVectors.load("wv_100000.model")