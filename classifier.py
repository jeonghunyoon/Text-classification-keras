from data_helpers import get_data, get_input_seqs, pad_str, get_onehot_labels, build_voca
from word2vec import Word2Vec
from hyper_params import HyperParams
from keras_model import KearsModel

# Embedding flag
IS_EMBEDDING = True
print('Use embedding? %s' %IS_EMBEDDING)

# 1. Loading the data
labels, messages = get_data('./data/spam.csv')
inputs = pad_str(messages)
voca_lookup, _ = build_voca(inputs)
if IS_EMBEDDING:
    hparams = HyperParams.get_hyper_params()
    word2vec = Word2Vec(inputs.tolist(), hparams.embedding_dim)  # Word2vec
    X = word2vec.get_embedding()
else:
    X = get_input_seqs(inputs, voca_lookup)
y = get_onehot_labels(labels)
print('Input shape : %s' % (X.shape,))
print('Output shape : %s' % (y.shape,))

# 2. Train
model = KearsModel(X=X, y=y, voca_lookup=voca_lookup, is_embedding=IS_EMBEDDING)
model.train()
