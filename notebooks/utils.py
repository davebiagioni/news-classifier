import keras
from keras.models import Sequential
from keras.layers import Dense, GRU, Bidirectional, Activation
from keras.layers.core import Dropout
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence


def create_rnn_model_for_training(p):
  model = Sequential()
  model.add(Embedding(p['top_words']+1, p['embedding_dim'], input_length=p['max_length']))
  if p['bidirectional']:
    model.add(Bidirectional(GRU(p['lstm_dim'], dropout_U=p['dropout_U'], dropout_W=p['dropout_W'])))
  else:
    model.add(GRU(p['lstm_dim'], dropout_U=p['dropout_U'], dropout_W=p['dropout_W']))
  model.add(Dropout(p['keep_prob']))
  model.add(Dense(1, activation='sigmoid'))

  model.compile(loss='binary_crossentropy',
                optimizer='adam',
                metrics=['accuracy'])

  return model

def split_model_layers(model, p):
  emb_model = Sequential()
  emb_model.add(Embedding(p['top_words']+1, p['embedding_dim'], input_length=p['max_length'], 
    weights=model.layers[0].get_weights()))
  if p['bidirectional']:
    emb_model.add(Bidirectional(GRU(p['lstm_dim'], return_sequences=True), 
      weights=model.layers[1].get_weights()))
  else:
    emb_model.add(GRU(p['lstm_dim'], return_sequences=True, weights=model.layers[1].get_weights()))

  out_model = Sequential()
  if p['bidirectional']:
    out_model.add(Dense(1, weights=model.layers[3].get_weights(), activation='sigmoid',
      input_dim=2*p['lstm_dim']))
  else:
    out_model.add(Dense(1, weights=model.layers[3].get_weights(), activation='sigmoid',
      input_dim=p['lstm_dim']))

  return emb_model, out_model

def evaluate_sequential_probs(X, idx, model_emb, model_out):
  embeddings = model_emb.predict_proba(X[idx:idx+1, :], verbose=0)
  return model_out.predict_proba(embeddings.squeeze(), verbose=0).squeeze()



