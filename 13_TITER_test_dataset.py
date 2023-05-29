import sys
import numpy as np
import h5py
import scipy.io
np.random.seed(1337) # for reproducibility

import theano
from keras.preprocessing import sequence
from keras.optimizers import RMSprop
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.regularizers import l2, l1  # from keras.regularizers import l2, activity_l1
from keras.constraints import maxnorm
from keras.layers.recurrent import LSTM, GRU
from sklearn.metrics import average_precision_score, roc_auc_score


import pandas as pd
from time import time

def decrease_length_samples(samples, upstream_size, downstream_size):

    tis_start_idx = 100

    up_first_idx = tis_start_idx - upstream_size
    down_last_idx = tis_start_idx + downstream_size

    trans_samples = []
    for sample in samples:
        new_sample = sample[up_first_idx:tis_start_idx] + sample[tis_start_idx:down_last_idx]
        trans_samples.append(new_sample)

    return trans_samples

def extract_TITER_samples_test():

    upstream_size = 100
    downstream_size = 103

    test_dataset_samples_file = 'datasets/splits/test/samples.tsv'
    df_samples = pd.read_csv(test_dataset_samples_file, sep='\t', header=0)

    df_pos_samples = df_samples[df_samples['sample_label'] == 1]
    df_neg_samples = df_samples[df_samples['sample_label'] == -1]
    df_samples_eval = pd.concat([df_pos_samples, df_neg_samples])

    pos_samples = df_pos_samples['sample_na'].to_list()
    neg_samples = df_neg_samples['sample_na'].to_list()

    pos_samples = decrease_length_samples(pos_samples, upstream_size, downstream_size)
    neg_samples = decrease_length_samples(neg_samples, upstream_size, downstream_size)

    return pos_samples, neg_samples, df_samples_eval

def save_TITER_samples_eval(y_test, y_test_pred_p, df_samples_eval):

    output_samples_eval_file_atg = 'outputs/TITER_samples_eval_ATG.tsv'
    output_samples_eval_file_nc = 'outputs/TITER_samples_eval_NC.tsv'

    df_samples_eval['TITER_real_label'] = y_test
    df_samples_eval['TITER_score'] = y_test_pred_p

    df_samples_eval_atg = df_samples_eval[df_samples_eval['TIS_type'] == 'ATG']
    df_samples_eval_atg.to_csv(output_samples_eval_file_atg, sep='\t', index=False)

    df_samples_eval_nc = df_samples_eval[df_samples_eval['TIS_type'] != 'ATG']
    df_samples_eval_nc.to_csv(output_samples_eval_file_nc, sep='\t', index=False)


def seq_matrix(seq_list,label):
  tensor = np.zeros((len(seq_list),203,8))
  for i in range(len(seq_list)):
    seq = seq_list[i]
    j = 0
    for s in seq:
      if s == 'A' and (j<100 or j>102):
        tensor[i][j] = [1,0,0,0,0,0,0,0]
      if s == 'T' and (j<100 or j>102):
        tensor[i][j] = [0,1,0,0,0,0,0,0]
      if s == 'C' and (j<100 or j>102):
        tensor[i][j] = [0,0,1,0,0,0,0,0]
      if s == 'G' and (j<100 or j>102):
        tensor[i][j] = [0,0,0,1,0,0,0,0]
      if s == '$':
        tensor[i][j] = [0,0,0,0,0,0,0,0]
      if s == 'A' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,1,0,0,0]
      if s == 'T' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,1,0,0]
      if s == 'C' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,0,1,0]
      if s == 'G' and (j>=100 and j<=102):
        tensor[i][j] = [0,0,0,0,0,0,0,1]
      j += 1
  if label == 1:
    y = np.ones((len(seq_list),1))
  else:
    y = np.zeros((len(seq_list),1))
  return tensor, y


start_time = time()


###### main function ######
codon_tis_prior = np.load('TITER/dict_piror_front_Gaotrain.npy')
codon_tis_prior = codon_tis_prior.item()

codon_list = []
for c in codon_tis_prior.keys():
  if codon_tis_prior[c]!='never' and codon_tis_prior[c] >= 1:
    codon_list.append(c)

print ('Loading test data...')

#  pos_seq_test = np.load('data/pos_seq_test.npy')
#  neg_seq_test = np.load('data/neg_seq_test_all_upstream.npy')


pos_seq_test, neg_seq_test, df_samples_eval = extract_TITER_samples_test()


pos_codon = []
neg_codon = []
for s in pos_seq_test:
  if s[100:103] in codon_list:
    pos_codon.append(codon_tis_prior[s[100:103]])
for s in neg_seq_test:
  if s[100:103] in codon_list:
    neg_codon.append(codon_tis_prior[s[100:103]])

pos_codon = np.array(pos_codon)
neg_codon = np.array(neg_codon)
codon = np.concatenate((pos_codon,neg_codon)).reshape((len(pos_codon)+len(neg_codon),1))
  
pos_seq_test1 = []
neg_seq_test1 = []
for s in pos_seq_test:
  if s[100:103] in codon_list:
    pos_seq_test1.append(s)
for s in neg_seq_test:
  if s[100:103] in codon_list:
    neg_seq_test1.append(s)

print (str(len(pos_seq_test1))+' positive test data loaded...')
print (str(len(neg_seq_test1))+' negative test data loaded...')

pos_test_X, pos_test_y = seq_matrix(seq_list=pos_seq_test1, label=1)
neg_test_X, neg_test_y = seq_matrix(seq_list=neg_seq_test1, label=0)
X_test = np.concatenate((pos_test_X,neg_test_X), axis=0)
y_test = np.concatenate((pos_test_y,neg_test_y), axis=0)

print ('Building model...')
model = Sequential()
model.add(Convolution1D(nb_filter=128,
                        filter_length=3,
                        input_dim=8,
                        input_length=203,
                        border_mode='valid',
                        W_constraint = maxnorm(3),
                        activation='relu',
                        subsample_length=1))
model.add(MaxPooling1D(pool_length=3))
model.add(Dropout(p=0.21370950078747658))
model.add(LSTM(output_dim=256,
               return_sequences=True))
model.add(Dropout(p=0.7238091317104384))
model.add(Flatten())
model.add(Dense(1))
model.add(Activation('sigmoid'))

print ('Compiling model...')
model.compile(loss='binary_crossentropy',
              optimizer='nadam',
              metrics=['accuracy'])

print ('Predicting on test data...')
y_test_pred_n = np.zeros((len(y_test),1))
y_test_pred_p = np.zeros((len(y_test),1))

for i in range(32):
  model.load_weights('TITER/model/bestmodel_'+str(i)+'.hdf5')
  y_test_pred = model.predict(X_test,verbose=1)
  y_test_pred_n += y_test_pred
  y_test_pred_p += y_test_pred*codon

y_test_pred_n = y_test_pred_n/32
y_test_pred_p = y_test_pred_p/32

print ('Perf without prior, AUC: '+str(roc_auc_score(y_test, y_test_pred_n)))
print ('Perf without prior, AUPR: '+str(average_precision_score(y_test, y_test_pred_n)))
print ('Perf with prior, AUC: '+str(roc_auc_score(y_test, y_test_pred_p)))
print ('Perf with prior, AUPR: '+str(average_precision_score(y_test, y_test_pred_p)))


save_TITER_samples_eval(y_test, y_test_pred_p, df_samples_eval)
end_time = time()
pred_time = end_time - start_time
print(f'Prediction time: {pred_time:.3f} secs')


'''
Using TensorFlow backend.
Loading test data...
979 positive test data loaded...
71204 negative test data loaded...
Building model...
Compiling model...
Predicting on test data...
72183/72183 [==============================] - 74s 1ms/step
72183/72183 [==============================] - 72s 994us/step
72183/72183 [==============================] - 71s 986us/step
72183/72183 [==============================] - 71s 981us/step
72183/72183 [==============================] - 71s 985us/step
72183/72183 [==============================] - 71s 983us/step
72183/72183 [==============================] - 71s 982us/step
72183/72183 [==============================] - 71s 979us/step
72183/72183 [==============================] - 71s 980us/step
72183/72183 [==============================] - 71s 978us/step
72183/72183 [==============================] - 70s 976us/step
72183/72183 [==============================] - 71s 981us/step
72183/72183 [==============================] - 71s 977us/step
72183/72183 [==============================] - 70s 975us/step
72183/72183 [==============================] - 71s 978us/step
72183/72183 [==============================] - 71s 985us/step
72183/72183 [==============================] - 71s 985us/step
72183/72183 [==============================] - 71s 978us/step
72183/72183 [==============================] - 71s 986us/step
72183/72183 [==============================] - 70s 974us/step
72183/72183 [==============================] - 71s 978us/step
72183/72183 [==============================] - 71s 979us/step
72183/72183 [==============================] - 71s 979us/step
72183/72183 [==============================] - 71s 979us/step
72183/72183 [==============================] - 70s 975us/step
72183/72183 [==============================] - 71s 983us/step
72183/72183 [==============================] - 72s 1ms/step
72183/72183 [==============================] - 71s 987us/step
72183/72183 [==============================] - 71s 983us/step
72183/72183 [==============================] - 71s 981us/step
72183/72183 [==============================] - 75s 1ms/step
72183/72183 [==============================] - 73s 1ms/step
Perf without prior, AUC: 0.7489088452009358
Perf without prior, AUPR: 0.08979607816478412
Perf with prior, AUC: 0.8704417393084675
Perf with prior, AUPR: 0.16206902057792805
Prediction time: 2295.705 secs
'''
