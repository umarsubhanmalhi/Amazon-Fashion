import os
os.getcwd()
import numpy as np
import pandas as pd
import gzip
import struct
import json
save=""
with open(save+'amazon-fashion-ids.json') as data_file:    
    data_json = json.load(data_file)
json_data=pd.read_json(data_json, orient='index')
json_df=pd.DataFrame(json_data)
json_df.head(5)
asin_list=json_df['asin'].values
asin_list.shape
path="image_features_Men.b" # download from https://jmcauley.ucsd.edu/data/amazon/
f = open(path, 'rb') 
def parse(path):
    k=0
    while k < 200000:
        asin = f.read(10)
        if asin == '': break
        asin=asin.decode("utf-8")
        feature = []
        for i in range(4096): 
            feature.append(struct.unpack('f', f.read(4)))
        if asin in asin_list:
            yield asin, feature
        k += 1
def getDF(path):
    i = 0 
    df = {} 
    for d in parse(path):
        df[i] = d 
        i += 1 
    return pd.DataFrame.from_dict(df, orient='index') 
df = getDF(path)
df.shape
df.head()

joined=json_df.set_index('asin').join(df.set_index(0))
joined.head()
joined = joined[pd.notnull(joined[1])]
joined.head()
joined.shape

joined.groupby('cat').count()

x_train=joined[1]
x_train.head()
x_train.shape
x=np.asarray(np.stack(x_train))
x_train=x.reshape(20904,4096)
x_train.shape
x=x_train
print(x_train)
y_train=joined['cat'].values
y_train.shape
y=y_train
y
from time import time
import keras.backend as K
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.models import Model
from keras.optimizers import SGD
from keras import callbacks
from keras.initializers import VarianceScaling
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=8, n_init=20, n_jobs=4)
y_pred_kmeans = kmeans.fit_predict(x)


# Evaluation metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
nmi = normalized_mutual_info_score
ari = adjusted_rand_score
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
#y=y_train
# Kmeans_acc=acc(y, y_pred_kmeans)
# kmeans_nmi=nmi(y, y_pred_kmeans)
# kmeans_ari=ari(y, y_pred_kmeans)

# print ("ACC:", Kmeans_acc , " Nmi" , kmeans_nmi, "Ari:" , kmeans_ari)

def autoencoder(dims, act='relu', init='glorot_uniform'):
    n_stacks = len(dims) - 1
    # input
    input_img = Input(shape=(dims[0],), name='input')
    x = input_img
    # internal layers in encoder
    for i in range(n_stacks-1):
        x = Dense(dims[i + 1], activation=act, kernel_initializer=init, name='encoder_%d' % i)(x)

    # hidden layer
    encoded = Dense(dims[-1], kernel_initializer=init, name='encoder_%d' % (n_stacks - 1))(x)  # hidden layer, features are extracted from here

    x = encoded
    # internal layers in decoder
    for i in range(n_stacks-1, 0, -1):
        x = Dense(dims[i], activation=act, kernel_initializer=init, name='decoder_%d' % i)(x)

    # output
    x = Dense(dims[0], kernel_initializer=init, name='decoder_0')(x)
    decoded = x
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')
x.shape
x = x.reshape((x.shape[0], -1))
x.shape
x = np.divide(x, 255.)
dims = [x.shape[-1], 8192, 2048, 512, 256]
init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
pretrain_optimizer = SGD(lr=1, momentum=0.9)
pretrain_epochs = 100
batch_size = 256

save_dir = 'D:/School/malhi/Data/result/'

autoencoder, encoder = autoencoder(dims, init=init)

autoencoder.compile(optimizer=pretrain_optimizer, loss='mse')
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs) #, callbacks=cb)
autoencoder.save_weights(save_dir + '/Final_FDC.h5')

autoencoder.load_weights(save_dir + 'Final_FDC.h5')

# Build clustring Model
class ClusteringLayer(Layer):
    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight((self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

n_clusters=8    
clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)

model.compile(optimizer=SGD(0.01, 0.9), loss='kld')

kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))
y_pred


y_pred_last = np.copy(y_pred)

model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

# Deep Clustering
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
y_pred
loss = 0
index = 0
maxiter = 8000
update_interval = 140
index_array = np.arange(x.shape[0])

tol = 0.001 # tolerance threshold to stop training

from sklearn import metrics
metrics.adjusted_rand_score(y, y_pred)
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)
        if y is not None:
            acc = np.round(metrics.accuracy_score(y, y_pred), 5)
            nmi = np.round(metrics.normalized_mutual_info_score(y, y_pred), 5)
            ari = np.round(metrics.adjusted_rand_score(y, y_pred), 5)
            loss = np.round(loss, 5)
            print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' % (ite, acc, nmi, ari), ' ; loss=', loss)

        # check stop criterion - model convergence
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

model.save_weights(save_dir + '/FDC_Clustring_final.h5')
model.load_weights(save_dir + 'FDC_Clustring_final.h5')


# Eval.
q = model.predict(x, verbose=0)
p = target_distribution(q)  # update the auxiliary target distribution p

# evaluate the clustering performance
y_pred = q.argmax(1)
y_pred.shape

# Evaluation metrics
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
nmi = normalized_mutual_info_score
ari = adjusted_rand_score
def acc(y_true, y_pred):
    y_true = y_true.astype(np.int64)
    assert y_pred.size == y_true.size
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    from sklearn.utils.linear_assignment_ import linear_assignment
    ind = linear_assignment(w.max() - w)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size
F_acc=acc(y, y_pred)
F_nmi=nmi(y, y_pred)
F_ari=ari(y, y_pred)
print ("ACC:", F_acc , " Nmi" , F_nmi, "Ari:" , F_ari)

import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y, y_pred)

plt.figure(figsize=(14, 12))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
unique_elements, counts_elements = np.unique(y_pred, return_counts=True)
print("Frequency of unique values of the said array:")
print(np.asarray((unique_elements, counts_elements)))