import os
os.getcwd()
import numpy as np
import pandas as pd
import gzip
import struct
import json
save="/malhi/FIDC"  # Change to, your local directory path
with open(save+'amazon-fashion-ids.json') as data_file:   # the following json file also uploaded on the same github repository
    data_json = json.load(data_file)
json_data=pd.read_json(data_json, orient='index')
json_df=pd.DataFrame(json_data)
json_df.head(5)
asin_list=json_df['asin'].values
asin_list.shape

path="/malhi/FIDC/images/image_features_Men.b" # can be downloaded from https://jmcauley.ucsd.edu/data/amazon/
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
xcopy=x
xcopy.shape
data3 = xcopy.reshape((20904, 64, 64))
data3.shape
x=data3