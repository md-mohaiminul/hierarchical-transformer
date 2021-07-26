import glob
import numpy as np
import collections
import torch
import os
import pandas as pd
from s3dg import S3D
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

net = S3D('s3d_dict.npy', 512)
net.load_state_dict(torch.load('s3d_howto100m.pth'))
num_clusters = 10

# number of files: 1238911
# embeddings dimention: 512
count = 0
for filename in glob.glob('/playpen/mohaiminul/howto100m_csv/*.csv'):
    # print("File :",count)
    # print("------------------------")
    df = pd.read_csv(filename)
    l = df['text'].values.tolist()
    if(len(l)< 100):
        continue
    f = os.path.split(filename)[-1]
    print('File name: ', f)

    # Compute embedding
    embedding = net.text_module(l)['text_embedding']
    embedding = embedding.detach().numpy()

    # Normalize data
    scalar = StandardScaler()
    scalar.fit(embedding)
    embedding = scalar.transform(embedding)

    #perform PCA
    pca = PCA(n_components=0.90, random_state=0)
    pca.fit(embedding)
    embedding = pca.transform(embedding)

    explaind = np.cumsum(pca.explained_variance_ratio_ *100)
    print('Number of pca components for 90% variability', explaind.shape[0])

    #kmeans
    kmeans = KMeans(n_clusters=num_clusters, init='k-means++', random_state=0,
                    n_init=10, max_iter=300, tol=0.0001, verbose=0).fit(embedding)

    #save files
    df['cluster_id'] = kmeans.labels_
    plt.hist(df['cluster_id'], bins=num_clusters)
    plt.xlabel('Cluster id')
    plt.ylabel('Number of data')
    plt.savefig('./Outputs/'+f+'_histogram.png')
    plt.clf()
    unique, counts = np.unique(df['cluster_id'], return_counts=True)
    print('Number of data in each cluster')
    print(dict(zip(unique, counts)))
    df = df.sort_values('cluster_id')
    df.to_csv(os.path.join("./Cluster_example",os.path.split(filename)[-1]))
    count += 1
    if (count == 100):
        break


