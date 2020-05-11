import numpy as np
import random
from collections import defaultdict
import sys
import time


class Point:
    def __init__(self, label, doc_id, tfidf):
        def normalize():
            # compute norm2 of feature vector
            ans = 0.0
            for x in tfidf.values():
                ans += x**2
            ans=ans**0.5
            for i in tfidf:
                tfidf[i] = tfidf[i]/ans # normalize

            self.tfidf = tfidf

        self.label = label
        self.doc_id = doc_id
        normalize()


class Cluster:
    def __init__(self):
        self.centroid = None
        self.points = []

    def set_centroid(self, new_centroid):
        l2_centroid=np.linalg.norm(new_centroid)
        self.centroid = new_centroid/l2_centroid

    def add_point(self, point):
        self.points.append(point)

    def reset_points(self):
        self.points = []

class Kmeans:
    def __init__(self,k_cluster,max_label_change):
        self.list_clusters = []
        self.k_cluster = k_cluster
        self.max_label_change=max_label_change
        self.n_doc = 0
        self.it=0
        with open("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/words_idf", 'r') as f:
            self.dim = len(f.read().splitlines())

    def dist_between_x_y(self, x, y):
        '''
        x is a point;
        y is a vector
        '''
        # compute dot product
        ans = 0.0
        for pos in x.tfidf:
            ans += x.tfidf[pos]*y[pos]
        # return distance
        return (2-2*ans)

    def init_centroid(self, X):
        '''
        init centroid with K-means algorithm
        '''
        rd = random.sample(range(self.n_doc), self.k_cluster)
        for i in rd:
            centroid = np.zeros(self.dim)
            for pos in X[i].tfidf:
                centroid[pos] = X[i].tfidf[pos]
            tmp = Cluster()
            tmp.set_centroid(centroid)
            self.list_clusters.append(tmp)

    
    def init_centroid_v1(self, X):
        '''
        init centroid with K-means++ algorithm
        '''
        def add_centroid(idx):
            centroid = np.zeros(self.dim)
            for pos in X[idx].tfidf:
                centroid[pos] = X[idx].tfidf[pos]
            new_cluster = Cluster()
            new_cluster.set_centroid(centroid)
            self.list_clusters.append(new_cluster)

        # init one centroid
        add_centroid(random.randint(1, self.n_doc))
        # min distance between point x and clusters
        dist_min_between_x_centroid=np.array([sys.maxsize]*self.n_doc,dtype=float)        

        for k in range(self.k_cluster-1):
            for i in range(self.n_doc):
                d=self.dist_between_x_y(X[i],self.list_clusters[-1].centroid)
                dist_min_between_x_centroid[i]=min(dist_min_between_x_centroid[i],d)
            idx_new_centroid=np.argmax(dist_min_between_x_centroid)
            add_centroid(idx_new_centroid)


    def update_centroid(self,new_clusters):
        for clr in new_clusters:
            new_centroid = np.zeros(self.dim)
            for pnt in clr.points:
                for pos in pnt.tfidf:
                    new_centroid[pos] += pnt.tfidf[pos]
            new_centroid = new_centroid/len(clr.points)
            new_centroid=new_centroid/np.linalg.norm(new_centroid)
            clr.set_centroid(new_centroid)
        

    def asign_cluster(self, X):
        new_clusters=[]
        for i in range(self.k_cluster):
            tmp=Cluster()
            new_clusters.append(tmp)

        for x in X:
            dis_min = sys.maxsize
            for i in range(self.k_cluster):
                dis = self.dist_between_x_y(x,self.list_clusters[i].centroid)
                if(dis <dis_min):
                    asign_clr = i
                    dis_min = dis
            new_clusters[asign_clr].add_point(x)
        return new_clusters

    
    def check_stop(self,new_clusters):
        count_labels_change = self.n_doc
        for i in range(self.k_cluster):
            labels_unchange =[ label for label in self.list_clusters[i].points
                              if label in new_clusters[i].points ]
            count_labels_change -= len(labels_unchange)
        if(count_labels_change <self.max_label_change):
            return True

        return False
    
    
    def fit(self, X):
        def run_batch(X_batch):
            for i in range(100):
                new_clusters=self.asign_cluster(X_batch)
                self.update_centroid(new_clusters)
                if(self.check_stop(new_clusters)):
                    self.it=i
                    self.list_clusters=new_clusters
                    break
                self.list_clusters=new_clusters

        
        self.n_doc = len(X)
        self.init_centroid_v1(X)
        run_batch(X)

    def purity(self, X_test):
        new_clusters=self.asign_cluster(X_test)
        n_test = len(X_test)
        count_predict_true = 0
        for clr in new_clusters:
            pre = [0]*self.k_cluster
            for x in clr.points:
                pre[int(x.label)] += 1
            count_predict_true += np.max(pre)
        return str(count_predict_true)+'/'+str(n_test)


def Load_data(pathin):
    """
    Read data
    """
    def tfidf(doc):
        tf_idf = defaultdict(int)
        fea = doc.split()
        for i in fea:
            tmp = i.split(':')
            tf_idf[int(tmp[0])] = float(tmp[1])
        return tf_idf

    with open(pathin, 'r') as f:
        d_lines = f.read().splitlines()
    data = []
    for d in d_lines:
        fea = d.split('<<>>')
        label = fea[0]
        doc_id = fea[1]
        data.append(Point(label, doc_id,tfidf(fea[2])))
    return data

X_train = Load_data("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/train_tfidf_df=3")
X_test = Load_data("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/test_tfidf")
model=Kmeans(20,10)
model.fit(X_train)
ans=model.purity(X_test)
print(ans)
