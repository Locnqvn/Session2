import numpy as np
from sklearn.svm import LinearSVC

def loaddata(pathin):
    def convert_dense_vector(text,dim):
        v=np.zeros(dim)
        x_fea=text.split()
        for x in x_fea:
            fea=x.split(':')
            v[int(fea[0])]=float(fea[1])
        return v

    with open(pathin,'r') as f:
        d_lines=f.read().splitlines()
    with open("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/words_idf",'r') as f:
        dim=f.read().splitlines()

    X=[]
    Y=[]
    for doc in d_lines:
        fea_doc=doc.split('<<>>')
        Y.append(int(fea_doc[0]))
        X.append(convert_dense_vector(fea_doc[2],dim))
    
    return X,Y

# accuracy function
def accuracy(pre,Y_test):
    pre_true=np.equal(pre,Y_test)
    return float(pre_true.sum()/len(pre))

#load data
X_train,Y_train=loaddata("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/train_tfidf_df=3")
X_test,Y_test=loaddata("/home/lnq/Desktop/20192/code_lab/Text Preprocessing/test_tfidf")

model=LinearSVC(tol=1e-4,C=10)
model.fit(X_train,Y_train)
pre=model.predict(X_test)
print(str(accuracy(pre,Y_test)))