from FUNCS import plot_probability_array, decision_boundary, get_file
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pandas as pd

pair=[1, 3]
iris = datasets.load_iris()
X = iris.data[:, pair]
y = iris.target
np.unique(y)

plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.RdYlBu)
plt.xlabel("sepal width (cm)")
plt.ylabel("petal width")
# plt.show()
# plt.show()

lr = LogisticRegression(random_state=0).fit(X, y)
probability=lr.predict_proba(X)
iris_plot = plot_probability_array(X,probability)
# iris_plot.show()

probability[0,:]
probability[0,:].sum()
print(np.argmax(probability[0,:]))
# We can apply the  argmax function to each sample
softmax_prediction=np.argmax(probability,axis=1)
print("softmax_prediction: ", softmax_prediction)

yhat =lr.predict(X)
acc_score = accuracy_score(yhat,softmax_prediction)
print("acc_score: ", acc_score)


#dummy class
dummy_class=y.max()+1
#list used for classifiers
my_models=[]
#iterate through each class
for class_ in np.unique(y):
    #select the index of our  class
    select=(y==class_)
    temp_y=np.zeros(y.shape)
    #class, we are trying to classify
    temp_y[y==class_]=class_
    #set other samples  to a dummy class
    temp_y[y!=class_]=dummy_class
    #Train model and add to list
    model=SVC(kernel='linear', gamma=.5, probability=True)
    my_models.append(model.fit(X,temp_y))
    #plot decision boundary
    decision_boundary (X,temp_y,model,iris).show()



