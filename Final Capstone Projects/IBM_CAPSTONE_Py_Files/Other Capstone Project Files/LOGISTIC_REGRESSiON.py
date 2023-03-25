from FUNCS import get_file, plot_confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import jaccard_score
import matplotlib.pyplot as plt


url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/ChurnData.csv"
pathname = 'ChurnData.csv'

churn_df = get_file(url, pathname)

churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
churn_df.head()

X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
y = np.asarray(churn_df['churn'])
print(X[0:5])
print(y[0:5])
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
print(LR)
yhat = LR.predict(X_test)
print("yhat: ", yhat)
yhat_prob = LR.predict_proba(X_test)
print("yhat_prob: ", yhat_prob[0][0:5])

# Jaccard Score
jac_score = jaccard_score(y_test, yhat,pos_label=0)
print("Jaccard_Score: ", jac_score)
print(confusion_matrix(y_test, yhat, labels=[1,0]))
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')
print (classification_report(y_test, yhat))


LR2 = LogisticRegression(C=.001, solver='liblinear')
prediction = LR2.fit(X_train, y_train)
yhat_prob2 = LR2.predict_proba(X_test)
print(yhat_prob2)
# print ("LogLoss: : %.2f" % log_loss(y_test, yhat_prob2))



