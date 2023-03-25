from FUNCS import get_file, plot_confusion_matrix
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import svm, preprocessing
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, jaccard_score, confusion_matrix
import matplotlib.pyplot as plt

# Modeling SVMs with scikitlearn or sklearn

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/cell_samples.csv"
pathname = "cell_samples.csv"
cell_df = get_file(url, pathname)
print(cell_df.head())
ax = cell_df[cell_df['Class'] == 4][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='DarkBlue', label='malignant');
cell_df[cell_df['Class'] == 2][0:50].plot(kind='scatter', x='Clump', y='UnifSize', color='Yellow', label='benign', ax=ax);
# plt.show()

print(cell_df.dtypes)
cell_df = cell_df[pd.to_numeric(cell_df['BareNuc'], errors='coerce').notnull()]
cell_df['BareNuc'] = cell_df['BareNuc'].astype('int')
cell_df.dtypes

# Cure Test_Train sets
feature_df = cell_df[['Clump', 'UnifSize', 'UnifShape', 'MargAdh', 'SingEpiSize', 'BareNuc', 'BlandChrom', 'NormNucl', 'Mit']]
X = np.asarray(feature_df)
cell_df['Class'] = cell_df['Class'].astype('int')
y = np.asarray(cell_df['Class'])

# Procure TEST TRAIN SETS
X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)

clf = svm.SVC(kernel='rbf')
clf.fit(X_train, y_train)
yhat = clf.predict(X_test)


cnf_matrix = confusion_matrix(y_test, yhat, labels=[2,4])
np.set_printoptions(precision=2)

# print (classification_report(y_test, yhat))

# Plot non-normalized confusion matrix
# plt.figure()
maxtrix_plot1 = plot_confusion_matrix(cnf_matrix, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')
maxtrix_plot1.show()
clf2 = svm.SVC(kernel="linear")
clf2.fit(X_train, y_train)
yhat2 = clf2.predict(X_test)
yhat2

cnf_matrix2 = confusion_matrix(y_test, yhat2, labels=[2,4])
np.set_printoptions(precision=2)

# print(classification_report(y_test, yhat))

# Plot non-normalized confusion matrix

print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))

print("Avg F1-score: %.4f" % f1_score(y_test, yhat2, average='weighted'))
print("Jaccard score: %.4f" % jaccard_score(y_test, yhat2,pos_label=2))

matrix_plot2 = plot_confusion_matrix(cnf_matrix2, classes=['Benign(2)','Malignant(4)'],normalize= False,  title='Confusion matrix')


matrix_plot2.show()