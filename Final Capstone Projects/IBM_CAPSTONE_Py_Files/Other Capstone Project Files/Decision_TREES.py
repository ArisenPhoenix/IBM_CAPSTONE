from FUNCS import get_file
from sklearn import preprocessing
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn import tree

# Notice: You might need to uncomment and install the pydotplus and graphviz libraries if you have not installed these before
#!conda install -c conda-forge pydotplus -y
#!conda install -c conda-forge python-graphviz -y

url= 'https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv'
filename="drug200.csv"

df = get_file(url, filename)

X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
y = df["Drug"]

le_sex = preprocessing.LabelEncoder()
le_sex.fit(['F','M'])
X[:,1] = le_sex.transform(X[:,1])

le_BP = preprocessing.LabelEncoder()
le_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
# print("PRE-BLOOD PRESSURE-TRANSFORMATION: ", X[:,3])
X[:,2] = le_BP.transform(X[:,2])
# print("BLOOD PRESSURE TRANSFORMATION: ", X[:,2])



le_Chol = preprocessing.LabelEncoder()
le_Chol.fit([ 'NORMAL', 'HIGH'])

# print("PRE-CHOLESTEROL-TRANSFORMATION: ", X[:,3])
X[:,3] = le_Chol.transform(X[:,3])
# print("CHOLESTEROL TRANSFORMATION: ", X[:,3])

print(X[0:5])
print(le_sex)


X_trainset, X_testset, y_trainset, y_testset = train_test_split(X, y, test_size=0.3, random_state=3)

drugTree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drugTree.fit(X_trainset,y_trainset)

prediction_tree = drugTree.predict(X_testset)
# print (prediction_tree[0:5])
# print (y_testset[0:5])

# print(df.nunique())
print("DecisionTrees's Accuracy: ", metrics.accuracy_score(y_testset, prediction_tree))
# text_format = tree.export_text(drugTree)
# with open("decision_tree.txt", "w") as file:
#     file.write(text_format)
# print(text_format)
print(df.head())
tree.plot_tree(drugTree)

plt.show()

