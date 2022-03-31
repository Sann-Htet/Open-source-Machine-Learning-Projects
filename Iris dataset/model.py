import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import matplotlib
matplotlib.use('Qt5Agg')

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ["sepal-length","sepal-width","petal-length","petal-width","class"]

dataset = pd.read_csv(url,names=names)

print(dataset.shape)
print(dataset.head())
print(dataset.describe())
print(dataset.groupby('class').size())

# Plot the univariate plot
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex = False, sharey = True)
plt.show()

dataset.hist()
#plt.show()

# Multivariate plot

# Scatter plot
scatter_matrix(dataset)
#plt.show()


array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.2
seed = 6
X_trian,X_test,Y_train,Y_test = model_selection.train_test_split(X,Y,test_size = validation_size, random_state = seed)

scoring = 'accuracy'