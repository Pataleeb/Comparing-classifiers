import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap

file_path="marriage.csv"
data=pd.read_csv(file_path)
#print(data.head())

##Separate label and features
X=data.iloc[:,:-1]
y=data.iloc[:,-1]

##Data split
Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.2, random_state=12, shuffle=True)
Xtrain = np.array(Xtrain, dtype=np.float64)
Xtest = np.array(Xtest, dtype=np.float64)
### Variance calculation for Gaussian NB
variance = Xtrain.var(axis=0)
variance[variance < 1e-3]=1e-3
#print(variance)

NB = GaussianNB(var_smoothing=1e-3)
NB.fit(Xtrain, ytrain)
y_train_pred_NB=NB.predict(Xtrain)
y_test_pred_NB=NB.predict(Xtest)

###KNN
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Xtrain, ytrain)
y_train_pred_knn=knn.predict(Xtrain)
y_test_pred_knn=knn.predict(Xtest)

##Logistic regression
logreg = LogisticRegression(max_iter=1000)
logreg.fit(Xtrain, ytrain)
y_train_pred_logreg=logreg.predict(Xtrain)
y_test_pred_logreg=logreg.predict(Xtest)

##Training and Testing accuracy
train_error_NB=1-accuracy_score(ytrain,y_train_pred_NB)
testing_error_NB=1-accuracy_score(ytest,y_test_pred_NB)

train_error_knn=1-accuracy_score(ytrain,y_train_pred_knn)
testing_error_knn=1-accuracy_score(ytest,y_test_pred_knn)

train_error_logreg=1-accuracy_score(ytrain,y_train_pred_logreg)
testing_error_logreg=1-accuracy_score(ytest,y_test_pred_logreg)

print("\nComparison of Classifier Performance:")
print(f"Naïve Bayes: Training Error = {train_error_NB:.4f}, Test  Error = {testing_error_NB:.4f}, Test Accuracy = {1 - testing_error_NB:.4f}")
print(f"Logistic Regression: Training Error = {train_error_logreg:.4f}, Test Error = {testing_error_logreg:.4f}, Test Accuracy = {1 - testing_error_logreg:.4f}")
print(f"KNN: Training Error = {train_error_knn:.4f}, Test  Error = {testing_error_knn:.4f}, Test Accuracy = {1 - testing_error_knn:.4f}")

### Q2
pca = PCA(n_components=2)
X_train_pca = pca.fit_transform(Xtrain)
X_test_pca = pca.transform(Xtest)

pca_NB=GaussianNB()
pca_NB.fit(X_train_pca, ytrain)

pca_logreg=LogisticRegression(max_iter=1000)
pca_logreg.fit(X_train_pca, ytrain)

pca_knn=KNeighborsClassifier(n_neighbors=5)
pca_knn.fit(X_train_pca, ytrain)

def plot_db(model, X, y, title):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))


    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)


    plt.contourf(xx, yy, Z, alpha=0.3, cmap=ListedColormap(['lightgreen', 'lightsalmon']))
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=ListedColormap(['darkgreen', 'darkred']), edgecolor='k')
    plt.xlabel("PCA Component 1")
    plt.ylabel("PCA Component 2")
    plt.title(title)


    plt.show()


plot_db(pca_NB, X_train_pca, ytrain, "Naïve Bayes Decision Boundary")
plot_db(pca_logreg, X_train_pca, ytrain, "Logistic Regression Decision Boundary")
plot_db(pca_knn, X_train_pca, ytrain, "KNN Decision Boundary")
