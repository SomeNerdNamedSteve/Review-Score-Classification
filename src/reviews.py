import pandas as pd
from sklearn import tree, svm, neural_network, neighbors
from textblob import TextBlob
import warnings
warnings.filterwarnings('ignore')

train_in = []
train_out = []

tree_clf = tree.DecisionTreeClassifier()
mlp_clf = neural_network.MLPClassifier()
line_clf = svm.LinearSVC()
knn_clf = neighbors.KNeighborsClassifier()

clfs = [tree_clf, mlp_clf, line_clf, knn_clf]

data = pd.read_csv('data.csv')

train = data[0:300]

for i in range(0,300):
    review = data['reviews.text'][i]
    rating = data['reviews.rating'][i]
    do_recommend = data['reviews.doRecommend'][i]
    analysis = TextBlob(review)
    polarity = analysis.sentiment.polarity
    train_in.append([polarity, rating])
    train_out.append([do_recommend])

test_review = data['reviews.text'][300]
test_rating = data['reviews.rating'][300]
test_recommend = data['reviews.doRecommend'][300]
test_analysis = TextBlob(test_review)
test_polarity = test_analysis.sentiment.polarity
test_data = [[test_polarity, test_rating]]

for clf in clfs:
    clf = clf.fit(train_in, train_out)
    pred = clf.predict(test_data)
    print(str(type(clf)) + ": " + str(pred))