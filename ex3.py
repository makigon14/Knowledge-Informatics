
# coding: utf-8

# ## 知識情報学 第3回演習サンプルプログラム ex3.ipynb
# - Programmed by Nattapong Thammasan, 監修　福井健一
# - Last updated: 2017/10/12
# - Checked with Python 3.6, scikit-learn 0.19
# - MIT License
# 
# ## 決定木学習による識別と決定木の描画
# - 要GraphViz

from sklearn import datasets
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier,export_graphviz
from sklearn.metrics import precision_recall_fscore_support
from sklearn.preprocessing import scale 
from sklearn.model_selection import cross_validate
import matplotlib.pyplot as plt

# テストデータの割合
test_proportion = 0.3
# Iris データセットをロード  
iris = datasets.load_iris()
# 特徴ベクトルを取得
X = iris.data
# クラスラベルを取得
y = iris.target
# z標準化
X_std = scale(X)


# 学習データとテストデータに分割
X_train, X_test, y_train, y_test = train_test_split(X_std, y, test_size = test_proportion, random_state = 1)

# エントロピーを指標とする決定木のインスタンスを生成し，決定木のモデルに学習データを適合させる
tree = DecisionTreeClassifier(criterion='entropy', max_depth=5)
tree.fit(X_train, y_train)

# ### 課題1(a) 学習した決定木を用いて学習データおよびテストデータのクラスを予測し，結果をy_train_predicted, y_test_predictedに格納する
y_train_predicted = tree.predict(X_train)
y_test_predicted = tree.predict(X_test)

# テストデータの正解クラスと決定木による予測クラスを出力
print("Test Data")
print("True Label     ", y_test)
print("Predicted Label", y_test_predicted)


# ### 課題1(b) 関数precision_recall_fscore_supportを使用して，学習データおよびテストデータに対するprecision，recall，F値の算出しfscore_train, fscore_testに格納する
fscore_train = precision_recall_fscore_support(y_train,y_train_predicted)
fscore_test = precision_recall_fscore_support(y_test,y_test_predicted)

# ### 平均precision, recall, F値
print('Training data')
print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][0], fscore_train[1][0], fscore_train[2][0]))
print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][1], fscore_train[1][1], fscore_train[2][1]))
print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_train[0][2], fscore_train[1][2], fscore_train[2][2]))
print('Average Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (np.average(fscore_train[0]), np.average(fscore_train[1]), np.average(fscore_train[2])))

print('Test data')
print('Class 0 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][0], fscore_test[1][0], fscore_test[2][0]))
print('Class 1 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][1], fscore_test[1][1], fscore_test[2][1]))
print('Class 2 Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (fscore_test[0][2], fscore_test[1][2], fscore_test[2][2]))
print('Average Precision: %.3f, Recall: %.3f, Fscore: %.3f' % (np.average(fscore_test[0]), np.average(fscore_test[1]), np.average(fscore_test[2])))

# ### 学習した決定木モデルをGraphviz形式で出力
# - 出力されたtree.dotファイルは，別途Graphviz(gvedit)から開くことで木構造を描画できる
# - コマンドラインの場合は，'dot -T png tree.dot -o tree.png'

export_graphviz(tree, out_file='tree.dot')
print("tree.dot file is generated")

# ### 課題(c) 10 fold cross-valiation を行い，最大深さを変化させたときの学習データおよびテストデータに対する平均Accuracyを算出し，グラフにプロットしなさい
# - ヒント：model_selection.cross_validateを使用すると良い

train_accuracy_li = []
test_accuracy_li = []

for i in range(1,10):
    tree = DecisionTreeClassifier(criterion='entropy', max_depth=i)
    cross = cross_validate(tree, X_std, y=y,groups=None, scoring=None, cv=10,n_jobs=1, verbose=0, fit_params=None, pre_dispatch='2*n_jobs', return_train_score='warn')
    train_accuracy_li.append(sum(cross["train_score"])/10)
    test_accuracy_li.append(sum(cross["test_score"])/10)

x = np.arange(1, 10, 1)
plt.plot(x, train_accuracy_li, label="train")
plt.plot(x, test_accuracy_li, label="test")
plt.xlabel("Maximum depth")
plt.ylabel("accuracy")
plt.legend()
 






