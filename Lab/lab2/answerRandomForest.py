from numpy.random import rand
import mnist
from answerTree import *
import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
num_tree = 20     # 树的数量
ratio_data = 0.8   # 采样的数据比例
ratio_feat = 0.5 # 采样的特征比例
hyperparams = {
    "depth":20, 
    "purity_bound":0.5,
    "gainfunc": negginiDA
    } # 每颗树的超参数


def buildtrees(X, Y):
    """
    构建随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @param Y: n, 样本的label
    @return: List of DecisionTrees, 随机森林
    """
    # TODO: YOUR CODE HERE
    # 提示：整体流程包括样本扰动、属性扰动和预测输出
    trees=[]
    n,d=X.shape
    for tree_idx in range(num_tree):
        #样本扰动
        #为了避免重复选择，只能用choice函数
        part_idx = np.random.choice(n,int(ratio_data*n),replace=False)#选出的样本标签
        part_X = X[part_idx]
        part_Y = Y[part_idx]

        #属性扰动
        part_feat=list(np.random.choice(d,int(d*ratio_feat),replace=False))

        trees.append(buildTree(part_X,part_Y,part_feat,**hyperparams))
    return trees
        

    raise NotImplementedError    

def infertrees(trees, X):
    """
    随机森林预测
    @param trees: 随机森林
    @param X: n*d, 每行是一个输入样本。 n: 样本数量， d: 样本的维度
    @return: n, 预测的label
    """
    pred = [inferTree(tree, X)  for tree in trees]
    pred = list(filter(lambda x: not np.isnan(x), pred))
    upred, ucnt = np.unique(pred, return_counts=True)
    return upred[np.argmax(ucnt)]
