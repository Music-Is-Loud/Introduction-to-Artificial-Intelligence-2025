import numpy as np

# 超参数
# TODO: You can change the hyperparameters here
lr = 0.25  # 学习率
wd = 1e-4 # l2正则化项系数
eps=1e-7#防止log(0)


def predict(X, weight, bias):
    """
    使用输入的weight和bias，预测样本X是否为数字0。
    @param X: (n, d) 每行是一个输入样本。n: 样本数量, d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @return: (n,) 线性模型的输出，即wx+b
    """
    # TODO: YOUR CODE HERE
    return X@weight+bias;
    raise NotImplementedError

def sigmoid(x):
    return 1 / (np.exp(-x) + 1)


def step(X, weight, bias, Y):
    """
    单步训练, 进行一次forward、backward和参数更新
    @param X: (n, d) 每行是一个训练样本。 n: 样本数量， d: 样本的维度
    @param weight: (d,)
    @param bias: (1,)
    @param Y: (n,) 样本的label, 1表示为数字0, -1表示不为数字0
    @return:
        haty: (n,) 模型的输出, 为正表示数字为0, 为负表示数字不为0
        loss: (1,) 由交叉熵损失函数计算得到
        weight: (d,) 更新后的weight参数
        bias: (1,) 更新后的bias参数
    """
    # TODO: YOUR CODE HERE
    n=Y.shape[0]
    haty=predict(X,weight,bias)#算出现在的模型输出
    loss=0.0#计算每一个数据的logLoss的总和，最后加正则化取平均
    dL_dweight=np.zeros_like(weight)#计算关于w梯度总合，最后加正则化取平均
    dL_dbias=0.0#计算关于b梯度总合，最后取平均

    for i in range(n):
        z=Y[i]*haty[i]
        if z>500:#得到的yi(wTxi+b)值很大，导致损失函数近似于0
            loss_i=0.0
            grad_b=0.0
        elif z<-500:#得到的yi(wTxi+b)值很小，损失函数近似于-z
            loss_i=-z
            grad_b=-Y[i]
        else:
            loss_i=np.log(1+np.exp(-z))
            grad_b=-Y[i]*np.exp(-z)/(1+np.exp(-z))
        loss+=loss_i
        dL_dweight+=grad_b*X[i]
        dL_dbias+=grad_b

    #正则化项并取平均
    loss=loss/n+wd*(np.dot(weight,weight))
    dL_dweight=dL_dweight/n+2*wd*weight
    dL_dbias=dL_dbias/n
    weight-=lr*dL_dweight
    bias-=lr*dL_dbias

    return (haty, loss, weight, bias)
    raise NotImplementedError
