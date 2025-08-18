import mnist
import numpy as np
import pickle
from autograd.utils import PermIterator
from util import setseed
from autograd.BaseNode import *
from autograd.BaseGraph import *
from scipy.ndimage import rotate,zoom,shift
setseed(0) # 固定随机数种子以提高可复现性

save_path = "model/your.npy"


#超参数设置
epoch = 50 #epoch数量
lr = 1e-3
wd1 = 0
wd2 = 1e-5
batchsize = 128
p = 0.01 #dropout概率


def augment_data(dataset, label):
    noise_mask = np.random.rand(len(dataset)) < 0.2
    augmented_dataset = []
    augmented_label = []
    for i,(img, label) in enumerate(zip(dataset, label)):
        img_2d = img.reshape(28,28)#规整图像大小
        #旋转
        angle = np.random.uniform(-15,15)
        rotated = rotate(img_2d, angle=angle, reshape=False)     
        augmented_dataset.append(rotated.ravel())  #展平
        augmented_label.append(label)
        #平移
        dx, dy = np.random.randint(-3, 3, size=2)  #平移范围 [-3, 3]
        shifted = shift(img_2d, shift=(dy, dx))
        augmented_dataset.append(shifted.ravel())
        augmented_label.append(label)
        #噪声
        '''if noise_mask[i]:
            noise = np.random.normal(loc=0, scale=0.1, size=img_2d.shape)  #均值0，标准差0.1
            noisy = img_2d + noise
            noisy = np.clip(noisy, 0.0, 1.0)  #确保像素值在 [0,1] 范围内
            augmented_dataset.append(noisy.ravel())
            augmented_label.append(label)'''
    return np.array(augmented_dataset), np.array(augmented_label)

#提取训练的X和Y,为了使数据更多，把训练集和测试集全部合并
X = np.concatenate([mnist.trn_X,mnist.val_X])
Y = np.concatenate([mnist.trn_Y,mnist.val_Y])
#增强数据，并合并
augmented_X, augmented_Y = augment_data(X, Y)
X = np.concatenate([X,augmented_X])
Y = np.concatenate([Y,augmented_Y])


if __name__ == "__main__":
    #构建训练流程
    graph = Graph([Linear(mnist.num_feat,256),BatchNorm(256),relu(),Dropout(p),\
             Linear(256, 64),BatchNorm(64),relu(),#Dropout(p),\
             Linear(64, mnist.num_class),LogSoftmax(), NLLLoss(Y)])
   
   
    # 训练
    best_train_acc = 0
    dataloader = PermIterator(X.shape[0], batchsize)#初始化迭代器
    for i in range(1, epoch+1):
        hatys = []
        ys = []
        losss = []
        #训练模式，与eval相对，开启dropout
        graph.train()
        #这里调用了dataloader __iter__方法
        #perm是长度为batchsize的索引数组，抽取当前批次样本
        #for循环实现总数/batchsize次数的梯度下降
        for perm in dataloader:
            tX = X[perm]
            tY = Y[perm]
            graph[-1].y = tY #最后结果，便于计算梯度
            graph.flush() #初始化（清空）节点数据
            pred, loss = graph.forward(tX)[-2:] #最后softmax概率，损失
            hatys.append(np.argmax(pred, axis=1)) #预测值
            ys.append(tY) #真实值
            graph.backward()
            graph.optimstep(lr, wd1, wd2) #梯度下降
            losss.append(loss) #记录这一批的损失
        loss = np.average(losss)
        acc = np.average(np.concatenate(hatys)==np.concatenate(ys))
        print(f"epoch {i} loss {loss:.3e} acc {acc:.4f}")
        #保存最优参数到指定路径
        if acc > best_train_acc:
            best_train_acc = acc
            with open(save_path, "wb") as f:
                pickle.dump(graph, f)

