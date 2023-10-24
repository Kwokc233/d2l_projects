# 从零开始实现整个线性回归
# 包括数据流水线、模型、损失函数和小批量随机梯度下降优化器
# 只使用张量和自动求导

import random
from mxnet import autograd, np, npx
from d2l import mxnet as d2l

npx.set_np()


# 生成数据集
# 生成一个包含1000个样本的数据集，每个样本包含从标准正态分布中采样的2个特征
def synthetic_data(w, b, num_examples):  # @save
    """生成y=Xw+b+噪声"""
    X = np.random.normal(0, 1, (num_examples, len(w)))
    y = np.dot(X, w) + b
    y += np.random.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))


# 读取数据集
# 训练模型时要对数据集进行遍历，每次抽取一小批量样本，并使用它们来更新我们的模型

def data_iter(batch_size, features, labels):
    # 打乱数据集中的样本并以小批量方式获取数据
    # 该函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为batch_size的小批量。每个小批量包含一组特征和标签
    num_examples = len(features)
    indices = list(range(num_examples))
    # 这些样本是随机读取的，没有特定的顺序
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = np.array(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]


# 定义模型
def linreg(X, w, b):  # @save
    """线性回归模型"""
    return np.dot(X, w) + b


# 定义损失函数
def squared_loss(y_hat, y):  # @save
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2


# 定义优化方法
def sgd(params, lr, batch_size):  # @save
    """小批量随机梯度下降"""
    for param in params:
        param[:] = param - lr * param.grad / batch_size


true_w = np.array([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
print('features:', features[0], '\nlabel:', labels[0])
d2l.set_figsize()
d2l.plt.scatter(features[:, (1)].asnumpy(), labels.asnumpy(), 1);

# 初始化模型参数
# 通过从均值为0、标准差为0.01的正态分布中采样随机数来初始化权重，偏置初始化为0
w = np.random.normal(0, 0.01, (2, 1))
b = np.zeros(1)
w.attach_grad()
b.attach_grad()

# 初始化参数之后，更新这些参数，直到这些参数足够拟合我们的数据
# 每次更新计算损失函数关于模型参数的梯度，向减小损失的方向更新每个参数
# 迭代周期个数`num_epochs`和学习率`lr`都是超参数，分别设为3和0.03。
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss
batch_size = 10

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        with autograd.record():
            l = loss(net(X, w, b), y)  # X和y的小批量损失
        # 计算l关于[w,b]的梯度
        l.backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    train_l = loss(net(features, w, b), labels)
    print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')

print(f'w的估计误差: {true_w - w.reshape(true_w.shape)}')
print(f'b的估计误差: {true_b - b}')
