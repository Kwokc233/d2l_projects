# 线性回归的简洁实现
# （1）通过张量来进行数据存储和线性代数；
# （2）通过自动微分来计算梯度。

from mxnet import autograd, gluon, np, npx
from d2l import mxnet as d2l

npx.set_np()


# 读取数据集
def load_array(data_arrays, batch_size, is_train=True):  # @save
    """构造一个Gluon数据迭代器"""
    dataset = gluon.data.ArrayDataset(*data_arrays)
    return gluon.data.DataLoader(dataset, batch_size, shuffle=is_train)


true_w = np.array([2, -3.4])
true_b = 4.2
# 构造数据集
features, labels = d2l.synthetic_data(true_w, true_b, 1000)
batch_size = 10
data_iter = load_array((features, labels), batch_size)

# 定义模型
from mxnet.gluon import nn  # nn是神经网络的缩写
net = nn.Sequential()
net.add(nn.Dense(1))

# 初始化模型参数
from mxnet import init
net.initialize(init.Normal(sigma=0.01))

# 定义损失函数
loss = gluon.loss.L2Loss()

# 定义优化算法
trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})

# 训练
num_epochs = 3
for epoch in range(num_epochs):
    for X, y in data_iter:
        with autograd.record():
            l = loss(net(X), y)
        l.backward()
        trainer.step(batch_size)
    l = loss(net(features), labels)
    print(f'epoch {epoch + 1}, loss {l.mean().asnumpy():f}')

w = net[0].weight.data()
print(f'w的估计误差： {true_w - w.reshape(true_w.shape)}')
b = net[0].bias.data()
print(f'b的估计误差： {true_b - b}')