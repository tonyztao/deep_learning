### 吴恩达深度学习课程笔记

本文作为自己在网易云课堂上学习由吴恩达老师讲授的《神经网络和深度学习》的学习笔记，整个专题共包括五门课程：

1.神经网络和深度学习；

2.改善深层神经网络-超参数调试、正则化以及优化；

3.结构化机器学习项目；

4.卷积神经网络；

5.序列模型。

吴恩达老师作为机器学习的布道师，一直致力于以清晰简洁的方式向大众推广机器学习理论；故作为机器学习领域内的入门者，我也选择此课程开启自己的深度学习之旅。

下面就是从我个人角度出发，记录的主要知识点。

#### 第一课 神经网络和深度学习
##### 第一周：[深度学习概述](https://github.com/tonyztao/deep_learning/tree/master/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%AC%AC%E4%B8%80%E5%91%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E6%A6%82%E8%BF%B0)
课程主要内容：

神经网络的概念、深度学习兴起的原因等；

##### 第二周：[神经网络基础](https://github.com/tonyztao/deep_learning/tree/master/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%AC%AC%E4%BA%8C%E5%91%A8%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%9F%BA%E7%A1%80)
课程主要内容：

逻辑回归的神经网络表示？损失函数的定义？梯度下降法？链式法则？

对于逻辑回归的向量化的表示；为什么要向量化？

Python相关：什么是广播？Numpy库的应用等;

##### 第三周：[浅层神经网络](https://github.com/tonyztao/deep_learning/tree/master/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%AC%AC%E4%B8%89%E5%91%A8%E6%B5%85%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
课程主要内容：

单隐层神经网络的向量化表示；

各类激活函数介绍，优缺点，适用场景等；

为什么需要非线性的激活函数？

参数的随机初始化的重要性；

##### 第四周：[深层神经网络](https://github.com/tonyztao/deep_learning/tree/master/%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%E5%92%8C%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0/%E7%AC%AC%E5%9B%9B%E5%91%A8%E6%B7%B1%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
课程主要内容：

深层神经网络的向量化表示；

深层神经网络维数的确定；

深层神经网络的前向传播和反向传播；

深层神经网络中的参数和超参数；

#### 第二课 改善深层神经网络：超参数调试、正则化以及优化
##### 第一周：[深度学习的实用层面](https://github.com/tonyztao/deep_learning/tree/master/%E6%94%B9%E5%96%84%E6%B7%B1%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E8%AF%95%E3%80%81%E6%AD%A3%E5%88%99%E5%8C%96%E4%BB%A5%E5%8F%8A%E4%BC%98%E5%8C%96/%E7%AC%AC%E4%B8%80%E5%91%A8%E6%B7%B1%E5%BA%A6%E5%AD%A6%E4%B9%A0%E7%9A%84%E5%AE%9E%E7%94%A8%E5%B1%82%E9%9D%A2)
课程主要内容：

训练集、测试集和验证集；

方差和偏差的概念以及应对策略；

正则化方法：L1正则、L2正则、Dropout;

为什么正则化可以减少过拟合？

梯度消失和梯度爆炸;

##### 第二周：[优化算法](https://github.com/tonyztao/deep_learning/tree/master/%E6%94%B9%E5%96%84%E6%B7%B1%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E8%AF%95%E3%80%81%E6%AD%A3%E5%88%99%E5%8C%96%E4%BB%A5%E5%8F%8A%E4%BC%98%E5%8C%96/%E7%AC%AC%E4%BA%8C%E5%91%A8%E4%BC%98%E5%8C%96%E7%AE%97%E6%B3%95)
课程主要内容：

介绍了若干优化算法来提升训练速度；

Mini-Batch梯度下降算法；

指数加权平均以及偏差修正；

RMSprop算法；

Adam算法;

##### 第三周：[超参数调试 Batch归一化和程序框架](https://github.com/tonyztao/deep_learning/tree/master/%E6%94%B9%E5%96%84%E6%B7%B1%E5%B1%82%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C%EF%BC%9A%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E8%AF%95%E3%80%81%E6%AD%A3%E5%88%99%E5%8C%96%E4%BB%A5%E5%8F%8A%E4%BC%98%E5%8C%96/%E7%AC%AC%E4%B8%89%E5%91%A8%E8%B6%85%E5%8F%82%E6%95%B0%E8%B0%83%E8%AF%95Batch%E6%AD%A3%E5%88%99%E5%8C%96%E5%92%8C%E7%A8%8B%E5%BA%8F%E6%A1%86%E6%9E%B6)
课程主要内容：

众多超级参数调优策略和技巧;

Mini-Batch Norm归一化介绍、实现方法以及背后的原理;

softmax多分类模型的介绍;

#### 第三课 结构化机器学习项目
##### 第一周：[机器学习策略一](https://github.com/tonyztao/deep_learning/tree/master/%E7%BB%93%E6%9E%84%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/%E7%AC%AC%E4%B8%80%E5%91%A8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AD%96%E7%95%A5(%E4%B8%80))
课程主要内容：

正交化概念；

单一数字评估指标；

训练、测试集划分；

偏差和方差；

人的表现；

改善模型表现；

##### 第二周：[机器学习策略二](https://github.com/tonyztao/deep_learning/tree/master/%E7%BB%93%E6%9E%84%E5%8C%96%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E9%A1%B9%E7%9B%AE/%E7%AC%AC%E4%BA%8C%E5%91%A8%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0%E7%AD%96%E7%95%A5(%E4%BA%8C))
课程主要内容：

误差分析；

错误样本清除；

数据分布不匹配问题；

迁移学习；

多任务学习；

端到端的深度学习等；

#### 第四课 卷积神经网络
##### 第一周：[卷积神经网络基础](https://github.com/tonyztao/deep_learning/tree/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E7%AC%AC%E4%B8%80%E5%91%A8%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C)
课程主要内容：

计算机视觉都有哪些应用？

边缘检测示例；卷积核；

卷积神经网络示例；

Padding、Strid；

池化等；

##### 第二周：[深度卷积网络：实例探究](https://github.com/tonyztao/deep_learning/tree/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E7%AC%AC%E4%BA%8C%E5%91%A8%E6%B7%B1%E5%BA%A6%E5%8D%B7%E7%A7%AF%E7%BD%91%E7%BB%9C%EF%BC%9A%E5%AE%9E%E4%BE%8B%E6%8E%A2%E7%A9%B6)
课程主要内容：

经典的卷积网络介绍
- LeNet-5
- AlexNet
- VGG
- ResNet
- Inception

迁移学习；

开源实现；

数据扩充；

##### 第三周：[目标检测](https://github.com/tonyztao/deep_learning/tree/master/%E5%8D%B7%E7%A7%AF%E7%A5%9E%E7%BB%8F%E7%BD%91%E7%BB%9C/%E7%AC%AC%E4%B8%89%E5%91%A8%E7%9B%AE%E6%A0%87%E6%A3%80%E6%B5%8B)
课程主要内容：

目标定位；

目标检测；

卷积的滑动窗口实现；

Bounding Box预测；

交并比；

非极大值抑制；

Anchor Box；

YOLO算法；

##### 第四周：[特殊应用：人脸识别和神经风格转换]()

#### 第五课 [序列模型]()

课程主要内容：

为什么选择序列模型？

循环神经网络模型；

不同类型的循环神经网络；

循环神经网络的梯度消失问题；

GRU

LSTM网络 

双向循环神经网络；

深层循环神经网络；

#### 参考项目
在整理自己的笔记的时候，参考了如下内容，在此一并表示感谢：

- [完结撒花！吴恩达DeepLearning.ai《深度学习》课程笔记目录总集](https://blog.csdn.net/koala_tree/article/details/79913655)
- [网易-吴恩达《深度学习》课后练习编程作业和笔记](https://github.com/AlbertHG/Coursera-Deep-Learning-deeplearning.ai)
- [吴恩达《深度学习》系列课程笔记](http://kyonhuang.top/Andrew-Ng-Deep-Learning-notes/#/)
- [吴恩达老师的深度学习课程笔记及资源](https://github.com/fengdu78/deeplearning_ai_books)
