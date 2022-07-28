# Fashion-MNIST-CNN
1.关于CNN卷积神经网络

关于CNN网络的具体原理和结构，大家可以自行在网上查找相关资料进行深入学习，这里不再赘述。

2.FashionMNIST 图像数据集

图像分类数据集中最常用的是手写数字识别数据集MNIST 。但大部分模型在MNIST上的分类精度都超过了95%。为了更直观地观察算法之间的差异，我们将使用一个图像内容更加复杂的数据集Fashion-MNIST 。

FashionMNIST 是由 Zalando（一家德国的时尚科技公司）旗下的研究部门提供。其涵盖了来自 10 种类别的共 7 万个不同商品的正面图片。FashionMNIST 的大小、格式和训练集/测试集划分与原始的 MNIST 完全一致。60000/10000 的训练测试数据划分，28x28 的灰度图片。方便我们进行测试各种神经网络算法。 该数据集识别难度远大于原有的MNIST数据集。

3.数据库导入

本文的代码适用于 TensorFlow 2.3 及 keras 2.4.3 ，Python版本为3.8，如果你使用新版本的第三方库，请考虑降级为本文的版本，或者自行查阅文档修改代码。

所有代码都用keras.datasets接口来加载fashion_mnist数据，从网络上直接下载fashion_mnist数据，无需从本地导入，十分方便。

这意味着你可以将代码中的

(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()

修改为(X_train, y_train), (X_test, y_test) = mnist.load_data()

就可以直接对原始的MNIST数据集进行训练和识别。

4.Baseline版本代码（MLP实现，识别成功率为87.6%）（Baseline文件）

5.CNN卷积神经网络（迭代20次，识别成功率92%）（CNN文件）

6.可视化代码（Show文件）

7.测试代码，该代码是一个三层的MLP模型，可自行对学习率、隐藏层节点数进行设定，优化自己的神经网络（Test文件）
