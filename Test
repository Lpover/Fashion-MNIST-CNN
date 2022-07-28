import numpy
# 引入下面函数是为了激活函数sigmoid function expit()
import scipy.special

# 神经网络类框架
class neuralNetwork:

def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):
# 设置每个输入、隐藏、输出层中的节点数
self.inodes = inputnodes
self.hnodes = hiddennodes
self.onodes = outputnodes

# 链接权重矩阵，分别是隐藏层和输出层的权重
self.wih = numpy.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
self.who = numpy.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))

# 学习率设定
self. lr = learningrate

# 激活函数是sigmoid函数
self.activation_function = lambda \
x: scipy.special.expit(x)

pass

# 训练神经网络
def train(self, inputs_list, targets_list):
# 把输入的数据变成列向量(numpy数组)
inputs = numpy.array(inputs_list, ndmin=2).T
targets = numpy.array(targets_list, ndmin=2).T

# 计算隐藏层的输入，也就是计算输入层和隐藏层之间的权重系数与输入数据的乘积
hidden_inputs = numpy.dot(self.wih, inputs)
# 使用激活函数计算隐藏层的输出
hidden_outputs = self.activation_function(hidden_inputs)
# 计算输出层的输入，也就是计算隐藏层和输出层之间的权重系数与输入数据的乘积
final_inputs = numpy.dot(self.who, hidden_outputs)
# 使用激活函数计算输出层的输出
final_outputs = self.activation_function(final_inputs)
# 计算误差，这个误差是最开始的误差，也就是目标值和输出层输出的数据的差
output_errors = targets - final_outputs
# 这是输出层到隐藏层之间的误差反向传播
hidden_errors = numpy.dot(self.who.T, output_errors)
# 下面是利用误差的反向传播来更新各层之间的权重参数
# 更新隐藏层和输出层之间的权重参数
self.who += self.lr * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)),numpy.transpose(hidden_outputs))
# 更新输入层和隐藏层之间的权重参数
self.wih += self.lr * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)),numpy.transpose(inputs))
pass

# 查询神经网络
def query(self, inputs_list):
# 将输入转换为矩阵（转置T,numpy数组）
inputs = numpy.array(inputs_list, ndmin=2).T
# 计算隐藏层的矩阵积
hidden_inputs = numpy.dot(self.wih, inputs)
# 对隐藏层矩阵积使用激活函数
hidden_outputs = self.activation_function(hidden_inputs)
# 计算输出层的矩阵积
final_inputs = numpy.dot(self.who, hidden_outputs)
# 对输出层矩阵积使用激活函数
final_outputs = self.activation_function(final_inputs)
# 返回结果
return final_outputs

input_nodes = 784
hidden_nodes = 100
output_nodes = 10
learning_rate = 0.05

n = neuralNetwork(input_nodes,hidden_nodes,output_nodes, learning_rate)

training_test_data_file=open("source/fashion-mnist_train.csv", 'r')
training_test_data_list=training_test_data_file.readlines()
training_test_data_file.close()

for record in training_test_data_list:
all_values = record.split(',')
scaled_input = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
targets = numpy.zeros(output_nodes) + 0.01
targets[int(all_values[0])] = 0.99
n.train(scaled_input, targets)
pass

test_data_file = open("source/fashion-mnist_test.csv", 'r')
test_data_list = test_data_file.readlines()
test_data_file.close()

clothe=['T袖','裤子','套衫','裙子','外套','凉鞋','衬衫','运动鞋','包','短鞋']

def train():
scorecard = []
for record in test_data_list:
all_values = record.split(',')
correct_label = int(all_values[0])
inputs = (numpy.asfarray(all_values[1:]) / 255.0 * 0.99) + 0.01
outputs = n.query(inputs)
label = numpy.argmax(outputs)


print("正确服装是:",clothe[correct_label])
print("识别结果为:", clothe[label])
if(correct_label==label):
print("正确")
else:
print("错误")
print()

if (label == correct_label):
scorecard.append(1)
else:
scorecard.append(0)
pass
pass
scorecard_array = numpy.asarray(scorecard)
# print(scorecard)
print("MLP模型,学习率0.05,隐藏层100")
print("识别成功率= ", scorecard_array.sum() / scorecard_array.size * 100,"%")

train()
