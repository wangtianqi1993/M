#!/usr/bin/python
# -*- coding: utf-8 -*-

__author__ = 'wtq'

import math
import random
from Test import Read
from Test import autoNorm
from Test import autiNorm


# sigmoid function
def sigmoid(x):
    return math.tanh(x)


def dsigmoid(y):
    return 1.0 - y ** 2


def makeMatrix(Y, X, fill=0.0):
    m = []
    for i in range(Y):
        m.append([fill] * X)
    return m


class Nauron:
    def __init__(self):
        pass


class NN:
    def __init__(self, numinput, numhidden, numoutput):

        self.numinput = numinput + 1  # +1 for bias input node
        self.numhidden = numhidden
        self.numoutput = numoutput
        # ＊使intput列表长度变为了self.numinput
        self.inputact = [1.0] * self.numinput
        self.hiddenact = [1.0] * self.numhidden
        self.outputact = [1.0] * self.numoutput

        # 输出层上每个节点的诱导局部域, 在更新w时会用到
        self.output_in = [1] * self.numoutput
        # 隐藏层上每个节点的诱导局部域, 在更新w时会用到
        self.hidden_in = [1] * self.numhidden

        # 目标的训练参数， 训练网络的过程就是要让下面的两个weights参数收敛
        self.inputweights = makeMatrix(self.numinput, self.numhidden)
        self.outputweights = makeMatrix(self.numhidden, self.numoutput)

        # randomize weights随机生成两个权值矩阵
        for i in range(self.numinput):
            for j in range(self.numhidden):
                self.inputweights[i][j] = random.uniform(-0.2, 0.2)
        for j in range(self.numhidden):
            for k in range(self.numoutput):
                self.outputweights[j][k] = random.uniform(-2.0, 2.0)

        self.inputchange = makeMatrix(self.numinput, self.numhidden)
        self.outputchange = makeMatrix(self.numhidden, self.numoutput)
        # TODO:Random fill matrix of weights

    def update(self, inputs):
        """Update network"""
        # 训练好网络后，可以将测试数据输入到update中，返回网络所预测的结果
        if len(inputs) != self.numinput - 1:
            raise ValueError('Wrong number of inputs, should have %i inputs.' % self.numinput)

        # ACTIVATE ALL NEURONS INSIDE A NETWORK

        # 对输入数据归一化处理
        self.inputNorm, self.inputRange, self.inputMin = autoNorm(inputs)
        # Activate input layers neurons (-1 ignore bias node)

        for i in range(self.numinput - 1):
            # self.inputact[i] = inputs[i]
            self.inputact[i] = self.inputNorm[i]

        # Activate hidden layers neurons
        for h in range(self.numhidden):
            sum = 0.0
            # 下面这个for循环在计算第h个隐藏层节点的输入值，即为各个输入节点的输入值与该节点与隐藏层节间的权值之积再求和
            for i in range(self.numinput):
                sum = sum + self.inputact[i] * self.inputweights[i][h]

            # 隐藏层第h个节点的诱导局部域
            self.hidden_in[h] = sum
            # 隐藏层第h个节点的输出值
            self.hiddenact[h] = sigmoid(sum)

        # Activate output layers neurons
        for o in range(self.numoutput):
            sum = 0.0
            for h in range(self.numhidden):
                sum = sum + self.hiddenact[h] * self.outputweights[h][o]

            # 输出层第h个节点的诱导局部域
            self.output_in[o] = sum
            # 输出层第h个节点的输出值
            self.outputact[o] = sigmoid(sum)

        # 将计算得到的输出反归一化

        self.output = autiNorm(self.outputact, self.inputRange, self.inputMin)
        # return self.outputact[:]

        return self.output

    def backPropagate(self, targets, learningrate, momentum):
        """Back Propagate """

        if len(targets) != self.numoutput:
            raise ValueError('Wrong number of target values.')

        # calculate error for output neurons 由输出层到输入层反向计算误差
        output_deltas = [0.0] * self.numoutput
        for k in range(self.numoutput):
            # error = targets[k]-self.outputact[k]
            error = targets[k] - self.output[k]
            # dsigmoid里的值换成self.output_in[k]
            output_deltas[k] = dsigmoid(self.outputact[k]) * error
            print(output_deltas[k])

        # calculate error for hidden neurons
        hidden_deltas = [0.0] * self.numhidden
        for j in range(self.numhidden):
            error = 0.0
            for k in range(self.numoutput):
                error = error + output_deltas[k] * self.outputweights[j][k]

            # dsigmoid里的值换成self.hidden_in[k]
            hidden_deltas[j] = dsigmoid(self.hiddenact[j]) * error

        # update output weights
        for j in range(self.numhidden):
            for k in range(self.numoutput):
                change = output_deltas[k] * self.hiddenact[j]
                self.outputweights[j][k] += learningrate * change + momentum * self.outputchange[j][k]
                self.outputchange[j][k] = change

        # update input weights
        for i in range(self.numinput):
            for j in range(self.numhidden):
                change = hidden_deltas[j] * self.inputact[i]
                self.inputweights[i][j] += learningrate * change + momentum * self.inputchange[i][j]
                self.inputchange[i][j] = change

        # calculate error
        error = 0.0
        for k in range(len(targets)):
            print("target, output", targets[k], self.output[k])
            error = error + 0.5 * (targets[k] - self.output[k]) ** 2
        return error

    def train(self, patterns, iterations=50, learningrate=0.05, momentum=0.01):
        """Train network a patterns"""

        for i in range(iterations):
            error = 0.0

            for p in patterns:
                inputs = p[0]
                targets = p[1]

                self.update(inputs)
                error = error + self.backPropagate(targets, learningrate, momentum)
            # if i % 100 == 0:
            print('error %-.5f' % error)


def test():
    # Teach network XOR function
    patterns = [
        [[3418, 3717, 3033, 3324, 3269, 3032], [2969]],
        [[3250, 3306, 3387, 3269, 3032, 2969], [3306]],
        [[3151, 3223, 3324, 3032, 2969, 3306], [3338]],
        [[3272, 3347, 3269, 2969, 3306, 3338], [3450]],
        [[3054, 3740, 3032, 3306, 3338, 3450], [3102]],
        [[3717, 3033, 2969, 3338, 3450, 3102], [3095]],
        [[3306, 3387, 3306, 3450, 3102, 3095], [3165]],
    ]
    # buspattern = Read()
    # create a network with two input, two hidden, and one output nodes
    network = NN(6, 9, 1)
    # train it with some patterns
    network.train(patterns)
    # test it
    for pat in patterns:
        # print(pat[0], '=', int(network.update(pat[0])[0]+0.5))
        print(pat[0], '=', network.update(pat[0])[0])


if __name__ == '__main__':
    test()
    print("End of it!!!")
