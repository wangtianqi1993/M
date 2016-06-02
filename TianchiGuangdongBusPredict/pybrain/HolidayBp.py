__author__ = 'wtq'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter
from Holiday import Read


def bpnn():
    ds = SupervisedDataSet(3, 1)
    inputPatterns = Read()

    for p in inputPatterns:
        inputs = p[0]
        targets = p[1]

        inputs = tuple(map(lambda n: float(n) / 4000, inputs))
        targets = tuple(map(lambda n: float(n) / 4000, targets))
        ds.addSample(inputs, targets)
        print(inputs)
        print(targets)
    net = buildNetwork(3, 7, 1)
    trainer = BackpropTrainer(net, ds, verbose=True, learningrate=0.01)
    trainer.trainEpochs(1700)
    trainer.trainUntilConvergence(maxEpochs=4000)

    # save net
    NetworkWriter.writeToFile(net, '/home/wtq/BigData-MachineLearning/Bpnn/BusHolidyNet.xml')


def usebpnn():
    patterns = [
        [[1813, 1839, 1625], [1537]],
        [[1565, 1463, 1215], [1433]],
        [[1839, 1625, 1537], [1660]],
        [[1463, 1215, 1433], [1482]],
        [[1625, 1537, 1660], [1256]],
        [[1215, 1433, 1482], [1391]],
        [[1537, 1660, 1256], [0]]
    ]

    net = NetworkReader.readFrom('/home/wtq/BigData-MachineLearning/Bpnn/BusHolidyNet.xml')

    for p in patterns:
        testInput = p[0]
        targetOut = p[1]
        testInput = tuple(map(lambda n: float(n) / 4000, testInput))
        out = net.activate(testInput)
        # print(out * 1000)
        distance = list(map(lambda x: 4000 * x[0] - x[1], zip(out, targetOut)))
        print(distance)


if __name__ == "__main__":
    i = 0
    if (i == 1):
        bpnn()
    usebpnn()
    print("end of it")
