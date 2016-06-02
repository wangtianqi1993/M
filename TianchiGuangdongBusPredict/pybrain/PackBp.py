__author__ = 'wtq'

from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter
from Test import Read


def bpnn():
    ds = SupervisedDataSet(9, 1)
    inputPatterns = Read()

    for p in inputPatterns:
        inputs = p[0]
        targets = p[1]

        inputs = tuple(map(lambda n: float(n) / 6000, inputs))
        targets = tuple(map(lambda n: float(n) / 6000, targets))
        ds.addSample(inputs, targets)
        print(inputs)
        print(targets)
    net = buildNetwork(9, 14, 1)
    trainer = BackpropTrainer(net, ds, verbose=True, learningrate=0.01)
    trainer.trainEpochs(2500)
    trainer.trainUntilConvergence(maxEpochs=3500)

    # save net
    NetworkWriter.writeToFile(net, '/home/wtq/BigData-MachineLearning/Bpnn/BusWorkNet.xml')


def usebp():
    patterns = [

        [[3158, 3503, 3342, 644, 937, 750, 546, 503, 593], [4751]],
        [[3092, 3011, 3217, 675, 882, 881, 543, 598, 564], [4445]],
        [[3180, 3043, 3031, 785, 830, 799, 448, 517, 564], [4514]],
        [[3389, 3469, 3450, 794, 933, 804, 544, 556, 578], [4755]],
        [[3224, 3201, 3433, 904, 737, 772, 522, 591, 585], [4864]],
        [[3503, 3342, 3410, 937, 750, 725, 503, 593, 616], [4646]],
        [[3011, 3217, 3143, 882, 881, 701, 598, 564, 601], [0]],
        [[3043, 3031, 3209, 830, 799, 701, 517, 564, 604], [0]],
        [[3469, 3450, 3446, 933, 804, 756, 556, 578, 553], [0]],
        [[3201, 3433, 3436, 737, 772, 817, 591, 585, 611], [0]],
        [[3342, 3410, 3277, 750, 725, 837, 593, 616, 532], [0]],

    ]

    net = NetworkReader.readFrom('/home/wtq/BigData-MachineLearning/Bpnn/BusWorkNet.xml')
    for p in patterns:
        testInput = p[0]
        targetOut = p[1]
        testInput = tuple(map(lambda n: float(n) / 6000, testInput))
        out = net.activate(testInput)
        print"out->", (out * 6000)
        distance = list(map(lambda x: 6000 * x[0] - x[1], zip(out, targetOut)))
        print(distance)


if __name__ == "__main__":
    i = 0
    if (i == 1):
        bpnn()
    usebp()
    print("end of it")
