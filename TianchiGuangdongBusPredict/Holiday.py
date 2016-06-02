#!/usr/bin/python
# -*- coding: utf-8 -*-
__author__ = 'wtq'

import math
import numpy as np


def Read():
    Tarresult = []

    for i in range(1, 39+1):
        lnum = 0
        result = []
        content = " "
        target = " "
        day = "21"
        with open("/home/wtq/BigData-MachineLearning/Data/BusData/11lineBus/all/Weekend/" + day + "/part-r-00000", "r") as f:
            for line in f:
                lnum += 1
                if (lnum == i or lnum == i + 2 or lnum == i + 4):
                    s = line.split('\t')
                    content += s[1]
                   # content += " "
                if (lnum == i + 6):
                    target = line.split('\t')[1]
                    # 利用map函数将int作用在content.split()中的每一个元素
        result.append(list(map(int, content.split())))

        # mormMat = autoNorm(list(map(int, target.split())))
        result.append(list(map(int, target.split())))
        print(result)
        Tarresult.append(result)

        #  mormMat = autoNorm(Tarresult)
        # print(mormMat)
        # print(Tarresult)
    return Tarresult


def autoNorm(dataSetIn):
    # minVals = dataSet.min(0)
    # maxVals = dataSet.max(0)
    dataSet = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    maxVals = max(dataSetIn)
    minVals = min(dataSetIn)
    ranges = maxVals - minVals
    for i in range(len(dataSetIn)):
        dataSet[i] = float(float(dataSetIn[i] - minVals) / ranges)

    return dataSet, ranges, minVals


def autiNorm(dataSet, ranges, minVals):
    # print(ranges)
    for i in range(len(dataSet)):
        dataSet[i] = int(dataSet[i] * ranges + minVals)
    return dataSet


# fd = file("/home/wtq/BigData-MachineLearning/Data/BusData/10lineBus8th.txt", "r")

# for line in fd.readlines():
#    result.append(list(map(int, line.split(","))))
# print(result)


if __name__ == "__main__":
    l = [1, 2, 3, 4, 5, 6, 7, 8]
    dataSet, ranges, minVal = autoNorm(l)
    print(dataSet)
    autiNo = autiNorm(dataSet, ranges, minVal)
    print(autiNo)
    Read()
