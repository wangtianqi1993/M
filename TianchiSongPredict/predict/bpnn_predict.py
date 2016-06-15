# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from TianchiSongPredict.data_process.conn_mongo import conn_mongo
from TianchiSongPredict.data_process.standardization import standardization
from TianchiSongPredict.data_process.normailization import get_list_max
from TianchiSongPredict.data_process.extract_feature import extract_feature
from TianchiSongPredict.predict_algorithm.bpnn import NN

client = conn_mongo()
db = client.tianchi


def bpnn_predict(file_path):
    """
    use the bp neural network to predict artist's play times during two month ,then save into file
    :return:
    """
    artists = db.mars_song.distinct("artist_id")

    for artist_id in artists:
        train_data, predict_data = extract_feature(artist_id)

        network = NN(3, 3, 1)
        network.train(train_data)
        date_9 = 20150901
        date_10 = 20151001

        csvfile = file(file_path, 'a+')
        write = csv.writer(csvfile)

        for item in predict_data:

            play_time1 = network.update(item[0])[0]
            write.writerow([artist_id, str(int(play_time1)), str(date_9)])
            item[1].append(play_time1)
            play_time2 = network.update(item[1])[0]
            write.writerow([artist_id, str(int(play_time2)), str(date_10)])
            date_9 += 1
            date_10 += 1


def pybrain_train(file_path):
    """
    using pybrain to train and predict the song play times
    :return:
    """
    artists = db.mars_song.distinct("artist_id")

    csvfile = file(file_path, 'a+')
    write = csv.writer(csvfile)

    for artist_id in artists:

        # 每一个艺人都要重新设计pybrain的数据，防止下一个艺人数据的叠加
        ds = SupervisedDataSet(3, 1)
        date_9 = 20150901
        date_10 = 20151001
        train_data, predict_data = extract_feature(artist_id)

        max_data = get_list_max(train_data, predict_data)

        # x_mat y_mat 为训练时使用的数据与标签 p_9与p_10为预测9月与10月所用的样本
        x_mat = []
        y_mat = []
        p_9 = []
        p_10 = []

        # 数据标准化
        for k in train_data:
            x_mat.append(k[0])
            y_mat.append(k[1])
        x_mat, y_mat = standardization(x_mat, y_mat)

        # 加载训练数据到神经网络中
        for index in range(0, len(x_mat)):
            inputs = x_mat[index]
            targets = tuple(map(lambda n: float(n) / max_data, y_mat[index]))
            ds.addSample(inputs, targets)

        net = buildNetwork(3, 5, 1)
        trainer = BackpropTrainer(net, ds, verbose=True, learningrate=0.01)
        trainer.trainEpochs(800)
        trainer.trainUntilConvergence(maxEpochs=800)

        # using net to predict
        for item in predict_data:
            p_9.append(item[0])
            p_10.append(item[1])

        # 标准化预测值（减去均值，除以方差）
        p_9, pt = standardization(p_9)
        p_10, pt = standardization(p_10)

        for i in range(0, 30):
            input_9 = p_9[i]
            input_10 = p_10[i]
            out_9 = net.activate(input_9)
            write.writerow([artist_id, str(int(out_9*max_data)), str(date_9)])
            out_10 = net.activate(input_10)
            write.writerow([artist_id, str(int(out_10*max_data)), str(date_10)])
            date_9 += 1
            date_10 += 1


if __name__ == '__main__':
    # bpnn_predict("/home/wtq/mars_tianchi_artist_plays_predict.csv")
    # pybrain_train("/home/wtq/mars_tianchi_artist_plays_predict_end.csv")

    # print groupAnagrams(["aa", "ab", "ba"])
    def test():
        vv = ['ssssdfsas', 'fgg', 'sd', 'ds']
        words = {}

        for i in vv:

            temp = tuple(sorted(i))

            if temp not in words:
                words[temp] = []

            words[temp].append(i)
        for j in words:
            print words[j]
    test()
