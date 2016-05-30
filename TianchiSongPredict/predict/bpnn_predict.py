# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

import csv
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import SupervisedDataSet
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.xml.networkwriter import NetworkWriter
from TianchiSongPredict.data_process.conn_mongo import conn_mongo
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
    ds = SupervisedDataSet(3, 1)

    csvfile = file(file_path, 'a+')
    write = csv.writer(csvfile)
    error = []
    for artist_id in artists:
        date_9 = 20150901
        date_10 = 20151001
        train_data, predict_data = extract_feature(artist_id)
        max_data = get_list_max(train_data, predict_data)

        # 加载训练数据到神经网络中
        for p in train_data:
            inputs = p[0]
            targets = p[1]
            inputs = tuple(map(lambda n: float(n) / max_data, inputs))
            targets = tuple(map(lambda n: float(n) / max_data, targets))
            ds.addSample(inputs, targets)

        # 生成并训练网络
        net = buildNetwork(3, 3, 1)
        trainer = BackpropTrainer(net, ds, verbose=True, learningrate=0.01)
        trainer.trainEpochs(400)
        trainer.trainUntilConvergence(maxEpochs=200)

        # using net to predict
        error_temp = 0
        sign = 1
        input = predict_data[0][0]

        for i in range(0, 60):

            input = list(map(lambda n: float(n) / max_data, input))
            out = net.activate(input)
            # write.writerow([artist_id, item[1][0]-int(out*max_data), item[1][0], out*max_data])
            write.writerow([artist_id, str(int(out*max_data)), str(date_9)])
            input.pop(0)
            input.append(int(out*max_data))
            # error_temp += abs(item[1][0]-int(out*max_data))
            # item[1].append(int(out*max_data))
            # item[1][2] = int(out*max_data)
            # input = item[1]
            # input = tuple(map(lambda n: float(n) / max_data, input))
            # out = net.activate(input)
            # write.writerow([artist_id, str(int(out*max_data)), str(date_10)])
            # print int(out*max_data)
            date_9 += 1
            if date_9 > 20150930 and sign:
                date_9 = 20151001
                sign = 0
            # date_10 += 1
        # error.append(error_temp)

if __name__ == '__main__':
    # bpnn_predict("/home/wtq/mars_tianchi_artist_plays_predict.csv")
    pybrain_train("/home/wtq/mars_tianchi_artist_plays_predict.csv")
