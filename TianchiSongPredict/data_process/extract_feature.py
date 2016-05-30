# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

from conn_mongo import conn_mongo

client = conn_mongo()
db = client.tianchi


def extract_feature(artist_id):
    """
    采用滑动窗口模式来提取特征，[[3.1, 4.1, 5.1],[6.1]]... [[4.1,5.1,6.1][7.1]]
    :return:
    """
    play_times = []
    predict_times = []
    download_times = []
    feature = []
    for item in db.artist_times.find({"artist_id": artist_id}).sort("date"):
        play_times.append(item['play_times'])
        download_times.append(item['download_times'])

    # remove 3.31 5.31 7.31
    play_times.pop(30)
    play_times.pop(90)
    play_times.pop(150)
    download_times.pop(30)
    download_times.pop(90)
    download_times.pop(150)

    for i in range(0, 178):
        if i < 177:
            #train_data = [play_times[i], play_times[i+30], play_times[i+60], download_times[i],
                          #download_times[i+30], download_times[i+60]]
            # train_data = [play_times[i], play_times[i+30], play_times[i+60]]
            train_data = [play_times[i], play_times[i+1], play_times[i+2]]
            # label = [play_times[i+90]]
            label = [play_times[i+3]]
            item = [train_data, label]
            feature.append(item)
        else:
            # predict_data = [play_times[i], play_times[i+30], play_times[i+60], download_times[i],
            #               download_times[i+30], download_times[i+60]]
            # predict_data = [play_times[i], play_times[i+30], play_times[i+60]]
            predict_data = [play_times[i], play_times[i+1], play_times[i+2]]
            # label = [play_times[i+90]]
            # label = [play_times[i+3]]
            item = [predict_data]
            predict_times.append(item)

        # 生成预测样本的输入predict_times[0]对应9月，predict_times[1]对应10月
        # if i < 30:
        #     predict_data1 = [play_times[i+90], play_times[i+120], play_times[i+150], download_times[i+90],
        #                      download_times[i+120], download_times[i+150]]
        #
        #     predict_data2 = [play_times[i+120], play_times[i+150], 0, download_times[i+120], download_times[i+150], download_times[i+150]]
        #     temp = [predict_data1, predict_data2]
        #     predict_times.append(temp)

    return feature, predict_times


if __name__ == '__main__':
    print extract_feature("f6e0f05fde7637afb8f8bc6bda74ca24")

