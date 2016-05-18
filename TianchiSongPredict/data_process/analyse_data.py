# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

import matplotlib.pyplot as plt
from conn_mongo import conn_mongo

client = conn_mongo()
db = client.tianchi


def analyse_data():
    """

    :return:
    """
    play_times = []
    download_times = []
    collection_times = []
    x_label = []
    for i in range(1, 32):
        x_label.append(i)
    x = x_label

    for item in db.artist_times.find({"artist_id": "be0c7a23c2aa9afb45163995b9ec938c"}).sort("date"):
        play_times.append(item['play_times'])
        download_times.append(item['download_times'])
        collection_times.append(item['collection_times'])
    y3 = play_times[:31]
    y4 = play_times[31:61]
    y5 = play_times[61:92]
    y6 = play_times[92:122]
    y7 = play_times[122:153]
    y8 = play_times[153:183]

    plt.plot(y3, 'b')
    plt.plot(y4, 'm')
    plt.plot(y5, 'c')
    plt.plot(y6, 'r')
    plt.plot(y7, 'y')
    plt.plot(y8, 'k')
    # plt.plot(play_times, 'k')
    plt.xlabel("day")
    plt.ylabel("play-times")
    plt.title("be0c7a23c2aa9afb45163995b9ec938c")
    plt.legend(['3m', '4m', '5m', '6m', '7m', '8m'])
    plt.show()

if __name__ == "__main__":
    analyse_data()

