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

