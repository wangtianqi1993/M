# !/usr/bin/env python
# -*-coding: utf-8-*-
__author__ = 'wtq'

from pymongo import MongoClient
from TianchiSongPredict.config import MONGODB_HOST, MONGODB_PORT


def conn_mongo(mongo_host=MONGODB_HOST, mongo_port=MONGODB_PORT):
    """

    :param mongo_host:
    :param mongo_port:
    :return:
    """
    client = MongoClient(mongo_host, mongo_port)
    return client
