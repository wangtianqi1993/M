# !/usr/bin/env python
# -*- coding:utf-8 -*-
__author__ = 'wtq'

import time
import traceback
from bson.code import Code
from conn_mongo import conn_mongo
from detector.logger import DetectorLogger

client = conn_mongo()
db = client.tianchi
logger = DetectorLogger()

def write_song_mongo(file_path):
    """
    save data to mongo
    :param file_path:
    :return:
    """
    start_time = time.clock()

    collection = db.mars_song
    try:
        with open(file_path) as input:
            for line in input:
                line = line.strip('\n')
                words = line.split(",")
                items = {
                    "song_id": words[0],
                    "artist_id": words[1],
                    "publish_time": words[2],
                    "init_play_times": words[3],
                    "language": words[4],
                    "gender": words[5]
                }
                collection.insert(items)

    except Exception, e:
        traceback.print_exc()
    end_time = time.clock()
    return end_time - start_time


def write_action_mongo(file_path):
    """

    :param filepath:
    :return:
    """
    start_time = time.clock()

    collection = db.mars_user_action
    try:
        with open(file_path) as input:
            for line in input:
                line = line.strip('\n')
                words = line.split(",")
                items = {
                    "user_id": words[0],
                    "song_id": words[1],
                    "gmt_create": words[2],
                    "action_type": words[3],
                    "action_date": words[4]
                }
                collection.insert(items)

    except Exception, e:
        traceback.print_exc()
    end_time = time.clock()
    return end_time - start_time


def combine_db():
    """
    combine two database by song_id
    :return:
    """
    # actions = db.mars_user_action.find()
    # songs = db.mars_song.find()
    songs = []
    for item in db.mars_song.find():
        songs.append(item)

    for action in db.mars_user_action.find():
        # logger.info(action)
        items = {
            "song_id": db.mars_song.find_one({"song_id": action["song_id"]})['song_id'],
            "publish_time": db.mars_song.find_one({"song_id": action["song_id"]})['publish_time'],
            "language": db.mars_song.find_one({"song_id": action["song_id"]})['language'],
            "gender": db.mars_song.find_one({"song_id": action["song_id"]})['gender'],
            "init_play_times": db.mars_song.find_one({"song_id": action["song_id"]})['init_play_times'],
            "artist_id": db.mars_song.find_one({"song_id": action["song_id"]})['artist_id'],
            "user_id": action['user_id'],
            "gmt_create": action['gmt_create'],
            "action_date": action['action_date'],
            "action_type": action['action_type']
        }
        db.combine_collection.insert(items)



def count_times():
    """

    :return:
    """
    start_cpu = time.clock()
    collection = db.mars_user_action
    date = collection.distinct("action_date")
    try:
        maps = Code("""function(){emit(this.song_id, {"action_type":this.action_type,"action_date":this.action_date});}""")

        reduces = Code("""function(key, values){

                        for(var play_times = [], n = 0; n < 188; play_times[n++] = 0);
                        for(var download_times = [], n = 0; n < 188; download_times[n++] = 0);
                        for(var collection_times = [], n = 0; n < 188; collection_times[n++] = 0);
                        var date = new Array();
                        for(var j=0; j<scope.length;j++)
                            {
                                date[j]=scope[j];
                                for(var i=0; i<values.length; i++)
                                {
                                    if(values[i].action_date == scope[j])
                                    {
                                        if(values[i].action_type == "1")
                                            play_times[j]+=1;
                                        else if(values[i].action_type == "2")
                                            download_times[j]+=1;
                                        else
                                            collection_times+=1;
                                    }
                                }

                            }
                        var ret={song_id:key,dates:date,play:play_times,download:download_times,collection:collection_times};
                        return ret;
                        ｝
                  """)
        db.mars_user_action.map_reduce(maps, reduces, out="song_times", scope=date)
    except Exception, e:
        traceback.print_exc()


def count_times_new():
    """

    :return:
    """
    start_cpu = time.clock()
    collection = db.mars_user_action

    artists = db.mars_song.distinct("artist_id")
    print len(artists)
    dates = collection.distinct("action_date")
    print len(dates)
    for artist in artists:
        for date in dates:
            play_times = db.combine_collection.find({"artist_id":artist, "action_date":date, "action_type": "1"}).count()
            download_times = db.combine_collection.find({"artist_id":artist, "action_date":date, "action_type": "2"}).count()
            collection_times = db.combine_collection.find({"artist_id":artist, "action_date":date, "action_type": "3"}).count()

            items = {
                "artist_id": artist,
                "date": date,
                "play_times": play_times,
                "download_times": download_times,
                "collection_times": collection_times
            }
            db.artist_times.insert(items)
    end = time.clock()
    return end-start_cpu

if __name__ == '__main__':
   # print write_action_mongo("/media/wtq/0001CADE0007CDD4/BigData-MachineLearning/Computation-Data/song-predict/mars_tianchi_user_actions.csv")
   print count_times_new()
   # combine_db()
