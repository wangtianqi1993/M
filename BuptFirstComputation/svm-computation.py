# !/usr/bin/env python
# -*- coding: utf-8
__author__ = 'wtq'

from numpy import *
import csv
from sklearn import svm
from sklearn.naive_bayes import MultinomialNB

def get_max_index(my_list):
    max = my_list[0]
    for i in my_list:
        if i > max:
            max = i
    return my_list.index(max)


def load_train_data():
    train_data = []
    train_label = []
    temp = []
    cat_id = '1206'
    brand_id = '4241'
    # 注意先删除文件的前两行
    with open('/home/wtq/BigData-MachineLearning/Data/'
              'bupt-computation/log_newtrain.csv') as file:
        next(file)
        lines = csv.reader(file)
        for line in lines:
            if cat_id != line[2] or brand_id != line[4]:
                cat_id = line[2]
                brand_id = line[4]
                temp.append(int(line[2]))
                temp.append(int(line[4]))
                train_data.append(temp)
                train_label.append(int(line[7]))
                temp = temp[1:1]
    file.close()
    return train_data, train_label


def load_test_data():
    test_dict = {}
    test_data = []
    temp = []
    user_id = '2702'
    # 注意先删除文件的前两行
    with open('/home/wtq/BigData-MachineLearning/Data/'
              'bupt-computation/log_test2.csv') as file:
        next(file)
        lines = csv.reader(file)
        for line in lines:
            user_id_new = line[0]
            if not cmp(user_id, user_id_new):
                temp.append(int(line[2]))
                temp.append(int(line[4]))
                test_data.append(temp)
                temp = temp[1:1]
            if cmp(user_id, user_id_new):
                test_dict[user_id] = test_data
                user_id = user_id_new
                test_data = test_data[1:1]

    return test_dict

if __name__ == '__main__':

    # train classifier
    file = open('/home/wtq/BigData-MachineLearning/Data/'
              'bupt-computation/test/predict2.csv', 'wb')
   # clf = svm.SVC(decision_function_shape='ovo')
    clf = MultinomialNB()
    customer_class = []
    train_data, train_label = load_train_data()
    test_dict = load_test_data()
    clf.fit(train_data, train_label)
    writer = csv.writer(file, delimiter=',', quoting=csv.QUOTE_NONE)
    for (key, value) in test_dict.items():
        customer_class = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        for i in range(len(test_dict[key])):
            customer_class[int(clf.predict([test_dict[key][i]])[0])] += 1
        writer.writerow([key, get_max_index(customer_class)])
        # print customer_class
    print "end!"
    file.close()



