import csv
import numpy as np
import MySQLdb
import matplotlib.pyplot as plt
import math
import datetime

from sklearn import svm
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

class TrainGPSData(object):
    def __init__(self, train_id):
        self.train_id = train_id
        # np.array of data, each row is <longitude, latitude>
        self.data = []
        # time list, each time is corresponding to an elemement in data
        self.time = []
        # list of data stat, each row is <num, mean, std, min, max>]}
        self.stat = []

    def __hash__(self):
        return self.train_id

    def need_new_array(self, time):
        res = False
        if not self.time:
            self.time.append(time)
            res = True
        assert (len(self.time))
        d = time - self.time[-1]
        s = (d.days * 24 * 3600 + d.seconds)
        #assert (s >= 0)
        # if time difference is less than 2 hours, it will be treated as same stop, otherwise new stop
        if (abs(s) > 7200):
            self.time.append(time)
            res = True
        return res

    def add_data(self, row):
        '''
        field in the row is defined as the following 
        +-------------+-------------+------+-----+---------+----------------+
        | Field       | Type        | Null | Key | Default | Extra          |
        +-------------+-------------+------+-----+---------+----------------+
        | id          | bigint(11)  | NO   | PRI | NULL    | auto_increment |
        | traintypeid | int(5)      | YES  |     | NULL    |                |
        | trainid     | varchar(10) | YES  |     | NULL    |                |
        | terminalid  | int(7)      | YES  |     | NULL    |                |
        | lon         | double      | YES  |     | NULL    |                |
        | lat         | double      | YES  |     | NULL    |                |
        | alt         | double      | YES  |     | NULL    |                |
        | time        | datetime    | YES  |     | NULL    |                |
        | iskn        | int(1)      | YES  |     | NULL    |                |
        | trainorder  | varchar(10) | YES  |     | NULL    |                |
        | speed       | int(3)      | YES  |     | NULL    |                |
        | inserttime  | datetime    | YES  |     | NULL    |                |
        +-------------+-------------+------+-----+---------+----------------+
        '''
        assert (row[2] == self.train_id)
        t = None
        if type(row[7]) == datetime.datetime:
            t = row[7]
        else:
            assert (type(row[7]) == str)
            t = datetime.datetime.strptime(row[7], '%Y/%m/%d %H:%M:%S')
        if self.need_new_array(t):
            self.data.append(np.array([[float(row[4]), float(row[5])]]))
        else:
            self.data[-1] = np.append(self.data[-1], [[float(row[4]), float(row[5])]], axis=0)
        assert (len(self.data) == len(self.time))

    def calculate_stat(self):
        assert(self.data)
        assert (not self.stat)
        for d in self.data:
            a = [d.shape[0]]
            a += np.mean(d, axis=0).tolist()
            a += np.std(d, axis=0).tolist()
            a += np.min(d, axis=0).tolist()
            a += np.max(d, axis=0).tolist()
            self.stat.append(a)
        assert (len(self.stat) == len(self.data))

    def print_stat(self):
        assert (self.stat)
        res = ""
        i = 0
        for k in self.stat:
            res += self.train_id + ", " + str(self.time[i]) + ", " + str(k).strip('[]')
            res += "\n"
            i += 1
        return res


class TrainGPS(object):
    def __init__(self):
        # {"train_id" : TrainGPSData}
        self.data = {}
        # model is svm model
        self.model = None
        self.model_input_X = None
        self.model_input_y = None
        self.model_score = 0

    def add_data(self, row):
        '''
        +-------------+-------------+------+-----+---------+----------------+
        | Field       | Type        | Null | Key | Default | Extra          |
        +-------------+-------------+------+-----+---------+----------------+
        | id          | bigint(11)  | NO   | PRI | NULL    | auto_increment |
        | traintypeid | int(5)      | YES  |     | NULL    |                |
        | trainid     | varchar(10) | YES  |     | NULL    |                |
        | terminalid  | int(7)      | YES  |     | NULL    |                |
        | lon         | double      | YES  |     | NULL    |                |
        | lat         | double      | YES  |     | NULL    |                |
        | alt         | double      | YES  |     | NULL    |                |
        | time        | datetime    | YES  |     | NULL    |                |
        | iskn        | int(1)      | YES  |     | NULL    |                |
        | trainorder  | varchar(10) | YES  |     | NULL    |                |
        | speed       | int(3)      | YES  |     | NULL    |                |
        | inserttime  | datetime    | YES  |     | NULL    |                |
        +-------------+-------------+------+-----+---------+----------------+
        '''
        data = None
        if row[2] in self.data:
            data = self.data[row[2]]
        else:
            data = TrainGPSData(row[2])
            self.data[row[2]] = data
        data.add_data(row)

    def read_data(self, f):
        reader = csv.reader(f, delimiter=',')
        for row in reader:
            # in station and speed is 0
            if int(row[8]) == 1 and int(row[10]) == 0:
                self.add_data(row)

    def read_db(self, num=100000):
        '''
        mysql> SHOW COLUMNS FROM terminal_jwd;
        +-------------+-------------+------+-----+---------+----------------+
        | Field       | Type        | Null | Key | Default | Extra          |
        +-------------+-------------+------+-----+---------+----------------+
        | id          | bigint(11)  | NO   | PRI | NULL    | auto_increment |
        | traintypeid | int(5)      | YES  |     | NULL    |                |
        | trainid     | varchar(10) | YES  |     | NULL    |                |
        | terminalid  | int(7)      | YES  |     | NULL    |                |
        | lon         | double      | YES  |     | NULL    |                |
        | lat         | double      | YES  |     | NULL    |                |
        | alt         | double      | YES  |     | NULL    |                |
        | time        | datetime    | YES  |     | NULL    |                |
        | iskn        | int(1)      | YES  |     | NULL    |                |
        | trainorder  | varchar(10) | YES  |     | NULL    |                |
        | speed       | int(3)      | YES  |     | NULL    |                |
        | inserttime  | datetime    | YES  |     | NULL    |                |
        +-------------+-------------+------+-----+---------+----------------+
        '''
        con = MySQLdb.Connect(host='localhost', user='gpsdata', passwd='gpsdata', db='train')
        cur = con.cursor()
        cur.execute("SELECT * FROM terminal_jwd where iskn = 1 and speed = 0 LIMIT %d" % num)
        #cur.execute("SELECT * FROM terminal_jwd where iskn=1 and speed < 5 LIMIT %d" % num)
        #cur.execute("SELECT * FROM terminal_jwd where iskn = 1 and speed = 0 and trainid = '7033B' LIMIT %d" % num)
        for row in cur.fetchall():
            self.add_data(row)

    def calculate_stat(self):
        for v in self.data.values():
            v.calculate_stat()

    def generate_csv(self):
        res = ""
        for v in self.data.values():
            res += v.print_stat()
        return res

    def calculate_model(self):
        assert (not self.model)
        clf = svm.SVC()
        self.model_input_X = []
        self.model_input_y = []
        i = 0
        for k in self.data.keys():
            for v in self.data[k].data:
                for t in v:
                    self.model_input_X.append(t)
                    self.model_input_y.append(i)
            i += 1
        self.model_input_X = preprocessing.scale(self.model_input_X)
        self.model = clf.fit(self.model_input_X, self.model_input_y)  
        y_pred = self.model.predict(self.model_input_X)
        self.model_score = accuracy_score(self.model_input_y, y_pred)

    def plot(self, key=None):
        x = []
        y = []
        title = None
        label_cnt = 0
        label_max = 20
        keys = [key] if key else self.data.keys()
        for k in keys:
            i = 0
            for d in self.data[k].data:
                x = []
                y = []
                for r in d:
                    x.append(r[0])
                    y.append(r[1])
                label = None
                if label_cnt < label_max:
                    label = k + " " + str(self.data[k].time[i])
                plt.plot(x, y, '.', label=label)
                label_cnt += 1
                i += 1

        #x = []
        #y = []
        #label_cnt = 0
        #for k in keys:
        #    i = 0
        #    if self.data[k].stat:
        #        v = self.data[k].stat
        #        for d in v:
        #            x = []
        #            y = []
        #            x.append(d[1])
        #            y.append(d[2])
        #            label = None
        #            if label_cnt < label_max:
        #                label = "mean " + k + " " + str(self.data[k].time[i])
        #            plt.plot(x, y, 'o', label=label)
        #            label_cnt += 1
        #            i += 1
        plt.legend()
        plt.show()

    def hist(self):
        g = int(math.sqrt(len(self.data.keys()))) + 1
        f, axs = plt.subplots(g, g)
        i = 0
        for key in self.data.keys():
            lon = []
            lat = []
            for r in self.data[key]:
                lon.append(r[0])
                lat.append(r[1])
            lon = preprocessing.scale(lon)
            lat = preprocessing.scale(lat)
            #axs[int((1.0*i)/r)][i%r].hist(lon)
            r = int((1.0*i)/g)
            c = i%g
            axs[r][c].hist(lon, histtype='step')
            axs[r][c].hist(lat, histtype='step')
            axs[r][c].set_title(key)
            i += 1
        plt.show()
