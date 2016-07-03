import unittest
import StringIO
import MySQLdb

from train_data import *
from sklearn.metrics import accuracy_score

class TestTraingGPS(unittest.TestCase):
    def setUp(self):
        pass

    def test_read_data(self):
        data = """\
7017,207,652A,60480,110.4409624468,39.0602087196,987.0083533302,2016/6/22 18:59:16,1,11111,0,2016/6/22 18:59:18
7027,207,652A,60480,110.4409630008,39.06020866,986.9847701034,2016/6/22 18:59:18,1,11111,0,2016/6/22 18:59:18
7031,207,653B,60452,110.4407427616,39.0607446034,986.3234268604,2016/6/22 18:59:18,1,11111,0,2016/6/22 18:59:18
7036,207,652A,60480,110.4409637713,39.0602087705,986.985514597,2016/6/22 18:59:20,1,11111,0,2016/6/22 18:59:18
7043,207,653B,60452,110.4407431139,39.0607446275,986.3677299526,2016/6/22 18:59:20,1,11111,0,2016/6/22 18:59:18
7056,207,652A,60480,110.4409643932,39.0602088477,986.9908231935,2016/6/22 18:59:22,1,11111,0,2016/6/22 18:59:18
7057,207,653B,60452,110.4407434259,39.0607445286,986.4130074708,2016/6/22 18:59:22,1,11111,0,2016/6/22 18:59:18
7067,207,653B,60452,110.4407436424,39.0607444889,986.4184309933,2016/6/22 18:59:24,1,11111,0,2016/6/22 18:59:18
7068,207,652A,60480,110.4409648389,39.0602090177,986.9834322613,2016/6/22 18:59:24,1,11111,0,2016/6/22 18:59:18
7076,207,652A,60480,110.4409653251,39.0602094507,986.9876695201,2016/6/22 18:59:26,1,11111,0,2016/6/22 18:59:18
7078,207,653B,60452,110.440744035,39.0607444264,986.4428553404,2016/6/22 18:59:26,1,11111,0,2016/6/22 18:59:18
7086,207,652A,60480,110.4409658112,39.0602098138,986.9856054122,2016/6/22 18:59:28,1,11111,0,2016/6/22 18:59:18
7093,207,653B,60452,110.4407444277,39.0607445037,986.4798825681,2016/6/22 18:59:28,1,11111,0,2016/6/22 18:59:18
7099,207,652A,60480,110.4409662294,39.0602104596,986.992457686,2016/6/22 18:59:30,1,11111,0,2016/6/22 18:59:18"""
        f = StringIO.StringIO()
        f.write(data)
        f.flush()
        f.seek(0)
        tgd = TrainGPS()
        tgd.read_data(f)
        self.assertEqual(set(tgd.data.keys()), set(['652A', '653B']))
        self.assertEqual(tgd.data['652A'].data[0].shape, (8, 2))
        self.assertEqual(tgd.data['653B'].data[0].shape, (6, 2))
        tgd.calculate_stat()
        self.assertEqual(tgd.data['652A'].stat[0][0], 8)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][1], 110.44096448)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][2], 39.06020922)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][3], 1.25035765e-06)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][4], 6.00828602e-07)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][5], 110.44096245)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][6], 39.06020866)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][7], 110.44096623)
        self.assertAlmostEqual(tgd.data['652A'].stat[0][8], 39.06021046)
        self.assertEqual(tgd.data['653B'].stat[0][0], 6)
        r = tgd.generate_csv()
        exp_r = """\
652A, 2016-06-22 18:59:16, 8, 110.4409644770875, 39.06020921745, 1.2503576479057145e-06, 6.008286016269832e-07, 110.4409624468, 39.06020866, 110.4409662294, 39.0602104596
653B, 2016-06-22 18:59:18, 6, 110.44074356775, 39.060744529749996, 5.536828568807164e-07, 6.832429494341883e-08, 110.4407427616, 39.0607444264, 110.4407444277, 39.0607446275
"""
        self.assertEqual(r, exp_r)
        tgd.calculate_model()
        self.assertAlmostEqual(tgd.model_score, 1.)

    def test_basic_read_db(self):
        con = MySQLdb.Connect(host='localhost', user='gpsdata', passwd='gpsdata', db='train')
        cur = con.cursor()
        cur.execute("SELECT * FROM terminal_jwd where iskn = 1 and speed = 0 LIMIT 10 ")
        data = cur.fetchall()
        self.assertEqual(len(data), 10)

    def test_read_db(self):
        tgd = TrainGPS()
        tgd.read_db(num=100000)
        self.assertTrue(tgd.data)
        tgd.calculate_stat()
        r = tgd.generate_csv()
        assert (r)
        tgd.calculate_model()
        self.assertGreater(tgd.model_score, 0.9)

    def test_plot(self):
        tgd = TrainGPS()
        tgd.read_db(num=100)
        tgd.plot()
        
if __name__ == '__main__':
    unittest.main()
