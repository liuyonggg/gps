from train_data import *

def get_csv(fp):
    tgd = TrainGPS()
    tgd.read_data(fp)
    tgd.calculate_stat()
    r = tgd.generate_csv()
    tgd.calculate_model()
    print tgd.model_score
    assert(tgd.model_score > 0.9)
    return r

def read_csv():
    fn = 'data/shenshuo_gps_no_head.csv'
    fn_out = 'data/shenshuo_gps_no_head.stat.csv'
    with open(fn, 'r') as fp:
        r = get_csv(fp)
        with open(fn_out, 'w') as fp2:
            fp2.write(r)

def plot(key=None):
    tgd = TrainGPS()
    tgd.read_db(num=1000000000000)
    #tgd.read_db(num=1000000)
    tgd.calculate_stat()
    tgd.plot(key)
    fn_out = 'data/1M.stat.csv'
    with open(fn_out, 'w') as fp2:
        fp2.write(tgd.generate_csv())

def hist():
    tgd = TrainGPS()
    tgd.read_db(num=1000000000000)
    tgd.hist()

if __name__ == "__main__":
    #plot('7033B')
    plot()
