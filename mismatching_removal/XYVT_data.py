from fileinput import close
import torch.utils.data as data
import numpy as np
import struct

class xyvt_data(data.Dataset):
  def __init__(self, data_names):
    self.data = self.load_data(data_names)
    self.len = len(self.data['xs1'])
    # self.batch_size = batch_size
    # self.batch_size = config.batch_size
    
  def load_data(self, data_names):
    print("Loading {} data".format(data_names))
    
    data_dump_prefix = '..\\learning_data\\XYVT\\'

    # Let's unpickle and save data
    data = {}
    infos, xs1, xs2, ds, epis, grds = [],[],[],[],[],[]
    data_names = data_names.split(".")
    for data_name in data_names:
        cur_data_folder = data_dump_prefix + data_name

        data_path = cur_data_folder + "\\info-xs-drs-epi-grd.dat"
        data_file = open(data_path,'rb')
        buffer = data_file.read(4)
        nPairs = struct.unpack('i',buffer)
        for _ in range(nPairs[0]):
            buffer = data_file.read(5*4)
            info = struct.unpack(5*'i',buffer) #number of points, w1, h1, w2, h2
            nPts = info[0]
            buffer = data_file.read(2*nPts*4)
            x1 = np.array(struct.unpack(2*nPts*'f',buffer)).reshape((-1,2))# matched points in left image
            buffer = data_file.read(2*nPts*4)
            x2 = np.array(struct.unpack(2*nPts*'f',buffer)).reshape((-1,2))# matched points in right image
            buffer = data_file.read(nPts*4)
            d = np.array(struct.unpack(nPts*'f',buffer))# ratio of distances
            buffer = data_file.read(nPts*4)
            epi = np.array(struct.unpack(nPts*'f',buffer))# epipolar error 
            buffer = data_file.read(nPts*4)
            grd = np.array(struct.unpack(nPts*'f',buffer))# ground error 

            infos += [info[1:5]]
            xs1 += [x1]
            xs2 += [x2]
            ds += [d]
            epis += [epi]
            grds += [grd]
        data_file.close()
    data['infos'] = infos
    data['xs1'] = xs1
    data['xs2'] = xs2
    data['ds'] = ds
    data['epis'] = epis
    data['grds'] = grds

    return data
  
  def shuffle(self):
    pass

  def __getitem__(self, index):
    class Record(object):
        pass
    rcd = Record()
    rcd.infos = np.array(self.data['infos'][index])
    rcd.xs1 = np.array(self.data['xs1'][index])
    rcd.xs2 = np.array(self.data['xs2'][index])
    rcd.ds = np.array(self.data['ds'][index])
    rcd.epis = np.array(self.data['epis'][index])
    rcd.grds = np.array(self.data['grds'][index])
    return rcd
    
  def __len__(self):
    return self.len                                                                          


