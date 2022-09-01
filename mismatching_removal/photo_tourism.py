import torch.utils.data as data
import numpy as np
import os
import pickle

class photo_tourism(data.Dataset):
  def __init__(self, data_names):
    self.data = self.load_data(data_names)
    self.len = len(self.data['xs'])
    # self.batch_size = batch_size
    # self.batch_size = config.batch_size
    
  def load_data(self, data_names):
    print("Loading {} data".format(data_names))

    # Now load data.
    var_name_list = [
        "xs", "ys", "Rs", "ts",
        "cx1s", "cy1s", "f1s",
        "cx2s", "cy2s", "f2s",
        "rts",
    ]
    
    data_dump_prefix = 'data_dump'
    data_folder = data_dump_prefix
    obj_num_kp = 2000
    obj_num_nn = 1

    # Let's unpickle and save data
    data = {}
    data_names = data_names.split(".")
    for data_name in data_names:
        cur_data_folder = "/".join([
            data_folder,
            data_name,
            "numkp-{}".format(obj_num_kp),
            "nn-{}".format(obj_num_nn),
        ])
        cur_data_folder = os.path.join(cur_data_folder, "nocrop")
        suffix = "tr-10000"
        cur_folder = os.path.join(cur_data_folder, suffix)
        ready_file = os.path.join(cur_folder, "ready")
        if not os.path.exists(ready_file):
            # data_gen_lock.unlock()
            raise RuntimeError("Data is not prepared!")

        for var_name in var_name_list:
            cur_var_name = var_name
            in_file_name = os.path.join(cur_folder, cur_var_name) + "_tr.pkl"
            with open(in_file_name, "rb") as ifp:
                if var_name in data:
                    data[var_name] += pickle.load(ifp)
                else:
                    data[var_name] = pickle.load(ifp)

    return data

  # def LoadImage(self, PATH,depth=True):
  #   if depth:
  #     img = cv2.imread(PATH,2)/1000.
  #   else:
  #     img = cv2.imread(PATH)
  #   return img
  
  def shuffle(self):
    pass

  def __getitem__(self, index):
    class Record(object):
        pass
    record = Record()
    ind_cur = [index]
    numkps = np.array([self.data['xs'][_i].shape[1] for _i in ind_cur])
    cur_num_kp = numkps.min()
    # Actual construction of the batch
    xs_b = np.array(
        [self.data['xs'][_i][:, :cur_num_kp, :] for _i in ind_cur]
    ).reshape(1, cur_num_kp, 4).transpose(0,2,1)
    ys_b = np.array(
        [self.data['ys'][_i][:cur_num_kp, :] for _i in ind_cur]
    ).reshape(1, cur_num_kp, 2).transpose(0,2,1)
    Rs_b = np.array(
        [self.data['Rs'][_i] for _i in ind_cur]
    ).reshape(1, 9)
    ts_b = np.array(
        [self.data['ts'][_i] for _i in ind_cur]
    ).reshape(1, 3)
    cx1s_b = np.array(
        [self.data['cx1s'][_i] for _i in ind_cur]
    )
    cy1s_b = np.array(
        [self.data['cy1s'][_i] for _i in ind_cur]
    )
    f1s_b = np.array(
        [self.data['f1s'][_i] for _i in ind_cur]
    )
    cx2s_b = np.array(
        [self.data['cx2s'][_i] for _i in ind_cur]
    )
    cy2s_b = np.array(
        [self.data['cy2s'][_i] for _i in ind_cur]
    )
    f2s_b = np.array(
        [self.data['f2s'][_i] for _i in ind_cur]
    )
    rts_b = np.array(
        [self.data['rts'][_i] for _i in ind_cur]
    ).reshape(1, cur_num_kp, 1).transpose(0,2,1)
    record.xs = xs_b
    record.ys = ys_b
    record.Rs = Rs_b
    record.ts = ts_b
    record.cxyf1 = np.array([cx1s_b, cy1s_b, f1s_b]).reshape(1, 3)
    record.cxyf2 = np.array([cx2s_b, cy2s_b, f2s_b]).reshape(1, 3)
    record.rts = rts_b
    return record
    
  def __len__(self):
    return self.len

    # for i in range(14723):
    #   m=data_loader.dataset.data['xs'][i].max()
    #   if m>5: print(m)                                                                            


