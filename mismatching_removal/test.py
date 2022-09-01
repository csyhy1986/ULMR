import torch
from photo_tourism import photo_tourism as ph_dataset
from torch.utils.data import DataLoader
from LGC_net import LGC_net
import environment as env
import numpy as np
import utils
import cv2
from utils import normalize_correspondences
from numpy.core import where

N = 100

def get_acc(pts1, pts2, mask):
    pts1, pts2 = pts1[:,mask.bool().numpy()].transpose(1,0).numpy(), pts2[:,mask.bool().numpy()].transpose(1,0).numpy()
    n_inlier = pts1.shape[0]
    if n_inlier > 8:
        F, _ = cv2.findFundamentalMat(pts1,pts2,cv2.FM_8POINT)
        homo_pts1, homo_pts2 = np.concatenate((pts1,np.ones((n_inlier,1))),axis = 1).transpose(1,0), \
                                        np.concatenate((pts2,np.ones((n_inlier,1))),axis = 1).transpose(1,0)
        accs = utils.epipolar_error(homo_pts1,homo_pts2,F)
        n = accs.shape[0]
        return accs.mean(), np.sort(accs)[n//2], accs.min(), accs.max()
    else:
        return 9999.9, 9999.9, 9999.9, 9999.9

def get_acc_ransac(pts1, pts2, mask):
    pts1, pts2 = pts1[:,mask.bool().numpy()].transpose(1,0).numpy(), pts2[:,mask.bool().numpy()].transpose(1,0).numpy()
    n_inlier = pts1.shape[0]
    if n_inlier > 8:
        F, mask = cv2.findFundamentalMat(pts1,pts2,cv2.RANSAC,3.0,0.999)
        mask = mask.squeeze()
        pts1, pts2 = pts1[where(mask == 1)], pts2[where(mask == 1)]
        n_inlier = pts1.shape[0]
        homo_pts1, homo_pts2 = np.concatenate((pts1,np.ones((n_inlier,1))),axis = 1).transpose(1,0), \
                                        np.concatenate((pts2,np.ones((n_inlier,1))),axis = 1).transpose(1,0)

        if F is None:
            return 9999.9, 9999.9, 9999.9, 9999.9
        accs = utils.epipolar_error(homo_pts1,homo_pts2,F)
        return accs.mean(), np.sort(accs)[n_inlier//2], accs.min(), accs.max()
    else:
        return 9999.9, 9999.9, 9999.9, 9999.9

def get_pre_r(labels, preds):
    nPt = labels.shape[0]
    n_false = torch.sum(torch.abs(preds-labels))
    pre = (nPt - n_false)/nPt

    com_mask = 1 - torch.abs(labels-preds)
    in_mask = (com_mask.bool() & labels.bool()).float()
    n_in = torch.sum(in_mask)
    out_mask = (com_mask.bool() & (1-labels).bool()).float()
    n_out = torch.sum(out_mask)
    n_lbl_in = torch.sum(labels)
    n_lbl_out = nPt-n_lbl_in
    in_recall, out_recall = n_in/n_lbl_in, n_out/ n_lbl_out

    return pre, in_recall, out_recall


test_data = 'reichstag'
val_dataset   = ph_dataset(test_data)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0,drop_last=True,
                        collate_fn=utils.collate_fn_cat, worker_init_fn=utils.worker_init_fn)
print("Finish loading reichstag data")

model_file = "LFGC_e92.pth"
policy_network = LGC_net()
policy_network.load_state_dict(torch.load(model_file))
policy_network = policy_network.cuda()
policy_network.eval()
print("Successfully loaded model.")

ULMR_file = open("test_ULMR_lfgc.rst", mode='w')
print("Precessing testing data")
with torch.no_grad():
    counter = 0
    for record in val_loader:
        xs = record['xs'].float()
        ys = record['ys'].float()
        imgsz1, imgsz2 = 2*record['cxyf1'][:,0:2], 2*record['cxyf2'][:,0:2]
        nor_pts = normalize_correspondences(xs,imgsz1,imgsz2)
        # for lfgc
        rts = record['rts'].float()
        nor_pts = torch.cat((xs,rts),dim=1)
        logit = policy_network(nor_pts.unsqueeze(-1).cuda())
        probs = torch.softmax(logit,dim=2)
        pts1, pts2 = xs[0, 0:2, :], xs[0, 2:4, :]
        new_state = torch.cat((pts1,pts2),dim=0)
        ULMR_pred = env.get_inliers(probs.squeeze(),new_state, N_HYPO = N, thres = 3.0).float()

        nPt = xs.shape[2]
        gt_geod_d = ys[0,0,:]
        lables = (gt_geod_d.squeeze(0)<0.0001).float()
        inlier_rate = torch.sum(lables)/nPt

        # mean_acc, mid_acc, min_acc, max_acc = get_acc(pts1,pts2,ULMR_pred)
        pre, in_recall, out_recall = get_pre_r(lables,ULMR_pred)

        counter += 1
        line = 'Inlier_Rate: {:6.3f} Presicision: {:8.3f} Inlier_recall: {:8.3f} Outlier_recall: {:8.3f}'.\
        format(counter, len(val_loader), inlier_rate, pre, in_recall, out_recall)
        print(line)
        ULMR_file.write(line + '\n')
ULMR_file.close()
print("Testing results are wtritten in test_ngransac.rst and test_ulmr_ransac.rst")


    

