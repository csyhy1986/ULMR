import numpy as np
import torch
import random
import os
import collections

numpy_type_map = {
     'float64': torch.DoubleTensor,
     'float32': torch.FloatTensor,
     'float16': torch.HalfTensor,
     'int64': torch.LongTensor,
     'int32': torch.IntTensor,
     'int16': torch.ShortTensor,
     'int8': torch.CharTensor,
     'uint8': torch.ByteTensor,
 }

def worker_init_fn(worker_id):                                                          
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def collate_fn_cat_xyvt(batch):
  "Puts each data field into a tensor with outer dimension batch size"

  infos, xs1, xs2, ds = [], [], [], []
  nB = len(batch) 
  length = np.array([batch[i].xs1.shape[0] for i in range(nB)])
  if np.any(length != length[0]):
    min_len = np.min(length)
    for i in range(nB):
      infos.append(batch[i].infos[:])
      xs1.append(batch[i].xs1[:min_len,:])
      xs2.append(batch[i].xs2[:min_len,:])
      ds.append(batch[i].ds[:min_len])
  infos, xs1, xs2, ds = np.stack(infos), np.stack(xs1), np.stack(xs2), np.stack(ds)
  return (torch.from_numpy(infos), torch.from_numpy(xs1), 
          torch.from_numpy(xs2),   torch.from_numpy(ds))

def collate_fn_cat_phtm(batch):
  "Puts each data field into a tensor with outer dimension batch size"
  string_classes = (str, bytes)
  
  # truncate first since the each example in the batch 
  # may have different length
  if type(batch[0]).__name__ == 'Record':
    n = len(batch)
    length = np.array([batch[i].xs.shape[2] for i in range(n)])
    if np.any(length != length[0]):
      min_len = np.min(length)
      keys = batch[0].__dict__.keys()
      for i in range(n):
        batch[i].xs = batch[i].xs[:,:,:min_len]
        batch[i].ys = batch[i].ys[:,:,:min_len]
        batch[i].rts = batch[i].rts[:,:,:min_len]
  
  if torch.is_tensor(batch[0]):
    out = None
    return torch.cat(batch, 0, out=out)
    # for rnn variable length input
  elif type(batch[0]).__module__ == 'numpy':
    elem = batch[0]
    if type(elem).__name__ == 'ndarray':
      try:
        torch.cat([torch.from_numpy(b) for b in batch], 0)
      except:
        import ipdb;ipdb.set_trace()
      return torch.cat([torch.from_numpy(b) for b in batch], 0)
    if elem.shape == ():  # scalars
      py_type = float if elem.dtype.name.startswith('float') else int
      return numpy_type_map[elem.dtype.name](list(map(py_type, batch)))
  elif isinstance(batch[0], int):
    return torch.LongTensor(batch)
  elif isinstance(batch[0], float):
    return torch.DoubleTensor(batch)
  elif isinstance(batch[0], string_classes):
    return batch
  elif isinstance(batch[0], collections.Mapping):
    return {key: collate_fn_cat_phtm([d[key] for d in batch]) for key in batch[0]}
  elif isinstance(batch[0], collections.Sequence):
    transposed = zip(*batch)
    return [collate_fn_cat_phtm(samples) for samples in transposed]
  elif isinstance(batch[0], object):
    return {key: collate_fn_cat_phtm([getattr(d,key) for d in batch]) for key in batch[0].__dict__.keys()}

def export_dpt_file(image_height1, image_height2, kps1, kps2, matches1to2, matchesMask, out_file):
  file = open(out_file, mode = 'w')
  nC = np.count_nonzero(matchesMask)
  file.write('{} {} 0 0\n'.format(nC,nC))
  for dm, mm in zip(matches1to2,matchesMask):
    if mm == 1:
      qidx, tidx = dm.queryIdx, dm.trainIdx
      kp1,kp2 = kps1[qidx], kps2[tidx]
      xl,yl,xr,yr = kp1.pt[0], image_height1 - kp1.pt[1], kp2.pt[0], image_height2 - kp2.pt[1]
      file.write('{:6d} {:10.4f} {:10.4f} {:10.4f} {:10.4f} 0.0 0.0  1\n'.format(qidx, xl, yl, xr, yr))
  file.close()
  return

def precision_difF(lbls, pred_lbls):
    pred_lbls = pred_lbls.squeeze(1)
    assert (lbls.shape == pred_lbls.shape)
    
    # sum_F_squre = torch.square(pred_F + F)
    # dif_F_squre = torch.square(pred_F - F)
    # sum_sqrue1 = torch.mean(sum_F_squre,dim=1)
    # sum_sqrue2 = torch.mean(dif_F_squre,dim=1)
    # min_sqr = torch.minimum(sum_sqrue1,sum_sqrue2)
    # dif_F = torch.mean(min_sqr)

    n_batch = lbls.shape[0]
    pt_pairs_batch = lbls.shape[1]
    mask1 = pred_lbls > 0
    pred_lbls[mask1] = 1.0
    mask2 = pred_lbls < 0
    pred_lbls[mask2] = 0.0

    dif_lbls = abs(lbls - pred_lbls)
    precision = 1 - torch.sum(dif_lbls)/(n_batch*pt_pairs_batch)

    false_mask = (pred_lbls < 0.00001) & (lbls < 0.00001)
    n_correct_false = torch.count_nonzero(false_mask)
    n_false = torch.count_nonzero(lbls < 0.00001)
    false_r = n_correct_false / n_false

    correct_mask = (pred_lbls > 0.00001) & (lbls > 0.00001)
    n_correct_true = torch.count_nonzero(correct_mask)
    n_correct = torch.count_nonzero(lbls > 0.00001)
    true_r = n_correct_true / n_correct


    return (precision, false_r, true_r)

def epi_distances(F, pts1, pts2):
    line_1 = torch.bmm(pts1, F.permute(0, 2, 1))
    line_2 = torch.bmm(pts2, F)

    scalar_product1 = (pts2 * line_1).sum(2)
    scalar_product2 = (pts1*line_2).sum(2)

    sed1 = scalar_product1.abs() * (
        1 / line_1[:, :, :2].norm(2, 2)
    )
    sed2 = scalar_product2.abs() * (
        1 / line_2[:, :, :2].norm(2, 2)
    )

    sed = (sed1 + sed2)/2

    return sed

def Epi_accuracy(label_w, F, pts1, pts2):
    bs = pts1.shape[0]
    residuals = []
    for b in range(bs):
        t_F = F[b].reshape(1,3,3)
        t_lbl = label_w[b]
        t_pts1, t_pts2 = pts1[b][torch.where(t_lbl>0.01)], pts2[b][torch.where(t_lbl>0.01)]
        ones = torch.ones((t_pts1.shape[0],1)).cuda()
        t_pts1, t_pts2 = torch.cat((t_pts1,ones),dim=1).unsqueeze(0), torch.cat((t_pts2,ones),dim=1).unsqueeze(0)
        t_residuals = epi_distances(t_F, t_pts1, t_pts2)
        for res in t_residuals:
            residuals += res
    return torch.stack(residuals).mean()


def epipolar_error(hom_pts1, hom_pts2, F):
    """Compute the symmetric epipolar error."""
    res  = 1 / (np.linalg.norm(F.T.dot(hom_pts2)[0:2], axis=0) + 1e-10)
    res += 1 / (np.linalg.norm(F.dot(hom_pts1)[0:2], axis=0) + 1e-10)
    res *= abs(np.sum(hom_pts2 * np.matmul(F, hom_pts1), axis=0))
    return res

def get_p_and_r(preds, lbls):
    n_pts = preds.shape[0]
    n_est_f = torch.count_nonzero((preds.float() + 0.01).int() - (lbls.float() + 0.01).int())
    n_corrects = n_pts - n_est_f
    p = n_corrects / n_pts

    n_gt_c = torch.sum(lbls)
    n_gt_f = n_pts - n_gt_c

    dif = torch.abs(preds.float() - lbls.float())
    n_est_c = torch.sum((dif < 0.001) & (lbls.float() > 0.001))
    if n_gt_c == 0:
        c_r = 1.0
    else:
        c_r = n_est_c / n_gt_c
    n_est_f = torch.sum((dif < 0.001) & (lbls.float() < 0.001))
    if n_gt_f == 0:
        f_r = 1.0
    else:
        f_r = n_est_f / n_gt_f
    return p, c_r, f_r


def load_test_data(data_path):
    data = []
    if (os.path.exists(data_path)):
        line_count = 0
        with open(data_path, "r") as ifp:
            first_line = ifp.readline()
            n_pairs = int(first_line.replace('\\n',''))
            for n in range(n_pairs):
                data_dic = {}
                info_line = ifp.readline().replace('\\n','').split()
                imgL_ID, imgR_ID, nPts = int(info_line[0]), int(info_line[1]),int(info_line[2])

                max_xl, max_yl, max_xr, max_yr = -999.9, -999.9, -999.9, -999.9
                min_xl, min_yl, min_xr, min_yr = 999.9, 999.9, 999.9, 999.9
                F_line = ifp.readline()
                IDs = []
                mptsL = []
                mptsR = []
                for p in range(nPts):
                    pts_line = ifp.readline().replace('\\n','').split()
                    pt_ID, xl, yl, xr, yr = int(pts_line[0]), float(pts_line[1]), \
                                            float(pts_line[2]),float(pts_line[3]),float(pts_line[4])
                    IDs.append(pt_ID)
                    mptsL.append([xl,yl])
                    mptsR.append([xr,yr])
                    if xl > max_xl:
                        max_xl = xl
                    if xl < min_xl:
                        min_xl = xl
                    if yl > max_yl:
                        max_yl = yl
                    if yl < min_yl:
                        min_yl = yl
                    if xr > max_xr:
                        max_xr = xr
                    if xr < min_xr:
                        min_xr = xr
                    if yr > max_yr:
                        max_yr = yr
                    if yr < min_yr:
                        min_yr = yr
                
                # if nPts < 200:
                #     n_gross_pts = randint(1, int(nPts*0.3 + 0.5))
                #     total_pts = n_gross_pts + nPts
                #     gap = pt_pair_per_batch - (total_pts - (total_pts//pt_pair_per_batch)*pt_pair_per_batch)
                #     n_gross_pts += gap
                #     for i in range(n_gross_pts):
                #         xl, yl = uniform(min_xl,max_xl), uniform(min_yl, max_yl)
                #         xr, yr = uniform(min_xr,max_xr), uniform(min_yr, max_yr)
                #         IDs.append(-1)
                #         mptsL.append([xl,yl])
                #         mptsR.append([xr,yr])

                # shuffle the point pairs
                idx = [i for i in range(len(IDs))]         
                random.shuffle(idx)
                mptsL = [mptsL[i] for i in idx ]
                mptsR = [mptsR[i] for i in idx ]
                IDs = [IDs[i] for i in idx ]
                
                mptsL = np.stack(mptsL)
                mptsR = np.stack(mptsR)
                IDs = np.stack(IDs)
                data_dic['IDX'] = torch.from_numpy(IDs)
                data_dic['PTL'] = torch.from_numpy(mptsL)
                data_dic['PTR'] = torch.from_numpy(mptsR)
                data_dic['IL_ID'] = imgL_ID
                data_dic['IR_ID'] = imgR_ID
                data.append(data_dic)
    return data


def output_atn_data(datas):
    for data in datas:
        l = data['layer']
        name = data['name']
        img = data['img']
        atn_data = data['p'].cpu()
        T = atn_data.shape[1] #T = 2
        for t in range(T):
            file_name = img + '_' + name + 'atn_layer' + str(l)+ '_head' + str(t)+'.dat'
            out_file = open(file_name,mode='w')
            atn = (atn_data[0][t]).numpy()
            nPts = atn.shape[0]
            for i in range(nPts):
                pt_atn = atn[i,:]
                valid_i = np.where(pt_atn>1e-15)
                valid_v = pt_atn[valid_i]
                n_valid = valid_v.shape[0]
                for j in range(n_valid):
                    vld_i = valid_i[0][j]
                    vld_v = valid_v[j]
                    out_file.write('{:10d}   {:.15f}'.format(vld_i, vld_v))
                out_file.write('\n')
            out_file.close()


def output_point_data(data, path):
    out_file = open(path,mode='w')
    out_file.write('1\n 1001 1002  200\n -0.0000   0.0000   0.2864  -0.0001  -0.0000   0.0352  -0.2972  -0.0179   1.0000\n')
    for pair_data in data:
        x1 = pair_data['PTL'].reshape(-1,200,2)
        x2 = pair_data['PTR'].reshape(-1,200,2)
        pt_IDs = pair_data['IDX'].reshape(-1,200)
        for i in range(200):
            out_file.write('{:d}   {:.3f}   {:.3f}    {:.3f}    {:.3f}\n'
            .format(pt_IDs[0][i], x1[0][i][0], x1[0][i][1],x2[0][i][0], x2[0][i][1]))
    out_file.close()


def output_detected_data(ids, ptsl, ptsr, path):
    out_file = open(path,mode='w')
    for ID, ptl, prt in zip(ids, ptsl,ptsr):
        out_file.write('{:10d}   {:.3f}   {:.3f}    {:.3f}    {:.3f}\n'
            .format(ID, ptl[0], ptl[1], prt[0], prt[1]))
    out_file.close()


def get_test_data_file_path(file_dir):
    data_file = []
    dir_list = os.listdir(file_dir)
    for cur_file in dir_list:
        path = os.path.join(file_dir, cur_file)
        if (path.endswith('.dat')):
            data_file.append(path)
    return data_file

def distance(lines, points):
    assert(lines.shape == points.shape)
    n = lines.shape[1]
    ds = 0.0
    for i in range(n):
        line = lines[:,i]
        point = points[:,i]
        a = torch.sum(line*point)
        b = torch.sqrt(line[0]*line[0] + line[1]*line[1])
        ds += torch.abs(a/b)
    return ds/n


def get_area(p1,p2,p3):
    a,b,c = torch.norm(p1-p2),torch.norm(p1-p3),torch.norm(p2-p3)
    s = (a+b+c)/2
    A = torch.sqrt(s*(s-a)*(s-b)*(s-c))
    return A

def get_max_angle(p1,p2,p3):
    a, b = p2-p1, p3-p1
    angle_ab = torch.arccos(torch.sum(a*b)/(torch.norm(a)*torch.norm(b)))
    c = p3-p2
    angle_bc = torch.arccos(torch.sum(c*b)/(torch.norm(c)*torch.norm(b)))
    c = p2-p3
    angle_ac = torch.arccos(torch.sum(c*a)/(torch.norm(c)*torch.norm(a)))
    
    max_angle = angle_ab
    if angle_bc > max_angle:
        max_angle = angle_bc
    if angle_ac > max_angle:
        max_angle = angle_ac
    return max_angle

# def dispersion(points):
#     assert(points.shape[0] == 3)
#     points = (points[0:2,:]).transpose(1,0)
#     tin = Delaunay(points)
#     tri_idx = tin.vertices
#     A_mean = 0.0
#     As = []
#     Ss = []
#     for id in tri_idx:
#         tri = points[id]
#         p1,p2,p3 = tri[0,:],tri[1,:],tri[2,:]
#         A = get_area(p1,p2,p3)
#         As.append(A)
#         A_mean += A
#         m_angle = get_max_angle(p1,p2,p3)
#         Ss.append(m_angle)
#     A_mean /= len(tri_idx)

#     n = len(As)
#     Da, Ds = 0, 0
#     for A, S in zip(As,Ss):
#         Da += ((A/A_mean - 1)*(A/A_mean - 1))
#         Ds += ((S - 1)*(S - 1))
#     Da /= (n-1)
#     Ds /= (n-1)
#     D = torch.sqrt(Da*Ds)
#     return D

def normalize_points(pts, image_size, pixel_size = 0.006):
    h,w = image_size[0], image_size[1]
    pts_x, pts_y = pts[:,0] - w/2, h/2 - pts[:,1]
    return np.stack((pts_x*pixel_size, pts_y*pixel_size),axis=0)

def normalize_correspondences(corrs, im_size1, im_size2):
    pts1, pts2 = corrs[:,0:2,:], corrs[:,2:4,:]
    pts1, pts2 = normalize_pts(pts1,im_size1), normalize_pts(pts2,im_size2)
    return torch.cat((pts1,pts2),dim=1)

def normalize_pts(pts, im_size):
    """Normalize image coordinate using the image size.

	Pre-processing of correspondences before passing them to the network to be 
	independent of image resolution.
	Re-scales points such that max image dimension goes from -0.5 to 0.5.
	In-place operation.

	Keyword arguments:
	pts -- 3-dim array conainting x and y coordinates in the last dimension, first dimension should have size 1.
	im_size -- image height and width
	"""	
	# pts[:, :, 0] -= float(im_size[:,1]) / 2
	# pts[:, :, 1] -= float(im_size[:,0]) / 2
	# pts /= float(max(im_size))
    pts = pts - im_size.unsqueeze(-1)/2
    max_idx = torch.argmax(im_size,dim=1,keepdim=True)
    max_dim = im_size.gather(dim=1,index = max_idx)
    pts /= max_dim[:,:,None]
    return pts

def denormalize_correspondences(corrs, im_size1, im_size2):
    pts1, pts2 = corrs[:,0:2,:], corrs[:,2:4,:]
    pts1, pts2 = denormalize_pts(pts1,im_size1), denormalize_pts(pts2,im_size2)
    return torch.cat((pts1,pts2),dim=1)

def denormalize_pts(pts, im_size):
    """Undo image coordinate normalization using the image size.

	In-place operation.

	Keyword arguments:
	pts -- N-dim array conainting x and y coordinates in the first dimension
	im_size -- image height and width
	"""	
    max_idx = torch.argmax(im_size,dim=1,keepdim=True)
    max_dim = im_size.gather(dim=1,index = max_idx)
    pts *= max_dim[:,:,None]
    pts = pts + im_size.unsqueeze(-1)/2
    return pts
	# pts *= max(im_size)
	# pts[:, :, 0] += im_size[1] / 2
	# pts[:, :, 1] += im_size[0] / 2


outdoor_test_datasets = \
'buckingham_palace,\
notre_dame_front_facade,\
sacre_coeur,\
reichstag,\
fountain,\
herzjesu'

indoor_test_datasets = \
'brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25, \
brown_cogsci_6---brown_cogsci_6---skip-10-dilate-25, \
brown_cogsci_8---brown_cogsci_8---skip-10-dilate-25,\
brown_cs_3---brown_cs3---skip-10-dilate-25,\
brown_cs_7---brown_cs7---skip-10-dilate-25,\
harvard_c4---hv_c4_1---skip-10-dilate-25,\
harvard_c10---hv_c10_2---skip-10-dilate-25,\
harvard_corridor_lounge---hv_lounge1_2---skip-10-dilate-25,\
harvard_robotics_lab---hv_s1_2---skip-10-dilate-25,\
hotel_florence_jx---florence_hotel_stair_room_all---skip-10-dilate-25,\
mit_32_g725---g725_1---skip-10-dilate-25,\
mit_46_6conf---bcs_floor6_conf_1---skip-10-dilate-25,\
mit_46_6lounge---bcs_floor6_long---skip-10-dilate-25,\
mit_w85g---g_0---skip-10-dilate-25,\
mit_w85h---h2_1---skip-10-dilate-25'

def get_inliers(correspondences, F, thres = 3.0):
    ones = torch.ones((correspondences.shape[1],1))
    pts1, pts2 = correspondences[0:2,:].transpose(1,0).cpu(), correspondences[2:4,:].transpose(1,0).cpu()
    homo_pts1, homo_pts2 = torch.cat((pts1,ones),dim=1), torch.cat((pts2,ones),dim=1)
    est_res = epipolar_error(homo_pts1.transpose(1,0).numpy(),homo_pts2.transpose(1,0).numpy(), F)
    inlier_mask = est_res < thres
    return inlier_mask
