import numpy as np
import cv2
import torch
from NM_net_V2 import NM_Net_v2
import environment as env
import utils

detector = cv2.SIFT_create(nfeatures = 2000, contrastThreshold=1e-5)

policy_network = NM_Net_v2()
policy_network.load_state_dict(torch.load("optimal_5.pth"))
policy_network = policy_network.cuda()
policy_network.eval()
print("Successfully loaded model.")
# read images
img1 = cv2.imread("images\\9-1.jpg")
# img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)

img2 = cv2.imread("images\\9-2.jpg")
# img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

# detect features
kp1, desc1 = detector.detectAndCompute(img1, None)
kp2, desc2 = detector.detectAndCompute(img2, None)

print("Feature found in image 1:", len(kp1))
print("Feature found in image 2:", len(kp2))

# feature matching
bf = cv2.BFMatcher()
matches = bf.knnMatch(desc1, desc2, k=2)

good_matches = []
pts1 = []
pts2 = []
ratios = []

for (m,n) in matches:
	if m.distance < 1.0*n.distance: # apply Lowe's ratio filter
		good_matches.append(m)
		pts2.append(kp2[m.trainIdx].pt)
		pts1.append(kp1[m.queryIdx].pt)
		# ratios.append(m.distance/n.distance)

print("Number of valid matches:", len(good_matches))

pts1 = np.stack(pts1)
pts2 = np.stack(pts2)
# ratios = torch.from_numpy(np.stack(ratios).reshape((-1,1))).transpose(1,0).unsqueeze(0)

pts1, pts2 = torch.from_numpy(np.transpose(pts1)).unsqueeze(0), torch.from_numpy(np.transpose(pts2)).unsqueeze(0)
im_size1, im_size2 = torch.tensor([img1.shape[1], img1.shape[0]]).unsqueeze(0), torch.tensor([img2.shape[1], img2.shape[0]]).unsqueeze(0)
nor_pts1, nor_pts2 = utils.normalize_pts(pts1, im_size1), utils.normalize_pts(pts2, im_size2)

# create data tensor of feature coordinates and matching ratios
correspondences = torch.cat([nor_pts1,nor_pts2],dim=1)
state = correspondences.cuda()
logit = policy_network(state.transpose(2,1).float())
probs = torch.softmax(logit,dim=1)

state = torch.cat((pts1,pts2),dim=1).cuda()
ULMR_inliers = env.get_inliers(probs.squeeze(),state.squeeze(), N_HYPO = 50, thres = 3.0)
print("ULMR Inliers: ", int(torch.sum(ULMR_inliers)))

# create a visualization of the matching, comparing results of RANSAC and NG-RANSAC
ULMR_inliers = ULMR_inliers.byte().numpy().ravel().tolist()
match_img_ULMR = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=2, matchColor=(200,130,0), matchesMask = ULMR_inliers)
# match_img = np.concatenate((match_img_ransac, match_img_ngransac), axis = 0)

cv2.imwrite("ULMR.bmp", match_img_ULMR)
print("Done. Visualization of the result stored as demo.jpg")

utils.export_dpt_file(img1.shape[0],img2.shape[0],kp1,kp2,good_matches,ULMR_inliers,"ULMR.dpt")
print("Done. Result stored as ULMR.dpt")
