import torch
import cv2
from utils import epipolar_error
from torch.distributions import Categorical

def sample_actions(probs, n_samples = 8):
    m = Categorical(probs)
    actions = m.sample([n_samples])
    return actions

def step(action, state, thres = 3.0):
    n_samples = action.shape[0]
    assert((torch.min(action) >= 0) & (torch.max(action) < state.shape[1]))
    if(n_samples < 8):
        return 0, torch.zeros(state.shape[1])
    pts1, pts2 = state[0:2,action].transpose(1,0).cpu(), state[2:4,action].transpose(1,0).cpu()
    F = cv2.findFundamentalMat(pts1.numpy(), pts2.numpy(), cv2.FM_8POINT)[0]
    if F is None:
        return 0, torch.zeros(state.shape[1])
    
    ones = torch.ones((state.shape[1],1))
    pts1, pts2 = state[0:2,:].transpose(1,0).cpu(), state[2:4,:].transpose(1,0).cpu()
    homo_pts1, homo_pts2 = torch.cat((pts1,ones),dim=1), torch.cat((pts2,ones),dim=1)
    est_res = epipolar_error(homo_pts1.transpose(1,0).numpy(),homo_pts2.transpose(1,0).numpy(), F)
    inlier_mask = est_res < thres
    reward = sum(inlier_mask)
    # if reward < n_samples:
    #     return 0, torch.zeros(state.shape[1])
    return reward, inlier_mask

def get_inliers(probs, state, N_HYPO = 32, thres = 3.0):
    best_reward = 0
    # best_inliers = np.zeros_like(probs.cpu())
    for _ in range(N_HYPO):
        action = sample_actions(probs)
        reward, in_mask = step(action,state, thres = thres)
        if reward > best_reward:
            best_reward = reward
            best_inliers = in_mask
    if best_reward == 0:
        return torch.zeros_like(probs).cpu()
    return torch.from_numpy(best_inliers)
