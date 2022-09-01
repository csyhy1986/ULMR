from numpy.core.fromnumeric import mean
import torch
from photo_tourism import photo_tourism as ph_dataset
from XYVT_data import xyvt_data
from torch.optim.lr_scheduler import CosineAnnealingLR
from LGC_net import LGC_net
# from NM_net_V2 import NM_Net_v2
from utils import collate_fn_cat_xyvt, normalize_correspondences, worker_init_fn
from torch.utils.data import DataLoader
import environment as env
from torch.distributions import Categorical

EPOCH = 100
BATCH_SIZE = 32
LR = 0.0001
N_HYPO = 64
nMIN_SET = 8
dataset_name = 'XYVT'
train_data = 'Pozhuang_Cross'

if dataset_name == 'IMC-PT':
    train_dataset = ph_dataset(train_data)
if dataset_name == 'XYVT':
    train_dataset = xyvt_data(train_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, 
                          drop_last=True,collate_fn=collate_fn_cat_xyvt, worker_init_fn=worker_init_fn)
policy_network = LGC_net().cuda()
# policy_network.load_state_dict(torch.load("optimal_3.pth"))
optimizer = torch.optim.Adam(policy_network.parameters(), lr=LR)
scheduler = CosineAnnealingLR(optimizer, EPOCH, 1e-8)

for epoch in range(1, EPOCH+1):
    i = 0
    mean_loss = []
    policy_network = policy_network.train()
    for record in train_loader:
        if(dataset_name == 'IMC-PT'):
            xs = record['xs'].float()
            imgsz1, imgsz2 = 2*record['cxyf1'][:,0:2], 2*record['cxyf2'][:,0:2]
            nor_pts = normalize_correspondences(xs,imgsz1,imgsz2)
            rts = record['rts'].float()
            nor_pts = torch.cat((xs,rts),dim=1)
        else:
            xs1, xs2 = record[1].float().transpose(2,1), record[2].float().transpose(2,1)
            # imgsz1, imgsz2 = torch.cat((record['infos'][:,1:2], record['infos'][:,0:1]),dim=1), \
            #                  torch.cat((record['infos'][:,3:4], record['infos'][:,2:3]),dim=1)
            imgsz1, imgsz2 = record[0][:,0:2], record[0][:,2:4]
            xs = torch.cat((xs1,xs2),dim=1)
            nor_pts = normalize_correspondences(xs,imgsz1,imgsz2)
            rts = record[3].float().unsqueeze(1)
            nor_pts = torch.cat((xs,rts),dim=1)

        logit = policy_network(nor_pts.unsqueeze(-1).cuda()).squeeze()
        probs = torch.softmax(logit,dim=1)

        loss_pool = []
        rs = []
        for j in range(BATCH_SIZE):
            pts1, pts2 = xs[j, 0:2, :], xs[j, 2:4, :]
            new_state = torch.cat((pts1,pts2),dim=0)
            new_probs = probs[j]
            reward_pool = torch.zeros(N_HYPO).cuda()
            action_pool = torch.zeros(N_HYPO,nMIN_SET).cuda()
            for k in range(N_HYPO):
                actions = env.sample_actions(new_probs)
                reward, _ = env.step(actions,new_state)
                reward_pool[k] = reward
                action_pool[k] = actions
                rs.append(reward)
            reward_mean = torch.mean(reward_pool)
            reward_std = torch.std(reward_pool)
            reward_pool = (reward_pool - reward_mean)/(reward_std + 1e-8)
            t_loss = 0
            for k in range(N_HYPO):
                m = Categorical(new_probs)
                log_p = m.log_prob(action_pool[k])
                loss = -torch.sum(log_p)*reward_pool[k]/N_HYPO
                t_loss += loss
            loss_pool.append(t_loss)
        
        loss = torch.stack(loss_pool).mean()
        policy_network.zero_grad()
        loss.backward()
        optimizer.step()
        r = mean(rs)
        print(f"[EPOCH:{epoch:4d}|Iter:{i:4d}|Total:{len(train_loader):4d}] Loss: {loss.detach().item(): .8f} R: {r: .2f}")
        i = i+1
    scheduler.step()
    torch.save(policy_network.state_dict(), "LFGC_e{}.pth".format(epoch))




