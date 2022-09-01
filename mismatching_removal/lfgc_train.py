import torch

# from photo_tourism import photo_tourism
# from dataset_kitti import SparseDataset
from photo_tourism import photo_tourism as ph_dataset
# from loss import mtl_loss
from torch.optim.lr_scheduler import StepLR
from LGC_net import LGC_net
from utils import collate_fn_cat, worker_init_fn
from torch.utils.data import DataLoader

EPOCH = 100
BATCH_SIZE = 32
LR = 0.0001

# train_datasets = 'brown_bm_3---brown_bm_3-maxpairs-10000-random---skip-10-dilate-25,st_peters_square'
# train_data = train_datasets.split(',') #support multiple training datasets used jointly
# train_data = ['../traindata/' + ds + '/train_data/' for ds in train_data]

# eval_datasets = 'brown_cogsci_2---brown_cogsci_2---skip-10-dilate-25,buckingham_palace'
# eval_data = eval_datasets.split(',') #support multiple training datasets used jointly
# eval_data = ['../traindata/' + ds + '/test_data/' for ds in eval_data]

# print('Using datasets:')
# for d in train_data:
# 	print(d)

# trainset = SparseDataset(train_data, 1.0, 2000, fmat=True)
# trainset_loader = torch.utils.data.DataLoader(trainset, shuffle=True, num_workers=0, batch_size=BATCH_SIZE, drop_last=True)
# evalset = SparseDataset(eval_data, 1.0, 2000, fmat=True)
# eval_loader = torch.utils.data.DataLoader(evalset, shuffle=True, num_workers=0, batch_size=BATCH_SIZE, drop_last=True)

train_data = 'st_peters.brown_bm_3_05'
train_dataset = ph_dataset(train_data)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, 
                          drop_last=True,collate_fn=collate_fn_cat, worker_init_fn=worker_init_fn)

net = LGC_net().cuda()
# loss_criea = mtl_loss().cuda()
optimizer = torch.optim.Adam(net.parameters(), lr=LR)
scheduler = StepLR(optimizer, step_size=1, gamma=0.9)

max_fr = 0.0
for epoch in range(1, EPOCH+1):
    i = 0
    mean_loss = []
    net = net.train()
    for record in train_loader:
        res = record['ys']
        xs = record['xs']
        cxyf1, cxyf2 = record['cxyf1'], record['cxyf2']
        # cx1,cy1,f1 = cxyf1[:,0], cxyf1[:,1], cxyf1[:,2]
        # cx2,cy2,f2 = cxyf2[:,0], cxyf2[:,1], cxyf2[:,2]
        K1, K2 = torch.zeros((BATCH_SIZE,3,3)), torch.zeros((BATCH_SIZE,3,3))
        for j in range(BATCH_SIZE):
            x1,x2 = xs[j,0:2,:], xs[j,2:4,:]
            cx1,cy1,f1= cxyf1[j,0],cxyf1[j,1],cxyf1[j,2]
            cx2,cy2,f2= cxyf2[j,0],cxyf2[j,1],cxyf2[j,2]
            xs[j,0:2,:] = x1*f1 + torch.tensor([[cx1],[cy1]])
            xs[j,2:4,:] = x2*f2 + torch.tensor([[cx2],[cy2]])
            K1[j,:,:] = torch.tensor([[f1,0,cx1],[0,f1,cy1],[0,0,1]])
            K2[j,:,:] = torch.tensor([[f2,0,cx2],[0,f2,cy2],[0,0,1]])

        gt_t, gt_R = record['ts'], record['Rs']
        gt_e = torch.matmul(
            torch_skew_symmetric(gt_t).view(gt_t.shape[0], 3, 3),
            gt_R.view(gt_t.shape[0], 3, 3)
        ).float()

        
        gt_f = torch.bmm(torch.bmm(torch.inverse(K2.transpose(2,1)),gt_e),torch.inverse(K1)).reshape((BATCH_SIZE,-1))
        gt_f =  gt_f/ torch.norm(gt_f, dim=1, keepdim=True)
        xs = xs.unsqueeze(-1)

        logit, pre_f = net(xs.float().cuda())
        gt_geod_d = res[:, 0, :]
        lables = (gt_geod_d < 0.0001).float()
        net.zero_grad()
        use_floss = False
        if epoch > EPOCH*0.5:
            use_floss = True
        loss = loss_criea(logit,lables.cuda(),pre_f,gt_f.cuda(), use_floss = use_floss)
        loss.backward()
        optimizer.step()
        mean_loss.append(loss)
        
        i = i+1
        if (i+1) % 100 == 0:
            line = 'Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(epoch, EPOCH, i+1, 
                len(train_loader), torch.mean(torch.stack(mean_loss)).item())
            print(line)
            mean_loss = []
    scheduler.step()
    torch.save(net.state_dict(), "model_weights.pth")