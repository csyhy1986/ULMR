import torch
import torch.nn as nn
from torch.nn import functional as F

class ResNet_Block(nn.Module):
    def __init__(self, inchannel, outchannel, pre=False):
        super(ResNet_Block, self).__init__()
        self.pre = pre
        self.right = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
        )
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(),
            nn.Conv2d(outchannel, outchannel, (1, 1)),
            nn.InstanceNorm2d(outchannel),
            nn.BatchNorm2d(outchannel),
        )
    def forward(self, x):
        x1 = self.right(x) if self.pre is True else x
        out = self.left(x)  
        out = out + x1
        return F.relu(out)

class LGC_net(nn.Module):
    def __init__(self):
        super(LGC_net, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(5, 128, (1, 1)),
                                   nn.BatchNorm2d(128),
                                   nn.ReLU(inplace=True),
                                   nn.Conv2d(128, 128, (1, 1)))
        self.conv2 = nn.Sequential(nn.Conv2d(128, 128, (1, 1)),
                                   nn.BatchNorm2d(128))
        self.res_blocks = nn.ModuleList()
        for _ in range(12):
            self.res_blocks.append(ResNet_Block(128,128))
        self.linear = nn.Conv2d(128, 1, (1, 1))
        self.initialize_weights()
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.xavier_normal_(m.weight.data)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias.data, 0)
    
    def forward(self, x):
        out = self.conv1(x)
        for res_blk in self.res_blocks:
            out = res_blk(out)
        out = self.conv2(out)
        logit = self.linear(out)
        # probs = torch.sigmoid(logit.squeeze(-1).squeeze(1))

        # log_probs = F.logsigmoid(logit.squeeze(-1).squeeze(1))
        # normalizer = torch.logsumexp(log_probs, dim=1,keepdim=True)
        # log_probs = log_probs - normalizer
        return logit



        
# model = LGC_net()
# model.cuda()
# x = torch.rand(32, 5, 200).cuda()  # Normalized matches: Batch_size * N *4
# weights, Es = model(x)       # Lists of predicted weights and Es 
# mask = weights > 0