import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        # クラスラベルの1次元配列をone-hot形式に変換
        targets_one_hot = F.one_hot(targets, num_classes=inputs.size(-1))

        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets_one_hot.float(), reduction='none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets_one_hot.float(), reduction='none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        # 各サンプルに対するlossを求めるためにdim=-1でsumする
        F_loss = F_loss.sum(dim=-1)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss
