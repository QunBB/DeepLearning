import torch
import torch.nn as nn
from torch.utils.data import DataLoader


# steps_accumulate为梯度累积的步数，即累积`steps_accumulate`再进行一次反向传播更新参数
# 实现`steps_accumulate * bs`的大批次训练
def train(model: nn.Module,
          dataloader: DataLoader,
          optimizer: torch.optim.Optimizer,
          steps_accumulate: int):

    model.zero_grad()
    model.train()

    for i, data in enumerate(dataloader):

        loss = model(data) / steps_accumulate
        loss.backward()

        if (i + 1) % steps_accumulate == 0:
            optimizer.step()
            optimizer.zero_grad()
