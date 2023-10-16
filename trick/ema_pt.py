import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class EMA:
    def __init__(self, model: nn.Module,
                 decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}

    def register(self):
        """创建shadow权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        """EMA平滑操作，更新shadow权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()

    def apply_shadow(self):
        """使用shadow权重作为模型权重，并创建原模型权重备份"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        """恢复模型权重"""
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}


def train_with_ema(model: nn.Module,
                   dataloader: DataLoader,
                   valid_steps: int,
                   optimizer: torch.optim.Optimizer):
    # 初始化EMA
    ema = EMA(model, 0.999)
    ema.register()

    for i, data in enumerate(dataloader):
        # 正常的训练代码
        model.train()
        loss = model(data)
        loss.backward()

        optimizer.step()
        optimizer.zero_grad()

        # 更新ema权重
        ema.update()

        # 验证&保存模型
        if (i + 1) % valid_steps == 0:
            # 使用ema权重
            ema.apply_shadow()

            # 验证工作
            print('do valid')

            # 保存模型工作
            print('save model')

            # 恢复原模型权重，继续正常的训练
            ema.restore()
