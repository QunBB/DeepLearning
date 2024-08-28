from enum import IntEnum
from dataclasses import dataclass
from typing import Optional, List, Sequence


class LinearTerms(IntEnum):
    """FwFMs中的线性项"""
    LW = 0
    FeLV = 1
    FiLV = 2


class FMType(IntEnum):
    """FMs选项"""
    FM = 1
    FwFM = 2
    FEFM = 3


@dataclass
class Field:
    name: str
    dim: int = 4  # embedding维度大小
    vocabulary_size: int = 1  # dense类型为1
    l2_reg: float = 0.  # embedding的正则惩罚
    init_mean: float = 0.  # 初始化参数
    init_std: float = 1.0  # 初始化参数


# PNN
class KernelType(IntEnum):
    """
    0-2对应KPNN不同kernel形式，3对应PIN的micro net
    """
    Num = 0
    Vec = 1
    Mat = 2
    Net = 3  # PIN


@dataclass
class DINField:
    """
    DIN系列模型的Field
    """
    name: str
    embedding_dim: int  # embedding向量维度: D
    vocabulary_size: int  # unique id数量: K
    l2_reg: Optional[float] = None  # 普通正则惩罚项
    mini_batch_regularization: bool = False  # 是否使用Mini-batch Aware Regularization, 优先于l2_reg
    ids_occurrence: Optional[Sequence[int]] = None  # 特征ID按照0-(K-1)统计在所有样本中出现的频次
    init_mean: float = 0.  # 初始化参数
    init_std: float = 1.0  # 初始化参数
