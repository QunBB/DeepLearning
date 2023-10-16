from enum import IntEnum
from dataclasses import dataclass


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


# PNN
class KernelType(IntEnum):
    """
    0-2对应KPNN不同kernel形式，3对应PIN的micro net
    """
    Num = 0
    Vec = 1
    Mat = 2
    Net = 3  # PIN
