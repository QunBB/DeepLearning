from enum import Enum

from .dcn import CrossNetwork
from .masknet import SerialMaskNet, ParallelMaskNet
from .pnn import InnerProduct, OuterProduct
from .contextnet import ContextBlock
from ..utils.interaction import SENet, BiLinear


class Expert(Enum):
    """
    专家交互网络枚举类
    """
    CrossNetwork = CrossNetwork
    InnerProduct = InnerProduct
    OuterProduct = OuterProduct
    SerialMaskNet = SerialMaskNet
    ParallelMaskNet = ParallelMaskNet
    ContextBlock = ContextBlock
    SENet = SENet
    BiLinear = BiLinear

    def init_layer(self, *args, **kwargs):
        return self.value(*args, **kwargs)
