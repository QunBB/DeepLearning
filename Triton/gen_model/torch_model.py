import os
import torch
from torch import nn


class MyNet(nn.Module):

    def __init__(self):
        super(MyNet, self).__init__()

        self.embedding = nn.Embedding(num_embeddings=100,
                                      embedding_dim=10)

    def forward(self, input0, input1):
        # tf.add(tf.multiply(x1, 0.5), 2)
        output0 = torch.add(torch.multiply(input0, 0.5), 2)

        output1 = self.embedding(input1)

        return output0, output1


def create_modelfile(model_version_dir, max_batch,
                     version_policy=None):
    # your model net

    # 定义输入的格式
    example_input0 = torch.zeros([2], dtype=torch.float32)
    example_input1 = torch.zeros([2], dtype=torch.int32)

    my_model = MyNet()

    traced = torch.jit.trace(my_model, (example_input0, example_input1))

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    traced.save(model_version_dir + "/model.pt")

    create_modelconfig(models_dir=os.path.dirname(model_version_dir),
                       max_batch=max_batch,
                       version_policy=version_policy)


def create_modelconfig(models_dir, max_batch, version_policy=None):
    model_name = os.path.basename(models_dir)
    config_dir = models_dir

    # version policy
    version_policy_str = "{ latest { num_versions: 1 }}"
    if version_policy is not None:
        type, val = version_policy
        if type == 'latest':
            version_policy_str = "{{ latest {{ num_versions: {} }}}}".format(
                val)
        elif type == 'specific':
            version_policy_str = "{{ specific {{ versions: {} }}}}".format(val)
        else:
            version_policy_str = "{ all { }}"

    # 这里的 name、data_type、dims 是根据以上模型的输入&输出节点配置
    # dims不包含batch_size维度
    # 默认forward中的第一个输入名称为INPUT__0，第一个返回(输出)名称为OUTPUT__0
    config = '''
name: "{}"
platform: "pytorch_libtorch"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }},
  {{
    name: "INPUT__1"
    data_type: TYPE_INT32
    dims: [ 2 ]
  }}
]
output [
  {{
    name: "OUTPUT__0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }},
  {{
    name: "OUTPUT__1"
    data_type: TYPE_FP32
    dims: [ 2,10 ]
  }}
]
'''.format(model_name, max_batch, version_policy_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    create_modelfile(
        model_version_dir="/Users/hong/Desktop/server/docs/examples/model_repository/torch_model/1",
        max_batch=8,
        # model_version_dir="/Users/hong/Desktop/server/docs/examples/model_repository/tf_savemodel/1",
        # save_type="savemodel"
    )
