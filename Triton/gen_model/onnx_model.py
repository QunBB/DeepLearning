"""
Exporting To ONNX From TensorFlow

pip install -U tf2onnx

# savedmodel
python -m tf2onnx.convert --saved-model tensorflow-model-path --output model.onnx

# checkpoint
python -m tf2onnx.convert --checkpoint tensorflow-model-meta-file-path --output model.onnx --inputs input0:0,input1:0 --outputs output0:0

# graphdef
python -m tf2onnx.convert --graphdef tensorflow-model-graphdef-file --output model.onnx --inputs input0:0,input1:0 --outputs output0:0
"""

import os
import torch
import torch.onnx

from torch_model import MyNet


def torch2onnx(model_version_dir, max_batch):
    # 定义输入的格式
    example_input0 = torch.zeros([max_batch, 2], dtype=torch.float32)
    example_input1 = torch.zeros([max_batch, 2], dtype=torch.int32)

    my_model = MyNet()

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    torch.onnx.export(my_model,
                      (example_input0, example_input1),
                      os.path.join(model_version_dir, 'model.onnx'),
                      # 输入节点的名称
                      input_names=("INPUT0", "INPUT1"),
                      # 输出节点的名称
                      output_names=("OUTPUT0", "OUTPUT1"),
                      # 设置batch_size的维度
                      dynamic_axes={"INPUT0": [0], "INPUT1": [0], "OUTPUT0": [0], "OUTPUT1": [0]},
                      verbose=True)


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
    config = '''
name: "{}"
platform: "onnxruntime_onnx"
max_batch_size: {}
version_policy: {}
input [
  {{
    name: "INPUT0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }},
  {{
    name: "INPUT1"
    data_type: TYPE_INT32
    dims: [ 2 ]
  }}
]
output [
  {{
    name: "OUTPUT0"
    data_type: TYPE_FP32
    dims: [ 2 ]
  }},
  {{
    name: "OUTPUT1"
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
    max_batch = 8
    model_version_dir = '/Users/hong/Desktop/server/docs/examples/model_repository/torch_onnx/1'
    torch2onnx(model_version_dir=model_version_dir,
               max_batch=max_batch)
    create_modelconfig(models_dir=os.path.dirname(model_version_dir),
                       max_batch=max_batch)
