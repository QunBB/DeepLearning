import tensorrt as trt
import os


def onnx2trt(model_version_dir, onnx_model_file, max_batch):
    logger = trt.Logger(trt.Logger.WARNING)

    builder = trt.Builder(logger)

    # The EXPLICIT_BATCH flag is required in order to import models using the ONNX parser
    network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

    parser = trt.OnnxParser(network, logger)

    success = parser.parse_from_file(onnx_model_file)
    for idx in range(parser.num_errors):
        print(parser.get_error(idx))

    if not success:
        pass  # Error handling code here

    profile = builder.create_optimization_profile()
    # INPUT0可以接收[1, 2] -> [max_batch, 2]的维度
    profile.set_shape("INPUT0", [1, 2], [1, 2], [max_batch, 2])
    profile.set_shape("INPUT1", [1, 2], [1, 2], [max_batch, 2])

    config = builder.create_builder_config()
    config.add_optimization_profile(profile)

    # tensorrt 8.x
    # config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 20)  # 1 MiB

    # tensorrt 7.x
    config.max_workspace_size = 1 << 20

    try:
        engine_bytes = builder.build_serialized_network(network, config)
    except AttributeError:
        engine = builder.build_engine(network, config)
        engine_bytes = engine.serialize()
        del engine

    with open(os.path.join(model_version_dir, 'model.plan'), "wb") as f:
        f.write(engine_bytes)


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
platform: "tensorrt_plan"
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
    onnx2trt(model_version_dir=model_version_dir,
             onnx_model_file='model.onnx',
             max_batch=max_batch)
    create_modelconfig(models_dir=os.path.dirname(model_version_dir),
                       max_batch=max_batch)
