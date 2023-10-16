import os
import tensorflow as tf
from tensorflow.python.framework import graph_io


def create_modelfile(model_version_dir, max_batch,
                     save_type="graphdef",
                     version_policy=None):
    # your model net
    input0_shape = [None, 2]
    input1_shape = [None, 2]
    x1 = tf.placeholder(tf.float32, input0_shape, name='INPUT0')
    inputs_id = tf.placeholder(tf.int32, input1_shape, name='INPUT1')

    out = tf.add(tf.multiply(x1, 0.5), 2)

    embedding = tf.get_variable("embedding_table", shape=[100, 10])
    pre = tf.nn.embedding_lookup(embedding, inputs_id)

    out0 = tf.identity(out, "OUTPUT0")
    out1 = tf.identity(pre, "OUTPUT1")

    try:
        os.makedirs(model_version_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        if save_type == 'graphdef':
            create_graphdef_modelfile(model_version_dir, sess,
                                      outputs=["OUTPUT0", "OUTPUT1"])
        elif save_type == 'savemodel':
            create_savedmodel_modelfile(model_version_dir,
                                        sess,
                                        inputs={
                                            "INPUT0": x1,
                                            "INPUT1": inputs_id
                                        },
                                        outputs={
                                            "OUTPUT0": out,
                                            "OUTPUT1": pre
                                        })
        else:
            raise ValueError("save_type must be one of ['tensorflow_graphdef', 'tensorflow_savedmodel']")

    create_modelconfig(models_dir=os.path.dirname(model_version_dir),
                       max_batch=max_batch,
                       save_type=save_type,
                       version_policy=version_policy)


def create_graphdef_modelfile(model_version_dir, sess, outputs):
    """
    tensorflow graphdef只能保存constant，无法保存Variable
    可以借助tf.graph_util.convert_variables_to_constants将Variable转化为constant
    :param model_version_dir:
    :param sess:
    :return:
    """
    graph = sess.graph.as_graph_def()
    new_graph = tf.graph_util.convert_variables_to_constants(sess=sess,
                                                             input_graph_def=graph,
                                                             output_node_names=outputs)
    graph_io.write_graph(new_graph,
                         model_version_dir,
                         "model.graphdef",
                         as_text=False)


def create_savedmodel_modelfile(model_version_dir, sess, inputs, outputs):
    """

    :param model_version_dir:
    :param sess:
    :param inputs: dict, {input_name: input_tensor}
    :param outputs: dict, {output_name: output_tensor}
    :return:
    """
    tf.saved_model.simple_save(sess,
                               model_version_dir + "/model.savedmodel",
                               inputs=inputs,
                               outputs=outputs)


def create_modelconfig(models_dir, max_batch, save_type, version_policy=None):
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

    if save_type == 'graphdef':
        platform = "tensorflow_graphdef"
    elif save_type == 'savemodel':
        platform = "tensorflow_savedmodel"
    else:
        raise ValueError("save_type must be one of ['tensorflow_graphdef', 'tensorflow_savedmodel']")

    # 这里的 name、data_type、dims 是根据以上模型的输入&输出节点配置
    # dims不包含batch_size维度
    config = '''
name: "{}"
platform: "{}"
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
'''.format(model_name, platform, max_batch, version_policy_str)

    try:
        os.makedirs(config_dir)
    except OSError as ex:
        pass  # ignore existing dir

    with open(config_dir + "/config.pbtxt", "w") as cfile:
        cfile.write(config)


if __name__ == '__main__':
    create_modelfile(
        # model_version_dir="/Users/hong/Desktop/server/docs/examples/model_repository/tf_graphdef/1",
        max_batch=8,
        model_version_dir="/Users/hong/Desktop/server/docs/examples/model_repository/tf_savemodel/1",
        save_type="savemodel"
    )
