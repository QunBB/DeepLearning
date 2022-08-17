import tensorflow as tf
import numpy as np


def apply_swa(checkpoint_list: list,
              weight_list: list,
              save_path: str,
              sess: tf.Session = None,
              strict: bool = True):
    """

    :param checkpoint_list: 要进行swa的模型路径列表
    :param weight_list: 每个模型对应的权重
    :param save_path: swa后的模型导出路径
    :param sess:
    :param strict: 是否需要完全匹配checkpoint的参数
    :return:
    """
    vars_list = tf.trainable_variables()
    saver = tf.train.Saver(var_list=vars_list)

    swa_op = []
    for var in vars_list:
        temp = []
        try:
            temp = [tf.train.load_variable(path, var.name) * w for path, w in zip(checkpoint_list, weight_list)]
        except tf.python.framework.errors_impl.NotFoundError:
            print(f"checkpoint don't match the model, var: '{var.name}' not in checkpoint")
            if strict:
                raise tf.python.framework.errors_impl.NotFoundError

        swa_op.append(tf.assign(var, np.sum(temp, axis=0)))

    if sess is None:
        sess = tf.Session()
    with sess.as_default() as sess:
        sess.run(swa_op)
        saver.save(sess, save_path)


if __name__ == '__main__':
    # 测试程序
    from NLP.sentence_bert.bert import BertConfig, BertModel
    model = BertModel(config=BertConfig.from_json_file('chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_config.json'),
                      is_training=False,
                      input_ids=tf.placeholder(tf.int32, [None, 128]))
    apply_swa(checkpoint_list=['chinese_roberta_wwm_ext_L-12_H-768_A-12/bert_model.ckpt',
                               'chinese_macbert_base/chinese_macbert_base.ckpt'],
              weight_list=[0.5, 0.5],
              save_path='bert_swa.ckpt')
