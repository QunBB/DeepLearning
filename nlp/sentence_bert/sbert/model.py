import tensorflow as tf

from sentence_bert.bert import BertConfig

from sentence_bert.utils import get_sentence_emb


class SBERT:

    def __init__(self, bert_config_file, objective,
                 pooling_mode='mean',
                 num_labels=None,
                 margin=1.):
        """

        :param bert_config_file: bert_config配置文件路径
        :param objective: 微调的结构
        :param pooling_mode: 句向量pooling的策略
        :param num_labels: 对于classification微调结构，需设置labels的数量
        :param margin: 对于triplet结构，需设置
        """
        assert objective in ('classification', 'regression', 'triplet')
        assert pooling_mode in ('mean', 'max', 'cls')

        self.bert_config = BertConfig.from_json_file(bert_config_file)
        self.objective = objective
        self.num_labels = num_labels
        self.margin = margin
        self.pooling_mode = pooling_mode

    def __call__(self, inputs_a, inputs_b, is_training, labels=None, inputs_c=None):
        """

        :param inputs_a: 句子a的输入，dict形式
            :keyword input_ids: int32 Tensor [batch_size, max_seq_len]
            :keyword seq_len: int32 Tensor [batch_size]
        :param inputs_b: 句子b的输入，同inputs_a。如为triplet结构，则为positive sentence
        :param is_training:
        :param labels:
        :param inputs_c: 同inputs_a。仅triplet结构时使用，为negative sentence
        :return:
        """
        sentence_emb_a = get_sentence_emb(input_ids=inputs_a['input_ids'],
                                          seq_len=inputs_a['seq_len'],
                                          is_training=is_training,
                                          bert_config=self.bert_config,
                                          pooling_mode=self.pooling_mode)

        sentence_emb_b = get_sentence_emb(input_ids=inputs_b['input_ids'],
                                          seq_len=inputs_b['seq_len'],
                                          is_training=is_training,
                                          bert_config=self.bert_config,
                                          pooling_mode=self.pooling_mode)

        loss = None
        if self.objective == 'classification':
            logits = tf.layers.dense(
                inputs=tf.concat([sentence_emb_a, sentence_emb_b, tf.abs(sentence_emb_a - sentence_emb_b)], axis=-1),
                units=self.num_labels,
                activation=tf.nn.softmax,
                kernel_initializer=tf.variance_scaling_initializer()
            )

            prob = tf.argmax(logits, axis=-1)

            if labels is not None:
                if len(labels) == 1:
                    labels = tf.one_hot(labels, self.num_labels)

                epsilon = 1e-8
                labels = tf.cast(labels, tf.float32)
                loss = labels * tf.log(logits + epsilon) + (1 - labels) * tf.log(1 - logits + epsilon)
                loss = tf.negative(loss)
                loss = tf.reduce_mean(tf.reduce_sum(loss, axis=-1))

            return loss, logits, prob

        if self.objective == 'regression':
            cos_sim = tf.matmul(tf.nn.l2_normalize(sentence_emb_a, axis=-1),
                                tf.nn.l2_normalize(sentence_emb_b, axis=-1))
            if labels is not None:
                loss = self._mse(cos_sim, labels)

            return loss, cos_sim

        if self.objective == 'triplet':
            sentence_emb_c = get_sentence_emb(input_ids=inputs_c['input_ids'],
                                              seq_len=inputs_c['seq_len'],
                                              is_training=is_training,
                                              bert_config=self.bert_config,
                                              pooling_mode=self.pooling_mode)

            negative_distance = self._euclidean(sentence_emb_a, sentence_emb_c)
            positive_distance = self._euclidean(sentence_emb_a, sentence_emb_b)
            loss = tf.maximum(negative_distance - positive_distance + self.margin, 0)
            loss = tf.reduce_mean(loss)

            return loss, negative_distance, positive_distance

    def _euclidean(self, t1, t2):
        """欧式距离"""
        return tf.sqrt(tf.reduce_mean(tf.pow(t1 - t2, 2)))

    def _mse(self, y1, y2):
        """MSE loss"""
        return tf.reduce_mean(tf.pow(y1 - y2, 2))

    def sentence_embedding(self, inputs):
        """
        根据输入获取句向量
        :param inputs: dict格式
            :keyword input_ids: int32 Tensor [batch_size, max_seq_len]
            :keyword seq_len: int32 Tensor [batch_size]
        :return:
        """
        return get_sentence_emb(input_ids=inputs['input_ids'],
                                seq_len=inputs['seq_len'],
                                is_training=False,
                                bert_config=self.bert_config,
                                pooling_mode=self.pooling_mode)
