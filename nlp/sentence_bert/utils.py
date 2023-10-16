import tensorflow as tf

from sentence_bert.bert import BertModel


def get_sentence_emb(input_ids, seq_len, is_training, bert_config, pooling_mode):
    token_emb, cls_emb = get_bert_output(input_ids, seq_len, is_training, bert_config)

    if pooling_mode == 'mean':
        sentence_emb = mean_pooling(token_emb, seq_len)
    elif pooling_mode == 'max':
        sentence_emb = max_pooling(token_emb, seq_len)
    elif pooling_mode == 'cls':
        sentence_emb = cls_emb
    else:
        raise ValueError("error pooling_mode")

    return sentence_emb


def get_bert_output(input_ids, seq_len, is_training, bert_config):
    max_len = input_ids.shape.as_list()[-1]
    input_mask = tf.sequence_mask(seq_len, max_len, dtype=tf.int32)
    segment_ids = tf.zeros_like(input_ids)

    # 在相同的variable_scope下，保证不同输入对应的bert模型是同一个，即参数绑定
    model = BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=False,
        scope="bert")
    return model.get_sequence_output(), model.get_pooled_output()


def mean_pooling(token_emb, seq_len):
    mask = tf.sequence_mask(seq_len, dtype=tf.float32)
    mask = tf.expand_dims(mask, axis=-1)
    seq_len = tf.cast(tf.expand_dims(seq_len, axis=-1), tf.float32)

    token_mean_pooling = tf.reduce_sum(token_emb * mask, axis=1)
    token_mean_pooling = token_mean_pooling / seq_len
    return token_mean_pooling


def max_pooling(token_emb, seq_len):
    mask = tf.sequence_mask(seq_len, dtype=tf.float32)
    mask = (1.0 - mask) * -10000.0
    mask = tf.expand_dims(mask, axis=-1)

    token_mean_pooling = tf.reduce_max(token_emb + mask, axis=1)
    return token_mean_pooling
