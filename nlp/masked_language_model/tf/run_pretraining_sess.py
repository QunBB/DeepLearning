import os
import tensorflow as tf

import modeling
import optimization
from config import parse_args


FLAGS = parse_args()


def init_from_checkpoint(init_checkpoint):
    tvars = tf.trainable_variables()

    initialized_variable_names = {}
    if init_checkpoint:
        (assignment_map, initialized_variable_names
         ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
        tf.logging.info('Restoring parameters from %s' % init_checkpoint)

    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
        init_string = ""
        if var.name in initialized_variable_names:
            init_string = ", *INIT_FROM_CKPT*"
        tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                        init_string)


def eval_metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
              masked_lm_weights, next_sentence_example_loss=None,
              next_sentence_log_probs=None, next_sentence_labels=None):
    """Computes the loss and accuracy of the model."""
    metrics = {
               # for test
               # 'num_accumulate': tf.metrics.true_positives(tf.ones_like(next_sentence_labels, dtype=tf.int32),
               #                                             tf.ones_like(next_sentence_labels, dtype=tf.int32),
               #                                             name='num_accumulate_eval_metrics')
               }

    masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                     [-1, masked_lm_log_probs.shape[-1]])
    masked_lm_predictions = tf.argmax(
        masked_lm_log_probs, axis=-1, output_type=tf.int32)
    masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
    masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
    masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
    masked_lm_accuracy = tf.metrics.accuracy(
        labels=masked_lm_ids,
        predictions=masked_lm_predictions,
        weights=masked_lm_weights,
        name='masked_lm_accuracy_eval_metrics')
    masked_lm_mean_loss = tf.metrics.mean(
        values=masked_lm_example_loss, weights=masked_lm_weights, name='masked_lm_mean_loss_eval_metrics')

    metrics.update({"masked_lm_accuracy": masked_lm_accuracy,
                    "masked_lm_loss": masked_lm_mean_loss})

    if next_sentence_labels is not None:
        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions, name='next_sentence_accuracy_eval_metrics')
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss, name='next_sentence_mean_loss_eval_metrics')
        metrics['next_sentence_accuracy'] = next_sentence_accuracy
        metrics['next_sentence_loss'] = next_sentence_mean_loss

    return metrics


def model_fn(is_training, bert_config,
             input_ids, input_mask, segment_ids, masked_lm_positions,
             masked_lm_ids, masked_lm_weights, next_sentence_labels=None,
             ):
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        scope='bert')

    train_metrics = {}

    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
        bert_config, model.get_sequence_output(), model.get_embedding_table(),
        masked_lm_positions, masked_lm_ids, masked_lm_weights)
    train_metrics['masked_lm_loss'] = masked_lm_loss

    next_sentence_example_loss, next_sentence_log_probs = None, None
    if next_sentence_labels is not None:
        (next_sentence_loss, next_sentence_example_loss,
         next_sentence_log_probs) = get_next_sentence_output(
            bert_config, model.get_pooled_output(), next_sentence_labels)
        total_loss = masked_lm_loss + next_sentence_loss
        train_metrics['next_sentence_loss'] = next_sentence_loss

    else:
        total_loss = masked_lm_loss
    train_metrics['total_loss'] = total_loss

    if is_training:
        return train_metrics

    else:
        eval_metrics = eval_metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                                      masked_lm_weights, next_sentence_example_loss,
                                      next_sentence_log_probs, next_sentence_labels)
        return eval_metrics


def get_tower_inputs(features, num_towers):
    for name in ['input_ids', 'input_mask', 'segment_ids', 'masked_lm_positions', 'masked_lm_ids', 'masked_lm_weights']:
        features[name] = tf.split(features[name], num_towers)
    if "next_sentence_labels" in features:
        features["next_sentence_labels"] = tf.split(features["next_sentence_labels"], num_towers)
    else:
        features["next_sentence_labels"] = [None] * num_towers
    tower_inputs = [{key: value[i] for key, value in features.items()} for i in range(num_towers)]
    return tower_inputs


def model_fn_builder(train_features, bert_config, learning_rate,
                     num_train_steps, num_warmup_steps,
                     optimizer, poly_power,
                     start_warmup_step, device_ids, epsilon,
                     eval_features=None):

    tf.logging.info("*** Features ***")
    for name in sorted(train_features.keys()):
        tf.logging.info("  name = %s, shape = %s" % (name, train_features[name].shape))

    num_towers = len(device_ids)

    # optimizer
    optimizer = optimization.create_optimizer(
        None, learning_rate, num_train_steps, num_warmup_steps,
        False, optimizer, poly_power, start_warmup_step, epsilon, return_optimizer=True)

    # train op
    train_inputs = get_tower_inputs(train_features, num_towers)
    train_metrics_list = {}
    tower_gradients = []
    for i, device in enumerate(device_ids):
        with tf.device(device):
            with tf.variable_scope('', reuse=tf.AUTO_REUSE):
                _inputs = {'is_training': True, 'bert_config': bert_config}
                _inputs.update(train_inputs[i])
                train_metrics = model_fn(**_inputs)
                gradients = optimizer.compute_gradients(train_metrics['total_loss'], colocate_gradients_with_ops=False)
                tower_gradients.append(gradients)
                for name in train_metrics:
                    train_metrics_list.setdefault(name, [])
                    train_metrics_list[name] = train_metrics[name]

    train_metrics = {k: tf.reduce_mean(v) for k, v in train_metrics_list.items()}

    if num_towers > 1:
        merged_gradients = optimization.combine_gradients(tower_gradients)
    else:
        merged_gradients = tower_gradients[0]
    merged_gradients = optimization.clip_gradient_norms(merged_gradients, 1.0)
    train_op = optimizer.apply_gradients(merged_gradients, global_step=None)

    # eval op
    if eval_features is not None:
        # eval just using first gpu
        eval_inputs = get_tower_inputs(eval_features, num_towers=1)[0]
        with tf.variable_scope('', reuse=tf.AUTO_REUSE):
            _inputs = {'is_training': False, 'bert_config': bert_config}
            _inputs.update(eval_inputs)
            eval_metrics = model_fn(**_inputs)

        return train_op, train_metrics, eval_metrics

    return train_op, train_metrics, None


def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
    """Get loss and log probs for the masked LM."""
    input_tensor = gather_indexes(input_tensor, positions)

    with tf.variable_scope("cls/predictions"):
        # We apply one more non-linear transformation before the output layer.
        # This matrix is not used after pre-training.
        with tf.variable_scope("transform"):
            input_tensor = tf.layers.dense(
                input_tensor,
                units=bert_config.hidden_size,
                activation=modeling.get_activation(bert_config.hidden_act),
                kernel_initializer=modeling.create_initializer(
                    bert_config.initializer_range))
            input_tensor = modeling.layer_norm(input_tensor)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        output_bias = tf.get_variable(
            "output_bias",
            shape=[bert_config.vocab_size],
            initializer=tf.zeros_initializer())
        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)

        label_ids = tf.reshape(label_ids, [-1])
        label_weights = tf.reshape(label_weights, [-1])

        one_hot_labels = tf.one_hot(
            label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

        # The `positions` tensor might be zero-padded (if the sequence is too
        # short to have the maximum number of predictions). The `label_weights`
        # tensor has a value of 1.0 for every real prediction and 0.0 for the
        # padding predictions.
        per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
        numerator = tf.reduce_sum(label_weights * per_example_loss)
        denominator = tf.reduce_sum(label_weights) + 1e-5
        loss = numerator / denominator

    return (loss, per_example_loss, log_probs)


def get_next_sentence_output(bert_config, input_tensor, labels):
    """Get loss and log probs for the next sentence prediction."""

    # Simple binary classification. Note that 0 is "next sentence" and 1 is
    # "random sentence". This weight matrix is not used after pre-training.
    with tf.variable_scope("cls/seq_relationship"):
        output_weights = tf.get_variable(
            "output_weights",
            shape=[2, bert_config.hidden_size],
            initializer=modeling.create_initializer(bert_config.initializer_range))
        output_bias = tf.get_variable(
            "output_bias", shape=[2], initializer=tf.zeros_initializer())

        logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
        logits = tf.nn.bias_add(logits, output_bias)
        log_probs = tf.nn.log_softmax(logits, axis=-1)
        labels = tf.reshape(labels, [-1])
        one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
        per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
        loss = tf.reduce_mean(per_example_loss)
        return (loss, per_example_loss, log_probs)


def gather_indexes(sequence_tensor, positions):
    """Gathers the vectors at the specific positions over a minibatch."""
    sequence_shape = modeling.get_shape_list(sequence_tensor, expected_rank=3)
    batch_size = sequence_shape[0]
    seq_length = sequence_shape[1]
    width = sequence_shape[2]

    flat_offsets = tf.reshape(
        tf.range(0, batch_size, dtype=tf.int32) * seq_length, [-1, 1])
    flat_positions = tf.reshape(positions + flat_offsets, [-1])
    flat_sequence_tensor = tf.reshape(sequence_tensor,
                                      [batch_size * seq_length, width])
    output_tensor = tf.gather(flat_sequence_tensor, flat_positions)
    return output_tensor


def input_fn_builder(input_files,
                     batch_size,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4,
                     nsp_or_sop=True):
    """The actual input function."""

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
    }
    if nsp_or_sop:
        name_to_features["next_sentence_labels"] = tf.FixedLenFeature([1], tf.int64)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
        d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
        # d = d.repeat()
        d = d.shuffle(buffer_size=len(input_files))

        # `cycle_length` is the number of parallel files that get read.
        cycle_length = min(num_cpu_threads, len(input_files))

        # `sloppy` mode means that the interleaving is not exact. This adds
        # even more randomness to the training pipeline.
        d = d.apply(
            tf.contrib.data.parallel_interleave(
                tf.data.TFRecordDataset,
                sloppy=is_training,
                cycle_length=cycle_length))
        d = d.shuffle(buffer_size=100)
    else:
        d = tf.data.TFRecordDataset(input_files)
        # Since we evaluate for a fixed number of steps we don't want to encounter
        # out-of-range exceptions.
        # d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True if is_training else False))
    iterator = d.make_initializable_iterator()
    batch = iterator.get_next()
    return iterator, batch


def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
        t = example[name]
        if t.dtype == tf.int64:
            t = tf.to_int32(t)
        example[name] = t

    return example


def get_input_files(dataset_tfrecord_dir):
    input_files = []
    for input_pattern in os.listdir(dataset_tfrecord_dir):
        if input_pattern.endswith('.tfrecord'):
            input_files.extend(tf.gfile.Glob(os.path.join(dataset_tfrecord_dir, input_pattern)))

    tf.logging.info("*** Input Files ***")
    for input_file in input_files:
        tf.logging.info("  %s" % input_file)

    return input_files


def validate(sess, saver, eval_iterator, eval_metrics, save_type_tuple,
             early_stop=False, best_perf=None, max_steps_without_increase=None):
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    sess.run(eval_iterator.initializer)

    while True:
        try:
            # run the metric update op
            sess.run([metric[1] for metric in eval_metrics.values()])
            # sess.run(eval_metrics)
        except tf.errors.OutOfRangeError:
            tf.logging.info("***** End evaluation *****")
            break
    eval_metrics_result = sess.run({k: v[0] for k, v in eval_metrics.items()})

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "a") as writer:
        tf.logging.info("***** Eval results *****")
        writer.write("%s = %s\n" % save_type_tuple)
        for key in sorted(eval_metrics_result.keys()):
            tf.logging.info("  %s = %s", key, str(eval_metrics_result[key]))
            writer.write("%s = %s\n" % (key, str(eval_metrics_result[key])))
        writer.write("\n")

    # reset eval metrics
    sess.run(tf.initialize_variables([v for v in tf.local_variables() if 'eval_metrics' in v.name]))

    if early_stop:
        key_name = 'masked_lm_accuracy'
        _name = 'steps_without_increase'
        if best_perf.get(key_name) is None or eval_metrics_result[key_name] > best_perf[key_name]:
            best_perf[key_name] = eval_metrics_result[key_name]
            best_perf[_name] = 0  # reset `steps_without_increase`
        else:
            if best_perf.get(_name) is None or best_perf[_name] < max_steps_without_increase - 1:
                best_perf[_name] = best_perf.get(_name, 0) + 1
                tf.logging.info(f'No increase in metric "{key_name}" for {save_type_tuple[1]} {save_type_tuple[0]}, skip saving model')
                return True
            else:
                tf.logging.info(f'No increase in metric "{key_name}" which is greater than or equal to max steps '
                                f'({max_steps_without_increase}) configured for early stopping.')
                tf.logging.info(f'Requesting early stopping at {save_type_tuple[0]} {save_type_tuple[1]}')
                return False

    saver.save(sess, save_path=os.path.join(FLAGS.output_dir, f'model-{save_type_tuple[0]}-{save_type_tuple[1]}.ckpt'))

    return True


def main():
    tf.logging.set_verbosity(tf.logging.INFO)
    # avoid logging double
    logger = tf.get_logger()
    logger.propagate = False

    bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

    nsp_or_sop = FLAGS.random_next_sentence or FLAGS.sentence_order_prediction

    tf.gfile.MakeDirs(FLAGS.output_dir)

    if FLAGS.device == 'cuda':
        device_ids = ["/gpu:"+index for index in FLAGS.device_ids.split(',')]
    else:
        device_ids = ["/cpu:0"]
    tf.logging.info(f'****** train device ids: {device_ids} ******')

    train_iterator, train_dataset = input_fn_builder(
        input_files=get_input_files(FLAGS.train_tfrecord_dir),
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True,
        batch_size=FLAGS.train_batch_size * len(device_ids),
        nsp_or_sop=nsp_or_sop)

    eval_iterator, eval_dataset = input_fn_builder(
        input_files=get_input_files(FLAGS.eval_tfrecord_dir),
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False,
        batch_size=FLAGS.eval_batch_size,
        nsp_or_sop=nsp_or_sop)

    # update op of global step
    global_step = tf.train.get_or_create_global_step()
    new_global_step = global_step + 1
    global_step_add_op = global_step.assign(new_global_step)

    train_op, train_metrics, eval_metrics = model_fn_builder(train_features=train_dataset,
                                                             bert_config=bert_config,
                                                             learning_rate=FLAGS.learning_rate,
                                                             num_train_steps=FLAGS.num_train_steps,
                                                             num_warmup_steps=FLAGS.num_warmup_steps,
                                                             optimizer=FLAGS.optimizer, poly_power=FLAGS.poly_power,
                                                             start_warmup_step=FLAGS.start_warmup_step,
                                                             epsilon=FLAGS.adam_epsilon,
                                                             device_ids=device_ids,
                                                             eval_features=eval_dataset)

    if FLAGS.init_checkpoint is not None:
        init_from_checkpoint(FLAGS.init_checkpoint)

    step = 1
    epoch = 1
    best_perf = {}
    saver = tf.train.Saver(max_to_keep=FLAGS.keep_checkpoint_max)
    sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
    sess_config.gpu_options.allow_growth = True
    with tf.Session(config=sess_config) as sess:
        sess.run(train_iterator.initializer)
        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        tf.logging.info("***** Running training *****")
        tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)

        while step < FLAGS.num_train_steps:
            try:
                _, train_metrics_result = sess.run([train_op, train_metrics])
                sess.run(global_step_add_op)

                if step % FLAGS.print_steps == 0:
                    log = f'step: {step} | '
                    log += '| '.join([f'{k}: {v}' for k, v in train_metrics_result.items()])
                    tf.logging.info(log)

                if not FLAGS.save_checkpoints_per_epoch and step % FLAGS.save_checkpoints_steps == 0:
                    flag = validate(sess, saver, eval_iterator, eval_metrics, save_type_tuple=('step', step),
                                    early_stop=FLAGS.early_stop, best_perf=best_perf,
                                    max_steps_without_increase=FLAGS.max_steps_without_increase)
                    if not flag:
                        return

                step += 1

            except tf.errors.OutOfRangeError:
                tf.logging.info('*'*20 + f' epoch: {epoch} end ' + '*'*20)

                if FLAGS.save_checkpoints_per_epoch:
                    flag = validate(sess, saver, eval_iterator, eval_metrics, save_type_tuple=('epoch', epoch),
                                    early_stop=FLAGS.early_stop, best_perf=best_perf,
                                    max_steps_without_increase=FLAGS.max_steps_without_increase)
                    if not flag:
                        return

                if epoch >= FLAGS.max_epochs:
                    tf.logging.info('***** max_epochs limit, exit *****')
                    break

                # reset train dataset
                sess.run(train_iterator.initializer)

                epoch += 1


if __name__ == '__main__':
    main()
