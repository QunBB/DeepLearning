import os
import random
import collections
import numpy as np
from typing import Union, Optional
import shutil
import tensorflow as tf

# import synonyms

import tokenization
from config import parse_args


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
FLAGS = parse_args()

current_writer_index = 0


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
                 is_random_next):
        self.tokens = tokens
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_positions = masked_lm_positions
        self.masked_lm_labels = masked_lm_labels

    def __str__(self):
        s = ""
        s += "tokens: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.tokens]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
        s += "is_random_next: %s\n" % self.is_random_next
        s += "masked_lm_positions: %s\n" % (" ".join(
            [str(x) for x in self.masked_lm_positions]))
        s += "masked_lm_labels: %s\n" % (" ".join(
            [tokenization.printable_text(x) for x in self.masked_lm_labels]))
        s += "\n"
        return s

    def __repr__(self):
        return self.__str__()


def get_masked_cand_indexes(tokens, do_whole_word_mask=True, stop_words=set()):
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    cand_indexes = []
    for (i, token) in enumerate(tokens):
        if token == CLS_TOKEN or token == SEP_TOKEN:
            continue

        if token in stop_words:
            continue

        if (do_whole_word_mask and len(cand_indexes) >= 1 and
                token.startswith("##")):
            cand_indexes[-1].append(i)
        else:
            cand_indexes.append([i])

    return cand_indexes


def get_masked_cand_indexes_wwm(tokens, stop_words):
    import jieba

    offsets = [(0, 0)]
    text = []
    for token in tokens:
        if token in (CLS_TOKEN, SEP_TOKEN, MASK_TOKEN):
            offsets.append((offsets[-1][1], offsets[-1][1]))
        elif token.startswith('##'):
            text.extend(list(token[2:]))
            offsets.append((offsets[-1][1], offsets[-1][1]+len(token[2:])))
        elif 65 <= ord(token[0]) <= 90 or 97 <= ord(token[0]) <= 122:  # english word
            text.append(' ')
            text.extend(list(token))
            offsets.append((offsets[-1][1]+1, offsets[-1][1]+1+len(token)))
        else:
            text.extend(list(token))
            offsets.append((offsets[-1][1], offsets[-1][1]+len(token)))

    text = ''.join(text)
    del offsets[0]

    max_ofs = max(offsets, key=lambda x: x[1])[1]

    cand_indexes = []
    cand_words = []
    ofs_start = 0
    text_start = 0

    for word in jieba.cut(text):
        if text_start >= max_ofs:
            break

        if word in stop_words:
            text_start += len(word)
            continue

        # put indexes including the whole word into `index_set`
        # almost, one `index_set` is corresponding to one word's indexes
        index_set = []
        for text_index in range(text_start, text_start+len(word)):
            for ofs_index in range(ofs_start, len(offsets)):
                if offsets[ofs_index][0] <= text_index < offsets[ofs_index][1]:
                    index_set.append(ofs_index)
                    ofs_start = ofs_index + 1
                    break
                elif offsets[ofs_index][0] > text_index:
                    break
        text_start += len(word)
        if index_set:
            cand_indexes.append(index_set)
            cand_words.append(word)

    return cand_indexes, cand_words


def create_int_feature(values):
    feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    return feature


def create_float_feature(values):
    feature = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
    return feature


def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, writers):
    """Create TF example files from `TrainingInstance`s."""
    global current_writer_index

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length

        masked_lm_positions = list(instance.masked_lm_positions)
        masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
        masked_lm_weights = [1.0] * len(masked_lm_ids)

        while len(masked_lm_positions) < max_predictions_per_seq:
            masked_lm_positions.append(0)
            masked_lm_ids.append(0)
            masked_lm_weights.append(0.0)

        features = collections.OrderedDict()
        features["input_ids"] = create_int_feature(input_ids)
        features["input_mask"] = create_int_feature(input_mask)
        features["segment_ids"] = create_int_feature(segment_ids)
        features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
        features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
        features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
        if instance.is_random_next is not None:
            next_sentence_label = 1 if instance.is_random_next else 0
            features["next_sentence_labels"] = create_int_feature([next_sentence_label])

        tf_example = tf.train.Example(features=tf.train.Features(feature=features))

        writers[current_writer_index].write(tf_example.SerializeToString())
        current_writer_index = (current_writer_index + 1) % len(writers)

        total_written += 1

        if inst_index < 20:
            tf.logging.info("*** Example ***")
            tf.logging.info("tokens: %s" % " ".join(
                [tokenization.printable_text(x) for x in instance.tokens]))

            for feature_name in features.keys():
                feature = features[feature_name]
                values = []
                if feature.int64_list.value:
                    values = feature.int64_list.value
                elif feature.float_list.value:
                    values = feature.float_list.value
                tf.logging.info(
                    "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

    tf.logging.info("Wrote %d total instances", total_written)


MaskedLmInstance = collections.namedtuple("MaskedLmInstance",
                                          ["index", "label"])


def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, stop_words):
    """Creates the predictions for the masked LM objective."""

    if FLAGS.do_whole_word_mask_cn:
        cand_indexes, _ = get_masked_cand_indexes_wwm(tokens, stop_words)
    else:
        cand_indexes = get_masked_cand_indexes(tokens, FLAGS.do_whole_word_mask, stop_words)

    num_to_predict = min(max_predictions_per_seq,
                         max(1, int(round(len(tokens) * masked_lm_prob))))

    # Note(mingdachen):
    # By default, we set the probilities to favor shorter ngram sequences.
    ngrams = np.arange(1, FLAGS.ngram + 1, dtype=np.int64)
    pvals = 1. / np.arange(1, FLAGS.ngram + 1)
    pvals /= pvals.sum(keepdims=True)

    if not FLAGS.favor_shorter_ngram:
        pvals = pvals[::-1]

    ngram_indexes = []
    for idx in range(len(cand_indexes)):
        ngram_index = []
        for n in ngrams:
            ngram_index.append(cand_indexes[idx:idx+n])
        ngram_indexes.append(ngram_index)

    rng.shuffle(ngram_indexes)

    output_tokens = list(tokens)

    masked_lm_positions = []
    masked_lm_labels = []
    masked_lms = []

    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if len(masked_lms) >= num_to_predict:
            break
        if not cand_index_set:
            continue
        # Note(mingdachen):
        # Skip current piece if they are covered in lm masking or previous ngrams.
        for index_set in cand_index_set[0]:
            for index in index_set:
                if index in covered_indexes:
                    continue

        n = np.random.choice(ngrams[:len(cand_index_set)],
                             p=pvals[:len(cand_index_set)] /
                               pvals[:len(cand_index_set)].sum(keepdims=True))
        index_set = sum(cand_index_set[n - 1], [])
        n -= 1
        # Note(mingdachen):
        # Repeatedly looking for a candidate that does not exceed the
        # maximum number of predictions by trying shorter ngrams.
        while len(masked_lms) + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if len(masked_lms) + len(index_set) > num_to_predict:
            continue
        is_any_index_covered = False
        for index in index_set:
            if index in covered_indexes:
                is_any_index_covered = True
                break
        if is_any_index_covered:
            continue
        for index in index_set:
            covered_indexes.add(index)

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token = MASK_TOKEN
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token = tokens[index]
                # 10% of the time, replace with random word
                else:
                    masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

            output_tokens[index] = masked_token

            masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))

    assert len(masked_lms) <= num_to_predict
    masked_lms = sorted(masked_lms, key=lambda x: x.index)

    for p in masked_lms:
        masked_lm_positions.append(p.index)
        masked_lm_labels.append(p.label)

    return output_tokens, masked_lm_positions, masked_lm_labels


def print_masked_encodings(tokens, masked_lm_positions, masked_lm_labels):
    def _get_token(t, i):
        if i not in masked_lm_positions:
            return t

        if t == MASK_TOKEN:
            return t
        # just for print
        if t == masked_lm_labels[masked_lm_positions.index(i)]:
            return '*' + t  # add `*` for tokens which keep original
        return '!' + t  # add `!` for tokens which are replaced with random word

    s = "masked tokens: %s" % (" ".join([_get_token(x, i) for i, x in enumerate(tokens)]))
    tf.logging.info(s)


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, stop_words):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    if FLAGS.random_next_sentence or FLAGS.sentence_order_prediction:
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = max_seq_length - 3
    else:
        # Account for [CLS], [SEP] when no NSP or SOP task
        max_num_tokens = max_seq_length - 2

    # We *usually* want to fill up the entire sequence since we are padding
    # to `max_seq_length` anyways, so short sequences are generally wasted
    # computation. However, we *sometimes*
    # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
    # sequences to minimize the mismatch between pre-training and fine-tuning.
    # The `target_seq_length` is just a rough target however, whereas
    # `max_seq_length` is a hard limit.
    target_seq_length = max_num_tokens
    if rng.random() < short_seq_prob:
        target_seq_length = rng.randint(2, max_num_tokens)

    # We DON'T just concatenate all of the tokens from a document into a long
    # sequence and choose an arbitrary split point because this would make the
    # next sentence prediction task too easy. Instead, we split the input into
    # segments "A" and "B" based on the actual "sentences" provided by the user
    # input.
    instances = []
    current_chunk = []
    current_length = 0
    i = 0
    while i < len(document):
        segment = document[i]
        current_chunk.append(segment)
        current_length += len(segment)
        if i == len(document) - 1 or current_length >= target_seq_length:
            if current_chunk:
                # `a_end` is how many segments from `current_chunk` go into the `A`
                # (first) sentence.
                a_end = 1
                if len(current_chunk) >= 2:
                    a_end = rng.randint(1, len(current_chunk) - 1)

                tokens_a = []
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = []

                is_random_next = None
                if FLAGS.random_next_sentence or FLAGS.sentence_order_prediction:
                    # Random next
                    if len(current_chunk) == 1 or (FLAGS.random_next_sentence and rng.random() < 0.5):
                        is_random_next = True
                        target_b_length = target_seq_length - len(tokens_a)

                        # This should rarely go for more than one iteration for large
                        # corpora. However, just to be careful, we try to make sure that
                        # the random document is not the same as the document
                        # we're processing.
                        for _ in range(10):
                            random_document_index = rng.randint(0, len(all_documents) - 1)
                            if random_document_index != document_index:
                                break

                        random_document = all_documents[random_document_index]
                        random_start = rng.randint(0, len(random_document) - 1)
                        for j in range(random_start, len(random_document)):
                            tokens_b.extend(random_document[j])
                            if len(tokens_b) >= target_b_length:
                                break
                        # We didn't actually use these segments so we "put them back" so
                        # they don't go to waste.
                        num_unused_segments = len(current_chunk) - a_end
                        i -= num_unused_segments
                    elif rng.random() < 0.5:
                        is_random_next = True
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                        # Note(mingdachen): in this case, we just swap tokens_a and tokens_b
                        tokens_a, tokens_b = tokens_b, tokens_a
                    # Actual next
                    else:
                        is_random_next = False
                        for j in range(a_end, len(current_chunk)):
                            tokens_b.extend(current_chunk[j])
                truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

                assert len(tokens_a) >= 1
                if FLAGS.random_next_sentence or FLAGS.sentence_order_prediction:
                    assert len(tokens_b) >= 1

                tokens = []
                segment_ids = []
                tokens.append(CLS_TOKEN)
                segment_ids.append(0)
                for token in tokens_a:
                    tokens.append(token)
                    segment_ids.append(0)

                tokens.append(SEP_TOKEN)
                segment_ids.append(0)

                for token in tokens_b:
                    tokens.append(token)
                    segment_ids.append(1)
                tokens.append(SEP_TOKEN)
                segment_ids.append(1)

                if document_index < 20:
                    tf.logging.info(f'document_index: {document_index}')
                    tf.logging.info(f'origin tokens: {" ".join(tokens)}')

                (tokens, masked_lm_positions,
                 masked_lm_labels) = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, stop_words)

                if document_index < 20:
                    print_masked_encodings(tokens, masked_lm_positions, masked_lm_labels)

                instance = TrainingInstance(
                    tokens=tokens,
                    segment_ids=segment_ids,
                    is_random_next=is_random_next,
                    masked_lm_positions=masked_lm_positions,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng):
    """Truncates a pair of sequences to a maximum sequence length."""
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        trunc_tokens = tokens_a if len(tokens_a) > len(tokens_b) else tokens_b
        assert len(trunc_tokens) >= 1

        # We want to sometimes truncate from the front and sometimes from the
        # back to add more randomness and avoid biases.
        if rng.random() < 0.5:
            del trunc_tokens[0]
        else:
            trunc_tokens.pop()


def main():

    stop_words = set()
    if FLAGS.stop_words_file is not None:
        with open(FLAGS.stop_words_file, 'r') as file:
            while True:
                word = file.readline()
                if not word:
                    break
                stop_words.add(word.strip())

    tokenizer = tokenization.FullTokenizer(
        vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    vocab_words = list(tokenizer.vocab.keys())
    rng = random.Random(FLAGS.seed)

    if os.path.exists(FLAGS.output_tfrecord_dir):
        tf.logging.warning(f'{FLAGS.output_tfrecord_dir} exists, will remove files in the dir')
        shutil.rmtree(FLAGS.output_tfrecord_dir)
        os.makedirs(FLAGS.output_tfrecord_dir)
    else:
        tf.logging.info(f'{FLAGS.output_tfrecord_dir} not exists, will create')
        os.makedirs(FLAGS.output_tfrecord_dir)

    writers = []
    for index in range(FLAGS.num_output_tfrecord):
        writers.append(tf.python_io.TFRecordWriter(os.path.join(FLAGS.output_tfrecord_dir, f'example_{index}.tfrecord')))

    total = 1
    all_documents = [[]]
    nsp_or_sop = FLAGS.random_next_sentence or FLAGS.sentence_order_prediction

    def _queue_write_to_files():
        instances = []
        for document_index in range(len(all_documents)):
            instances.extend(
                create_instances_from_document(
                    all_documents, document_index, FLAGS.max_seq_length, FLAGS.short_seq_prob,
                    FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq, vocab_words, rng, stop_words))
        rng.shuffle(instances)
        write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, writers)

    for _ in range(FLAGS.dupe_factor):
        for text_file in FLAGS.input_files.split(','):
            with open(text_file, 'r') as reader:
                while True:
                    line = tokenization.convert_to_unicode(reader.readline())
                    if not line:
                        break

                    line = line.strip()

                    # NSP or SOP: Empty lines are used as document delimiters
                    if (not nsp_or_sop) or (not line):
                        all_documents.append([])
                        total += 1

                    tokens = tokenizer.tokenize(line)
                    if len(tokens) > 0:
                        all_documents[-1].append(tokens)

                        if FLAGS.random_queue_size > 0 and total % FLAGS.random_queue_size == 0:
                            _queue_write_to_files()

                            # reset the queue of all documents
                            all_documents = []

    _queue_write_to_files()

    for writer in writers:
        writer.close()


if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    main()

    # stop_words = {'了', '的'}
    # tokenizer = tokenization.FullTokenizer(
    #     vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)
    # # text = '而我们又无法直接排除这些很少的类别的数据，because这些类别也很重要，more even,仍然需要模型去预测这些类别。'
    # text = '缺点是，使用ensemble则会提高了部署的成本和带来性能问题。'
    # text = tokenization.convert_to_unicode(text)
    # tokens = ['[CLS]'] + tokenizer.tokenize(text) + ['[SEP]']
    #
    # cand_indexes, cand_words = get_masked_cand_indexes_wwm(tokens, stop_words)
    # for i in range(len(cand_indexes)):
    #     print(i, [tokens[idx] for idx in cand_indexes[i]], cand_words[i])
    #
    # cand_indexes = get_masked_cand_indexes(tokens, stop_words=stop_words)
    # for i in range(len(cand_indexes)):
    #     print(i, [tokens[idx] for idx in cand_indexes[i]])
