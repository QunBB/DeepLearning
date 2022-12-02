import os
import random
import collections
import numpy as np
from transformers import AutoTokenizer
import tokenizers
from typing import Union, Optional
from tfrecord import TFRecordWriter
from tfrecord.tools import create_index
import shutil
# import synonyms

from config import parse_args


CLS_TOKEN = '[CLS]'
SEP_TOKEN = '[SEP]'
MASK_TOKEN = '[MASK]'
PAD_TOKEN = '[PAD]'
CLS_TOKEN_ID = 101
SEP_TOKEN_ID = 102
MASK_TOKEN_ID = 103
PAD_TOKEN_ID = 0
FLAGS = parse_args()

current_writer_index = 0


def get_tokenizer(model_name, cache_dir):
    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir=cache_dir, use_fast=True)
    global CLS_TOKEN, SEP_TOKEN, MASK_TOKEN, PAD_TOKEN, CLS_TOKEN_ID, SEP_TOKEN_ID, MASK_TOKEN_ID, PAD_TOKEN_ID
    CLS_TOKEN = tokenizer.cls_token
    SEP_TOKEN = tokenizer.sep_token
    MASK_TOKEN = tokenizer.mask_token
    PAD_TOKEN = tokenizer.pad_token
    CLS_TOKEN_ID = tokenizer.cls_token_id
    SEP_TOKEN_ID = tokenizer.sep_token_id
    MASK_TOKEN_ID = tokenizer.mask_token_id
    PAD_TOKEN_ID = tokenizer.pad_token_id

    return tokenizer


class TrainingInstance(object):
    """A single training instance (sentence pair)."""

    def __init__(self, input_ids, segment_ids, masked_lm_labels, is_random_next):
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.is_random_next = is_random_next
        self.masked_lm_labels = masked_lm_labels


def get_masked_cand_indexes(encodings, do_whole_word_mask=True, stop_words=set()):
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    cand_indexes = []
    for (i, token) in enumerate(encodings.tokens):
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


def get_masked_cand_indexes_wwm(encodings, stop_words):
    import jieba

    offsets = encodings.offsets
    text = encodings.text
    max_ofs = max(offsets, key=lambda x: x[1])[1]

    cand_indexes = []
    cand_words = []
    ofs_start = 1
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


def write_instance_to_example_files(instances, max_seq_length, writers):
    """Create TF example files from `TrainingInstance`s."""
    global current_writer_index

    total_written = 0
    for (inst_index, instance) in enumerate(instances):
        input_ids = list(instance.input_ids)
        input_mask = [1] * len(input_ids)
        segment_ids = list(instance.segment_ids)
        masked_lm_labels = list(instance.masked_lm_labels)
        assert len(input_ids) <= max_seq_length

        while len(input_ids) < max_seq_length:
            input_ids.append(PAD_TOKEN_ID)
            input_mask.append(0)
            segment_ids.append(0)
            masked_lm_labels.append(-100)

        assert len(input_ids) == max_seq_length
        assert len(input_mask) == max_seq_length
        assert len(segment_ids) == max_seq_length
        assert len(masked_lm_labels) == max_seq_length

        features = collections.OrderedDict()
        features["input_ids"] = (input_ids, 'int')
        features["attention_mask"] = (input_mask, 'int')
        features["token_type_ids"] = (segment_ids, 'int')
        features["labels"] = (masked_lm_labels, 'int')
        if instance.is_random_next is not None:
            sentence_order_label = 1 if instance.is_random_next else 0
            features["next_sentence_label"] = ([sentence_order_label, 'int'])

        writers[current_writer_index].write(features)
        current_writer_index = (current_writer_index + 1) % len(writers)

        total_written += 1

    print("Wrote %d total instances", total_written)


def create_masked_lm_predictions(encodings, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng, stop_words):
    """Creates the predictions for the masked LM objective."""

    tokens = encodings.tokens

    if FLAGS.do_whole_word_mask_cn:
        cand_indexes, _ = get_masked_cand_indexes_wwm(encodings, stop_words)
    else:
        cand_indexes = get_masked_cand_indexes(encodings, FLAGS.do_whole_word_mask, stop_words)

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

    masked_lm_labels = [-100] * len(tokens)
    count = 0  # the number of masked tokens

    covered_indexes = set()
    for cand_index_set in ngram_indexes:
        if count >= num_to_predict:
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
        while count + len(index_set) > num_to_predict:
            if n == 0:
                break
            index_set = sum(cand_index_set[n - 1], [])
            n -= 1
        # If adding a whole-word mask would exceed the maximum number of
        # predictions, then just skip this candidate.
        if count + len(index_set) > num_to_predict:
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
            count += 1

            masked_lm_labels[index] = encodings.ids[index]

            masked_token = None
            # 80% of the time, replace with [MASK]
            if rng.random() < 0.8:
                masked_token_id = MASK_TOKEN_ID
                masked_token = MASK_TOKEN
            else:
                # 10% of the time, keep original
                if rng.random() < 0.5:
                    masked_token_id = encodings.ids[index]
                    masked_token = '*' + encodings.tokens[index]  # add `*` just for print
                # 10% of the time, replace with random word
                else:
                    masked_token_id = rng.randint(0, len(vocab_words) - 1)
                    masked_token = '!' + vocab_words[masked_token_id]  # add `!` just for print

            encodings.ids[index] = masked_token_id
            encodings.tokens[index] = masked_token

    assert count <= num_to_predict

    return encodings, masked_lm_labels


def create_instances_from_document(
        all_documents, document_index, max_seq_length, short_seq_prob,
        masked_lm_prob, max_predictions_per_seq, vocab_words, rng, stop_words):
    """Creates `TrainingInstance`s for a single document."""
    document = all_documents[document_index]

    if FLAGS.random_next_sentence or FLAGS.sentence_order_prediction:
        # Account for [CLS], [SEP], [SEP]
        max_num_tokens = max_seq_length - 3
    else:
        # Account for [CLS], [SEP]
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

                tokens_a = Encodings()  # empty Encodings
                for j in range(a_end):
                    tokens_a.extend(current_chunk[j])

                tokens_b = Encodings()

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

                tokens = tokens_a
                tokens.add_special_tokens(encodings=tokens_b)
                if document_index < 20:
                    print("document index: %d" % (document_index))
                    print("text: %s" % tokens.text)
                    print("origin tokens: %s" % (" ".join([x for x in tokens.tokens])))
                    print("origin tokens id: %s" % (" ".join([str(x) for x in tokens.ids])))

                tokens, masked_lm_labels = create_masked_lm_predictions(
                    tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng, stop_words)

                if document_index < 20:
                    print_masked_encodings(tokens, masked_lm_labels, is_random_next)

                instance = TrainingInstance(
                    input_ids=tokens.ids,
                    segment_ids=tokens.type_ids,
                    is_random_next=is_random_next,
                    masked_lm_labels=masked_lm_labels)
                instances.append(instance)
            current_chunk = []
            current_length = 0
        i += 1

    return instances


def print_masked_encodings(encodings, masked_lm_labels, is_random_next):
    s = ""
    # s += "text: %s\n" % encodings.text
    s += "masked tokens: %s\n" % (" ".join([x for x in encodings.tokens]))
    s += "masked tokens id: %s\n" % (" ".join([str(x) for x in encodings.ids]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in encodings.type_ids]))
    s += "masked_lm_labels: %s\n" % (" ".join([str(x) for x in masked_lm_labels]))
    s += "is_random_next: %s\n" % is_random_next
    s += "\n"
    print(s)


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
            trunc_tokens.remove_first()
        else:
            trunc_tokens.pop()


class BaseEncoding:
    ids = None
    tokens = None
    text = None
    offsets = None
    type_ids = None

    def __len__(self):
        raise NotImplementedError


class Encodings(BaseEncoding):
    def __init__(self, encodings: Optional[Union[tokenizers.Encoding, BaseEncoding]] = None,
                 text: Optional[str] = None):
        super().__init__()

        if encodings is not None and len(encodings) > 0:
            self._init(encodings, text)

    def _init(self, encodings: Union[tokenizers.Encoding, BaseEncoding],
              text: Optional[str] = None):
        assert isinstance(encodings, BaseEncoding) or text is not None
        self.text = encodings.text if isinstance(encodings, BaseEncoding) else text

        self.ids = encodings.ids
        self.tokens = encodings.tokens
        self.offsets = encodings.offsets
        self.type_ids = encodings.type_ids

    def __len__(self):
        if self.ids is None:
            return 0
        return len(self.ids)

    def append(self, encodings: BaseEncoding, pair: Optional[bool] = False):
        if len(encodings) > 0:
            if self.ids is None:
                self._init(encodings)
                return

            self.ids.extend(encodings.ids)
            self.tokens.extend(encodings.tokens)
            if not pair:
                self.type_ids.extend(encodings.type_ids)
            else:
                self.type_ids.extend([1] * len(encodings.type_ids))

            self.text = self.real_text + '\n' + encodings.text
            index = self.end_index + 1
            self.offsets.extend([(offset[0]+index, offset[1]+index) if offset != (0, 0) else (0, 0) for offset in encodings.offsets])

    def extend(self, encodings: BaseEncoding):
        if isinstance(encodings, list):
            for e in encodings:
                self.append(e)
        else:
            self.append(encodings)

    def pop(self):
        self.ids.pop()
        self.tokens.pop()
        self.offsets.pop()
        self.type_ids.pop()

    def remove_first(self):
        del self.ids[0]
        del self.tokens[0]
        del self.offsets[0]
        del self.type_ids[0]

    def __str__(self):
        s = ""
        s += "text: %s\n" % self.text
        s += "tokens: %s\n" % (" ".join([x for x in self.tokens]))
        s += "tokens id: %s\n" % (" ".join([str(x) for x in self.ids]))
        s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.type_ids]))
        return s

    @property
    def end_index(self):
        end_index = None
        for ofs in self.offsets[::-1]:
            if ofs != (0, 0):
                end_index = ofs[1]
                break
        assert end_index is not None
        return end_index

    @property
    def start_index(self):
        start_index = None
        for ofs in self.offsets:
            if ofs != (0, 0):
                start_index = ofs[0]
                break
        assert start_index is not None
        return start_index

    @property
    def real_text(self):
        start_index = self.start_index
        end_index = self.end_index
        assert start_index < end_index

        return self.text[start_index:end_index]

    def add_special_tokens(self, encodings: Optional[BaseEncoding] = None):
        self.ids = [CLS_TOKEN_ID] + self.ids
        self.tokens = [CLS_TOKEN] + self.tokens
        self.type_ids = [0] + self.type_ids
        self.offsets = [(0, 0)] + self.offsets
        self.ids.append(SEP_TOKEN_ID)
        self.tokens.append(SEP_TOKEN)
        self.type_ids.append(0)
        self.offsets.append((0, 0))

        if encodings is not None and len(encodings) > 0:
            encodings.ids.append(SEP_TOKEN_ID)
            encodings.tokens.append(SEP_TOKEN)
            encodings.type_ids.append(0)
            encodings.offsets.append((0, 0))

            self.append(encodings, pair=True)


def main():
    stop_words = set()
    if FLAGS.stop_words_file is not None:
        with open(FLAGS.stop_words_file, 'r') as file:
            while True:
                word = file.readline()
                if not word:
                    break
                stop_words.add(word.strip())

    tokenizer = get_tokenizer(FLAGS.model_name, cache_dir=FLAGS.cache_dir)
    vocab_words = list(tokenizer.vocab.keys())
    rng = random.Random(FLAGS.seed)

    if os.path.exists(FLAGS.output_tfrecord_dir):
        print(f'{FLAGS.output_tfrecord_dir} exists, will remove files in the dir')
        shutil.rmtree(FLAGS.output_tfrecord_dir)
        os.makedirs(FLAGS.output_tfrecord_dir)
    else:
        print(f'{FLAGS.output_tfrecord_dir} not exists, will create')
        os.makedirs(FLAGS.output_tfrecord_dir)

    writers = []
    for index in range(FLAGS.num_output_tfrecord):
        writers.append(TFRecordWriter(os.path.join(FLAGS.output_tfrecord_dir, f'example_{index}.tfrecord')))

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
        write_instance_to_example_files(instances, FLAGS.max_seq_length, writers)

    for _ in range(FLAGS.dupe_factor):
        for text_file in FLAGS.input_files.split(','):
            with open(text_file, 'r') as reader:
                while True:
                    line = reader.readline()
                    if not line:
                        break

                    line = line.strip()

                    # NSP or SOP: Empty lines are used as document delimiters
                    if (not nsp_or_sop) or (not line):
                        all_documents.append([])
                        total += 1

                    tokens = tokenizer(line, padding=False, add_special_tokens=False)._encodings[0]
                    tokens = Encodings(tokens, line)
                    if len(tokens) > 0:
                        all_documents[-1].append(tokens)

                        if FLAGS.random_queue_size > 0 and total % FLAGS.random_queue_size == 0:
                            _queue_write_to_files()

                            # reset the queue of all documents
                            all_documents = []

    _queue_write_to_files()

    for writer in writers:
        writer.close()

    for index in range(FLAGS.num_output_tfrecord):
        create_index(os.path.join(FLAGS.output_tfrecord_dir, f'example_{index}.tfrecord'),
                     os.path.join(FLAGS.output_tfrecord_dir, f'example_{index}.index'))


if __name__ == '__main__':
    main()
