import argparse


def str2bool(v):
    if isinstance(v, bool):
        return v

    if str(v).strip().lower() in ('yes', 'true', 't', 'y', '1'):

        return True

    elif str(v).strip().lower() in ('no', 'false', 'f', 'n', '0'):

        return False

    else:

        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse_args():
    parser = argparse.ArgumentParser(description="Bert Pretraining")

    parser.add_argument("--seed", type=int, default=123, help="random seed.")

    parser.add_argument('--device', default='cuda', type=str, help="device type for model training",
                        choices=['cuda', 'cpu'])
    parser.add_argument('--device_ids', default='0', type=str,
                        help="Device ids for model training(comma-separated list) when device='cuda', "
                             "`None` for all gpus")

    parser.add_argument("--do_train", type=str2bool, default=False, help="Whether to run training.")

    parser.add_argument("--do_eval", type=str2bool, default=False, help="Whether to run eval on the dev set.")

    # ========================= Model Configs ==========================
    parser.add_argument("--vocab_file", type=str, default=None,
                        help="The vocabulary file that the bert model was trained on.")
    parser.add_argument(
        "--bert_config_file", type=str, default=None,
        help="The config json file corresponding to the pre-trained BERT model. "
             "This specifies the model architecture.")
    parser.add_argument(
        "--init_checkpoint", type=str, default=None,
        help="Initial checkpoint (usually from a pre-trained BERT model).")

    # ========================= Create Pretraining Data Configs ==========================
    parser.add_argument("--stop_words_file", type=str, default=None,
                        help="Stop words which will never be masked.")

    parser.add_argument("--random_queue_size", type=int, default=100000,
                        help="Determines how many documents are queued to shuffle and sample for random_next_sentence."
                             "`-1` for global shuffle")

    parser.add_argument("--input_files", type=str, default='../data/example_sop.txt',
                        help="Input raw text file (or comma-separated list of files).")
    parser.add_argument("--output_tfrecord_dir", type=str, default='../data/tfrecord/',
                        help="Output dir of tfrecord example files.")
    parser.add_argument("--num_output_tfrecord", type=int, default=10,
                        help="Number of tfrecord example files.")

    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length.")
    parser.add_argument("--masked_lm_prob", type=float, default=0.15, help="Masked LM probability.")
    parser.add_argument("--max_predictions_per_seq", type=int, default=20,
                        help="Maximum number of masked LM predictions per sequence.")

    parser.add_argument(
        "--do_lower_case", type=str2bool, default=True,
        help="Whether to lower case the input text. Should be True for uncased "
             "models and False for cased models.")

    parser.add_argument(
        "--dupe_factor", type=int, default=40,
        help="Number of times to duplicate the input data (with different masks).")

    parser.add_argument(
        "--sentence_order_prediction", type=str2bool, default=True,
        help="Whether to use the sentence that's right before the current sentence "
             "as the negative sample for next sentence prediction(SOP).")
    parser.add_argument(
        "--random_next_sentence", type=str2bool, default=False,
        help="Whether to use the sentence that's from other random documents "
             "as the negative sample for next sentence prediction(NSP).")

    parser.add_argument('--do_whole_word_mask', type=str2bool, default=True,
                        help='Whether to use whole word masking rather than per-WordPiece masking.')
    parser.add_argument('--do_whole_word_mask_cn', type=str2bool, default=False,
                        help='Whether to use whole Chinese word masking rather than per-WordPiece masking.')
    parser.add_argument("--ngram", type=int, default=3, help="Maximum number of ngrams to mask.")
    parser.add_argument('--favor_shorter_ngram', type=str2bool, default=True,
                        help='Whether to set higher probabilities for sampling shorter ngrams.')
    parser.add_argument(
        "--short_seq_prob", type=float, default=0.1,
        help="Probability of creating sequences which are shorter than the "
             "maximum length.")

    # ========================= Learning Configs ==========================
    parser.add_argument(
        "--train_batch_size", type=int, default=32,
        help="Batch size for training per gpu.")
    parser.add_argument(
        "--eval_batch_size", type=int, default=128,
        help="Batch size for eval per gpu.")
    parser.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "lamb"],
                        help="The optimizer for training.")
    parser.add_argument("--poly_power", type=float, default=1.0, help="The power of poly decay.")
    parser.add_argument('--learning_rate', type=float, default=3e-5)
    parser.add_argument('--max_grad_norm', default=1.0, type=float, help='Max_grad_norm for grad clip')
    parser.add_argument("--weight_decay", default=0.01, type=float, help="Weight decay if we apply some.")
    parser.add_argument("--adam_epsilon", default=1e-6, type=float, help="Epsilon for Adam optimizer.")
    parser.add_argument('--max_epochs', type=int, default=10, help='How many epochs')
    parser.add_argument('--num_train_steps', default=50000, type=int, help='Number of total steps to run')
    parser.add_argument('--num_warmup_steps', default=5000, type=int, help="Warm ups for parameters not in bert")
    parser.add_argument("--start_warmup_step", type=int, default=0, help="The starting step of warmup.")
    parser.add_argument('--print_steps', type=int, default=100, help="Number of steps to log training metrics.")
    parser.add_argument('--output_dir', type=str, default='../data/tf_model',
                        help='The output directory where the model checkpoints will be written.')
    parser.add_argument("--save_checkpoints_steps", type=int, default=1000,
                        help="How often to save the model checkpoint.")
    parser.add_argument("--keep_checkpoint_max", type=int, default=5,
                        help="How many checkpoints to keep.")
    parser.add_argument("--save_checkpoints_per_epoch", type=str2bool, default=False,
                        help="Whether to save the model checkpoint every epoch.")
    parser.add_argument("--train_tfrecord_dir", type=str, default='../data/train_tfrecord/',
                        help="Dir of train tfrecord example files.")
    parser.add_argument("--eval_tfrecord_dir", type=str, default='../data/eval_tfrecord/',
                        help="Dir of eval tfrecord example files.")
    parser.add_argument("--early_stop", type=str2bool, default=False,
                        help="Whether to early stop when training.")
    parser.add_argument("--max_steps_without_increase", type=int, default=2,
                        help="After max steps without increase, it will early stop.")
    parser.add_argument("--throttle_secs", type=int, default=0,
                        help="The seconds of skipping eval after one eval when `early_stop`=True.")

    # ========================= TPU Configs ==========================
    parser.add_argument("--use_tpu", type=str2bool, default=False, help="Whether to use TPU or GPU/CPU.")

    parser.add_argument(
        "--tpu_name", type=str, default=None,
        help="The Cloud TPU to use for training. This should be either the name "
             "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
             "url.")

    parser.add_argument(
        "--tpu_zone", type=str, default=None,
        help="[Optional] GCE zone where the Cloud TPU is located in. If not "
             "specified, we will attempt to automatically detect the GCE project from "
             "metadata.")

    parser.add_argument(
        "--gcp_project", type=str, default=None,
        help="[Optional] Project name for the Cloud TPU-enabled project. If not "
             "specified, we will attempt to automatically detect the GCE project from "
             "metadata.")

    parser.add_argument("--master", type=str, default=None, help="[Optional] TensorFlow master URL.")

    parser.add_argument(
        "--num_tpu_cores", type=int, default=8,
        help="Only used if `use_tpu` is True. Total number of TPU cores to use.")

    parser.add_argument("--iterations_per_loop", type=int, default=1000,
                        help="How many steps to make in each estimator call.")

    return parser.parse_args()
