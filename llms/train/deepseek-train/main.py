import argparse

from grpo_train import train as grpo_train
from sft_train import train as sft_train
from inference import infer


def main():
    parser = argparse.ArgumentParser(description="Simple example of a training script.")

    parser.add_argument(
        "--task",
        type=str,
        default="train",
        required=True,
        choices=["grpo_train", "sft_train", "inference"],
        help="Which task to run.",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written when training."
             "And the output directory where the model will be loaded from when at inference",
    )
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        required=False,
        help="Path to pretrained model or model identifier from modelscope or huggingface.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        required=False,
        default=None,
        help="The cache folder of model from modelscope or huggingface.",
    )
    parser.add_argument(
        "--split_half",
        type=str,
        required=False,
        choices=["first_half", "second_half"],
        default=None,
        help="Whether to split dataset in half and which half to choose, such as use first half to sft-train and second half to grpo-train",
    )
    parser.add_argument(
        "--reward_funcs",
        type=str,
        required=False,
        default="xmlcount_reward_func,soft_format_reward_func,strict_format_reward_func,int_reward_func,correctness_reward_func",
        help="Which reward functions to choose, separating with commas.",
    )
    parser.add_argument(
        "--max_prompt_length",
        type=int,
        default=256,
        required=False,
        help="Maximum length of the prompt. If the prompt is longer than this value, it will be truncated left.",
    )
    parser.add_argument(
        "--max_completion_length",
        type=int,
        default=256,
        required=False,
        help="Maximum length of the generated completion.",
    )
    parser.add_argument(
        "--max_seq_length",
        type=int,
        default=512,
        required=False,
        help="Maximum length of the tokenized sequence. Sequences longer than `max_seq_length` are truncated from the right.",
    )
    parser.add_argument(
        "--logging_steps",
        type=int,
        default=10,
        required=False,
        help="Number of update steps between two logs.",
    )
    parser.add_argument(
        "--save_strategy",
        type=str,
        required=False,
        default="steps",
        choices=["no", "steps", "epoch", "best"],
        help="The checkpoint save strategy to adopt during training.",
    )
    parser.add_argument(
        "--save_steps",
        type=int,
        default=100,
        required=False,
        help="Number of updates steps before two checkpoint saves if `save_strategy=\"steps\"`.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        required=False,
        help="Total number of training epochs to perform.",
    )
    parser.add_argument(
        "--per_device_train_batch_size",
        type=int,
        default=2,
        required=False,
        help="The batch size per GPU/XPU/TPU/MPS/NPU core/CPU for training.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        required=False,
        help="Number of updates steps to accumulate the gradients for, before performing a backward/update pass.",
    )
    parser.add_argument(
        "--num_generations",
        type=int,
        default=8,
        required=False,
        help="Number of generations per prompt to sample. The global batch size (num_processes * per_device_batch_size) must be divisible by this value.",
    )
    parser.add_argument(
        "--adam_beta1",
        type=float,
        default=0.9,
        required=False,
        help="The beta1 hyperparameter for the [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--adam_beta2",
        type=float,
        default=0.99,
        required=False,
        help="The beta2 hyperparameter for the [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.1,
        required=False,
        help="The weight decay to apply (if not zero) to all layers except all bias and LayerNorm weights in the optimizer.",
    )
    parser.add_argument(
        "--max_grad_norm",
        type=float,
        default=0.1,
        required=False,
        help="Maximum gradient norm (for gradient clipping).",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        required=False,
        help="The initial learning rate for [`AdamW`] optimizer.",
    )
    parser.add_argument(
        "--lr_scheduler_type",
        type=str,
        default="cosine",
        required=False,
        help="The scheduler type to use.",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        required=False,
        help="Ratio of total training steps used for a linear warmup from 0 to `learning_rate`.",
    )
    parser.add_argument(
        "--bf16",
        action="store_true",
        help="Whether to use bf16 16-bit (mixed) precision training instead of 32-bit training. Requires Ampere or higher NVIDIA architecture or using CPU (use_cpu) or Ascend NPU.",
    )
    parser.add_argument(
        "--use_vllm",
        action="store_true",
        help="Whether to use vLLM for generating completions.",
    )
    parser.add_argument(
        "--vllm_device",
        type=str,
        default="cuda:0",
        help="Device where vLLM generation will run, e.g. `\"cuda:1\"`. If set to `\"auto\"` (default), the system will"
             "automatically select the next available GPU after the last one used for training. This assumes that"
             "training has not already occupied all available GPUs. If only one device is available, the device will be"
             "shared between both training and vLLM.",
    )
    parser.add_argument(
        "--vllm_gpu_ratio",
        type=float,
        default=0.2,
        help="Ratio (between 0 and 1) of GPU memory to reserve for the model weights, activations, and KV cache on the"
             "device dedicated to generation powered by vLLM. Higher values will increase the KV cache size and thus"
             "improve the model's throughput. However, if the value is too high, it may cause out-of-memory (OOM) errors"
             "during initialization.",
    )

    args = parser.parse_args()

    if args.task == "grpo_train":
        grpo_train(args)
    elif args.task == "sft_train":
        sft_train(args)
    elif args.task == "inference":
        infer(args)


if __name__ == '__main__':
    main()
