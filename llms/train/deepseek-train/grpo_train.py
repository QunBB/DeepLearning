import torch
from modelscope import AutoTokenizer, AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from utils import get_gsm8k_dataset
from reward import REWARD_FUNCS


def train(args):

    training_args = GRPOConfig(
        output_dir=args.checkpoint_dir,
        learning_rate=args.learning_rate,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
        logging_steps=args.logging_steps,
        bf16=args.bf16,
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        num_generations=args.num_generations,
        max_prompt_length=args.max_prompt_length,
        max_completion_length=args.max_completion_length,
        num_train_epochs=args.epochs,
        save_steps=args.save_steps,
        save_strategy=args.save_strategy,
        max_grad_norm=args.max_grad_norm,
        log_on_each_node=False,
        use_vllm=args.use_vllm,
        vllm_device=args.vllm_device,
        vllm_gpu_memory_utilization=args.vllm_gpu_ratio,
        report_to="none"
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map=None,
        cache_dir=args.cache_dir
    ).to("cuda")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
    tokenizer.pad_token = tokenizer.eos_token

    reward_funcs = [REWARD_FUNCS[func.strip()] for func in args.reward_funcs.split(',')]

    trainer = GRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=get_gsm8k_dataset(cache_dir=args.cache_dir,
                                        first_half=args.split_half == "first_half",
                                        second_half=args.split_half == "second_half"),
    )
    trainer.train()
