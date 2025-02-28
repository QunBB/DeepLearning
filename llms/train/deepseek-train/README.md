# 环境搭建

```bash
conda create -n vllm python=3.12

conda activate vllm
pip install vllm -U
pip install trl -U

# modelscope
pip install addict modelscope
```

# SFT+GRPO训练

```bash
# sft
python main.py --task=sft_train --model_name_or_path=Qwen/Qwen2.5-0.5B-Instruct --bf16 --checkpoint_dir=outputs/Qwen-0.5B-SFT-FirstHalf --per_device_train_batch_size=8 --save_strategy=epoch --epochs=1

# grpo
python main.py --task=grpo_train --model_name_or_path=outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-117 --bf16 --use_vllm --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf --per_device_train_batch_size=8 --save_strategy=epoch 
```

# 推理
```bash
python main.py --task=inference --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-934
```

```text
请输入你的问题：
Natalia sold clips to 22 of her friends in April, and then she sold half as many clips in May. How many clips did Natalia sell altogether in April and May?

Assistant:
<think>
In April, Natalia sold clips to 22 friends.
In May, she sold half as many clips as in April, which is 22/2 = <<22/2=11>>11 clips.
  Altogether, Natalia sold 22+11 = <<22+11=33>>33 clips in April and May.
</think>
<answer>
33
</answer>
```
