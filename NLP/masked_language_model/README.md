- [简介](#简介)
- [特性](#特性)
- [Requirements](#requirements)
- [快速开始](#快速开始)
  - [tensorflow版本](#tensorflow版本)
  - [torch版本](#torch版本)
- [要点](#要点)
  - [中文全词MASK](#中文全词mask)
  - [tensorflow-estimator](#tensorflow-estimator)
  - [torch-tfrecord](#torch-tfrecord)
- [关键参数](#关键参数)


# 简介
自谷歌2018年发表的《BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding》，彻底摒弃了LSTM等循环网络，取得了惊人的成果。到如今，预训练语言模型（Pre-trained Language Models）已经成为自然语言处理领域中非常重要的基础技术了，在许多任务中都取得了state-of-the-art的效果，这都离不开BERT模型，及其衍生的RoBERTa等。

一般情况下，使用BERT进行微调都可以取得不错的效果，但在某些垂直领域，可能由于现有的BERT模型的预训练语料涉及较少，需要使用该领域的文本语料继续进行预训练，才能达到满意的效果。

该仓库提供了tensorflow和torch两种框架的BERT预训练代码实现。
（也可前往单独的预训练代码仓库: [github](https://github.com/QunBB/bert-pretraining)）

PS：这篇文章[(链接)](https://zhuanlan.zhihu.com/p/598095233)讲述了BERT模型以其变种系列。

# 特性
- 针对中文句子，支持以词为粒度，而非字的全词掩码（Whole Word Masking），以及n-gram掩码（MASK）。
- 支持停用词不参与MASK
- 支持单机多卡的预训练
- 使用tfrecord作为存储介质，支持低内存资源下的超大语料使用

# Requirements
- **Python 3.x**

- **tensorflow**
```
jieba
tensorflow-gpu==1.15.5
# tensorflow==1.15.5
```
或者

- **torch**

```
crc32c
jieba
transformers
torch==1.9
huggingface-hub==0.4.0

```

# 快速开始

## tensorflow版本

- **拷贝仓库代码**

```sh
git clone https://github.com/QunBB/DeepLearning.git

cd DeepLearning/NLP/masked_language_model/tf

pip install -r requirements.txt
```

- **构建训练集**

```sh
python create_pretraining_data.py \
--input_files=../data/example_sop.txt \
--output_tfrecord_dir=../data/train_tfrecord/ \
--do_whole_word_mask_cn=true \
--ngram=4 \
--vocab_file=../data/bert/vocab.txt
```

- **构建验证集**

```sh
python create_pretraining_data.py \
--input_files=../data/example_sop.txt \
--output_tfrecord_dir=../data/eval_tfrecord/ \
--do_whole_word_mask_cn=true \
--ngram=4 \
--vocab_file=../data/bert/vocab.txt
```

- **训练&验证**

```sh
python run_pretraining.py \
--vocab_file=../data/bert/vocab.txt \
--init_checkpoint=../data/bert/bert_model.ckpt \
--do_train=1 \
--bert_config_file=../data/bert/bert_config.json \
--save_checkpoints_steps=1000 \
--keep_checkpoint_max=10 \
--early_stop=true \
--train_tfrecord_dir=../data/train_tfrecord/ \
--eval_tfrecord_dir=../data/eval_tfrecord/
```

run_pretraining.py: 如果设置early_stop为false，则训练过程中不会有eval阶段。

请在训练后再进行验证脚本：

（默认会加载最后的一个模型。如果要重新指定，通过--init_checkpoint进行设置，并将--output_dir设置为新的目录）

```sh
python run_pretraining.py \
--vocab_file=../data/bert/vocab.txt \
--init_checkpoint=../data/bert/bert_model.ckpt \
--do_eval=1 \
--bert_config_file=../data/bert/bert_config.json \
--eval_tfrecord_dir=../data/eval_tfrecord/
```

或者

run_pretraining_sess.py: 每次保存模型都会执行eval

```sh
python run_pretraining_sess.py \
--vocab_file=../data/bert/vocab.txt \
--init_checkpoint=../data/bert/bert_model.ckpt \
--do_train=1 \
--bert_config_file=../data/bert/bert_config.json \
--save_checkpoints_per_epoch=true \
--keep_checkpoint_max=10 \
--early_stop=true \
--train_tfrecord_dir=../data/train_tfrecord/ \
--eval_tfrecord_dir=../data/eval_tfrecord/
```

## torch版本

- **拷贝仓库代码**

```sh
git clone https://github.com/QunBB/DeepLearning.git

cd DeepLearning/NLP/masked_language_model/pt

pip install -r requirements.txt
```

- **构建训练集**

```sh
python create_pretraining_data.py \
--input_files=../data/example_sop.txt \
--output_tfrecord_dir=../data/pt_train_tfrecord/ \
--do_whole_word_mask_cn=true \
--ngram=4 \
--model_name=bert-base-chinese \
--cache_dir=../cache/
```

- **构建验证集**

```sh
python create_pretraining_data.py \
--input_files=../data/example_sop.txt \
--output_tfrecord_dir=../data/pt_eval_tfrecord/ \
--do_whole_word_mask_cn=true \
--ngram=4 \
--model_name=bert-base-chinese \
--cache_dir=../cache/
```

- **训练&验证**

```sh
python run_pretraining.py \
--save_checkpoints_steps=1000 \
--early_stop=true \
--train_tfrecord_dir=../data/pt_train_tfrecord/ \
--eval_tfrecord_dir=../data/pt_eval_tfrecord/
```

训练过程中，所有保存的模型会保存在<output_dir>。

程序会自动选择最好的一个模型，导出至<cache_dir>/<model_name>-pretrained，并从huggingface下载好完整的模型配置文件，如下：

```
<cache_dir>/<model_name>-pretrained
  pytorch_model.bin
  config.json
  tokenizer.json
  tokenizer_config.json
  vocab.txt
  README.md
```

- **微调**

见[run_finetuning.py](https://github.com/QunBB/DeepLearning/blob/main/NLP/masked_language_model/pt/run_finetuning.py)

# 要点

## 中文全词MASK

出自[Pre-Training with Whole Word Masking for Chinese BERT](https://github.com/ymcui/Chinese-BERT-wwm)，但其不提供预训练代码。

| 说明                   | 样例                                                         |
| :--------------------- | :----------------------------------------------------------- |
| 原始文本               | 使用语言模型来预测下一个词的probability。                    |
| 分词文本               | 使用 语言 模型 来 预测 下 一个 词 的 probability 。          |
| 原始Mask输入           | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 pro [MASK] ##lity 。 |
| 全词Mask输入           | 使 用 语 言 [MASK] 型 来 [MASK] 测 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |
| (中文分词)全词Mask输入 | 使 用 语 言 [MASK] [MASK] 来 [MASK] [MASK] 下 一 个 词 的 [MASK] [MASK] [MASK] 。 |

## tensorflow-estimator

使用[MirroredStrategy](https://www.tensorflow.org/guide/distributed_training#mirroredstrategy)来实现多GPU训练

## torch-tfrecord

使用了[tfrecord](https://github.com/vahidk/tfrecord)作为存储介质，解决了内存资源问题，并且优化了`num_workers > 0`场景下的问题，包括：

1. MultiTFRecordDataset数据重复问题，由于每个worker都会有一个不同的dataset对象副本，因此每个worker会重复读取iterator的数据。其实TFRecordDataset已经解决了这个问题，本人按照同样的思路，在MultiTFRecordDataset中实现；
2. 所有worker会平分所有数据，因此每个worker都会出现低于batch_size的情况，`drop_last=True`会过滤`num_workers`次。通过增加`batch_size`参数，优化数据分配，实现了有且仅有一个worker会出现低于batch_size的情况；
3. 这些实现依赖index文件

# 关键参数
|   参数名称   |  说明    |  类型  |  默认值  |
| ---- | ---- | ---- | ---- |
|   do_whole_word_mask   | 谷歌官方的bert对WordPiece的全词MASK，一般针对英文单词和数字 | 布尔型  | True |
|   do_whole_word_mask_cn   |   中文文本分词之后，以词为粒度的中文MASK，而非字   |   布尔型   |   False   |
| stop_words_file | 停用词文件，停用词不参与MASK | 字符串 | None |
| random_queue_size | 随机队列容量，该队列用于局部打散和random_next_sentence的负样本，取-1则相当于全局打散 | 整型 | 100000 |
| sentence_order_prediction | 使用SOP任务，当前句子前面的句子（即打乱句子顺序）作为负样本 | 布尔型 | True |
| random_next_sentence | 使用NSP任务，从其他文档随机抽取的句子作为负样本 | 布尔型 | False |
| ngram | 词或字掩码（MASK）最大的n-gram | 整型 | 3 |
| favor_shorter_ngram | 更短的n-gram，更高的概率会被掩码。如2-gram的MASK概率会比3-gram高 | 布尔型 | True |
| short_seq_prob | 构建序列长度小于max_seq_length的概率 | 浮点型 | 0.1 |
| max_seq_length | 文本序列的最大长度 | 整型 | 512 |
| masked_lm_prob | 每个token被MASK的概率 | 浮点型 | 0.15 |
| max_predictions_per_seq | 序列中MASK的最大可能数量 | 整型 | 20 |
| dupe_factor | 样本重复构建的次数，相同句子可以有不同的MASK，类似于动态MASK | 整型 | 40 |
| num_train_steps | 训练的总步数 | 整型 | 50000 |
| max_epochs | 训练的最大epoch数量 | 整型 | 10 |
| early_stop | 训练早停，依据为验证集的指标: masked_lm_accuracy | 布尔型 | False |
| max_steps_without_increase | 早停时，验证集masked_lm_accuracy不提升的最大次数，超过该次数，则早停退出 | 整型 | 2 |
| save_checkpoints_steps | 训练多少步，保存一次模型并执行一次eval | 整型 | 1000 |
| save_checkpoints_per_epoch | 每一个epoch保存一次模型。设置True时，save_checkpoints_steps不生效。<br />tensor flow版本的run_pretraining.py该参数无效 | 布尔型 | False |
| print_steps | 训练多少步，打印一次日志 | 整型 | 100 |
| device | 训练时，使用CPU还是GPU | 字符串 | cuda |
| device_ids | 使用GPU的编号，逗号分隔符 | 字符串 | 0 |
| input_files | 预训练语料文件，逗号分隔符。<br />一行代表一个句子，多个连续句子构成段落，段落之间使用空行。 | 字符串 | ../data/example_sop.txt |
| output_dir | 模型的保存目录 | 字符串 |  |
| output_tfrecord_dir | 构建预训练样本tfrecord的保存目录 | 字符串 |  |
| num_output_tfrecord | 数据集样本的tfrecord文件输出数量。当语料文件较大时，可扩大该参数，防止一个tfrecord文件过大。 | 整型 | 10 |
| train_tfrecord_dir | 训练集的tfrecord目录 | 字符串 |  |
| eval_tfrecord_dir | 验证集的tfrecord目录 | 字符串 |  |
| model_name | 使用的模型名称，该参数仅针对torch版本。https://huggingface.co/models | 字符串 | bert-base-chinese |
| cache_dir | 模型下载的缓存目录，该参数仅针对torch版本 | 字符串 | ../cache/ |
其他参数见:

[config.py - tensorflow](https://github.com/QunBB/DeepLearning/blob/main/NLP/masked_language_model/tf/config.py)

[config.py - torch](https://github.com/QunBB/DeepLearning/blob/main/NLP/masked_language_model/pt/config.py)