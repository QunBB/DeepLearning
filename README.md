[<img src="https://api.gitsponsors.com/api/badge/img?id=419746952" height="40">](https://api.gitsponsors.com/api/badge/link?p=1KgsCPEwLkeM9OhPLyONthSPwzpU86MFRdz5Zm2VJ72LipcdlWg6A0do/CZbX3Lck3ujRRVf0xsx6f3jPNJ1iMVLyFPzlJ2fnA+7PknPRKaFl2OA4OoRZqMMcJpKbg6LGwJcEgqhqcNnJzhW3d7d2A==)

# 1. tensorflow使用

项目文件夹：[_tensorflow](https://github.com/QunBB/DeepLearning/tree/main/_tensorflow)

- **batch_normalization**: [专栏](https://zhuanlan.zhihu.com/p/360842139)
- **dataset.shuffle、batch、repeat:** [专栏](https://zhuanlan.zhihu.com/p/360843167)
- **tfrecord读写数据:** [专栏](https://zhuanlan.zhihu.com/p/363959153)
- **estimator模型训练**: [专栏](https://zhuanlan.zhihu.com/p/367057708)
- **tf版本hashtable**：[专栏](https://zhuanlan.zhihu.com/p/386915385)

## 1.1 部署

项目文件夹：[_tensorflow/serving](https://github.com/QunBB/DeepLearning/tree/main/_tensorflow/serving)

- **tensorflow serving：** [专栏](https://zhuanlan.zhihu.com/p/407986666)

## 1.2 自定义算子

项目文件夹：[_tensorflow/tensorflow-custom-op](https://github.com/QunBB/DeepLearning/tree/main/_tensorflow/tensorflow-custom-op)（[独立实现git仓库](https://github.com/QunBB/tensorflow-custom-op)）

- **如何实现TensorFlow自定义算子？：** [专栏](https://zhuanlan.zhihu.com/p/672088843)

# 2. 多任务学习MTL

项目文件夹：[multitasklearning](https://github.com/QunBB/DeepLearning/tree/main/multitasklearning)

- **shared_bottom、mmoe、ple模型介绍:** [专栏](https://zhuanlan.zhihu.com/p/425209494)
- **多目标优化-Uncertainty Weight、GradNorm、Dynamic Weight Average、Pareto-Eficient**：[专栏](https://zhuanlan.zhihu.com/p/456089764)
- **STEM: 推荐模型中的维度坍塌&兴趣纠缠**：[专栏](https://zhuanlan.zhihu.com/p/19885938029)

# 3. 推荐系统

项目文件夹：[recommendation](https://github.com/QunBB/DeepLearning/tree/main/recommendation)
(本git的推荐系统使用的是tensorflow1.x，关于tensorflow2.x的实现可前往另外一个[git](https://github.com/QunBB/RecSys))

- **ctr训练提速(超大batch size)-CowClip**：[专栏](https://zhuanlan.zhihu.com/p/557451365)
- **基于二叉树的近似最近邻搜索-Annoy**: [专栏](https://zhuanlan.zhihu.com/p/714579473)

## 3.1 Match(召回)

项目文件夹：[recommendation/match](https://github.com/QunBB/DeepLearning/tree/main/recommendation/match)

- **多兴趣召回MIND**: [专栏](https://zhuanlan.zhihu.com/p/463064543)
- **多兴趣召回ComiRec**: [专栏](https://zhuanlan.zhihu.com/p/568781562)

- **深入浅出地理解Youtube DNN推荐模型**: [专栏](https://zhuanlan.zhihu.com/p/405907646)
- **引入对偶增强向量的双塔召回模型**: [专栏](https://zhuanlan.zhihu.com/p/608636233)

## 3.2 Rank(排序)

项目文件夹：[recommendation/rank](https://github.com/QunBB/DeepLearning/tree/main/recommendation/rank)

- **ctr特征重要性建模：FiBiNet&FiBiNet++模型**：[专栏](https://zhuanlan.zhihu.com/p/603262632)
- **ctr预估之FMs系列:FM/FFM/FwFM/FEFM**：[专栏](https://zhuanlan.zhihu.com/p/613030015)
- **ctr预估之DNN系列模型:FNN/PNN/DeepCrossing**：[专栏](https://zhuanlan.zhihu.com/p/623567076)
- **ctr预估之Wide&Deep系列模型:DeepFM/DCN**：[专栏](https://zhuanlan.zhihu.com/p/631668163)
- **ctr预估之Wide&Deep系列(下):NFM/xDeepFM**：[专栏](https://zhuanlan.zhihu.com/p/634584585)
- **CTR特征建模：ContextNet & MaskNet(Twitter在用的排序模型)**：[专栏](https://zhuanlan.zhihu.com/p/660375034)
- **CTR之行为序列建模用户兴趣：DIN**：[专栏](https://zhuanlan.zhihu.com/p/679852484)
- **CTR之行为序列建模用户兴趣：DIEN**：[专栏](https://zhuanlan.zhihu.com/p/685855305)
- **CTR之Session行为序列建模用户兴趣：DSIN**：[专栏](https://zhuanlan.zhihu.com/p/688338754)
- **CTR之行为序列建模用户兴趣：Temporal Interest Network**：[专栏](https://zhuanlan.zhihu.com/p/7832498217)
- **推荐模型中辅助排序损失的作用**：[专栏](https://zhuanlan.zhihu.com/p/10542978888)
- **GwPFM&HMoE: 推荐模型中的维度坍塌&兴趣纠缠**：[专栏](https://zhuanlan.zhihu.com/p/19885938029)

## 3.3 多场景建模(Multi-Domain)

项目文件夹：[recommendation/multidomain](https://github.com/QunBB/DeepLearning/tree/main/recommendation/multidomain)

- **多场景建模: STAR(Star Topology Adaptive Recommender)**：[专栏](https://zhuanlan.zhihu.com/p/717054800)
- **多场景建模（二）: SAR-Net（Scenario-Aware Ranking Network）**：[专栏](https://zhuanlan.zhihu.com/p/718704281)
- **多场景多任务建模（三）: M2M（Multi-Scenario Multi-Task Meta Learning）**：[专栏](https://zhuanlan.zhihu.com/p/939534954)
- **多场景多任务建模（四）: PEPNet（Parameter and Embedding Personalized Network）**：[专栏](https://zhuanlan.zhihu.com/p/4552106145)

# 4. TensorRT & Triton
TensorRT：一种深度学习框架，提升GPU模型推理性能

Triton：TensorRT对应的模型服务化，实现模型统一管理和部署

## 4.1 Triton

项目文件夹：[triton](https://github.com/QunBB/DeepLearning/tree/main/triton)

- **TensorRT&Triton学习笔记(一)**: triton和模型部署+client：[专栏](https://zhuanlan.zhihu.com/p/482170985)

# 5. 自然语言处理NLP

项目文件夹：[nlp](https://github.com/QunBB/DeepLearning/tree/main/nlp)

## 5.1 bert句向量

- **Sentence-BERT**: [专栏](https://zhuanlan.zhihu.com/p/504983847)

## 5.2 bert系列

**BERT预训练代码(tensorflow和torch版本)：[NLP/masked_language_model](https://github.com/QunBB/DeepLearning/tree/main/nlp/masked_language_model)**（[独立实现git仓库](https://github.com/QunBB/bert-pretraining)）

- **BERT模型系列大全解读**：[专栏](https://zhuanlan.zhihu.com/p/598095233)

# 6. 深度学习trick

项目文件夹：[trick](https://github.com/QunBB/DeepLearning/tree/main/trick)

带pt后缀的为pytorch实现版本，不带后缀的则为tensorflow版本。

- **变量初始化(initialization)、分层学习率(hierarchical_lr)、梯度累积(gradient_accumulation)**: [专栏](https://zhuanlan.zhihu.com/p/553277132)
- **Stochastic Weight Averaging (SWA)、Exponential Moving Average(EMA)**：[专栏](https://zhuanlan.zhihu.com/p/554955968)
- **(unbalance)分类模型-类别不均衡问题之loss设计 & Label Smoothing**：[专栏](https://zhuanlan.zhihu.com/p/582312784)

# 7. 多模态

项目文件夹：[multimodal](https://github.com/QunBB/DeepLearning/tree/main/multimodal)

## 7.1 Stable Diffusion

项目文件夹：[Stable Diffusion](https://github.com/QunBB/DeepLearning/tree/main/multimodal/stable_diffusion)

- **AI绘画Stable Diffusion原理之VQGANs/隐空间/Autoencoder**：[专栏](https://zhuanlan.zhihu.com/p/645939505)
- **AI绘画Stable Diffusion原理之扩散模型DDPM**：[专栏](https://zhuanlan.zhihu.com/p/645939505)

# 8. Embedding

项目文件夹：[embedding](https://github.com/QunBB/DeepLearning/tree/main/embedding)

- **Embedding压缩之hash embedding**：[专栏](https://zhuanlan.zhihu.com/p/669320977)
- **Embedding压缩之基于二进制码的Hash Embedding**：[专栏](https://zhuanlan.zhihu.com/p/670802301)

# 9. 大语言模型（LLMs）

项目文件夹：[llms](https://github.com/QunBB/DeepLearning/tree/main/llms)

## 9.1 LangChain

LangChain开发入门教程：：[llms/langchain_tutorial](https://github.com/QunBB/DeepLearning/tree/main/llms/langchain_tutorial)

- **Model I/O（prompts、llms、chat model、output parsers）**：[专栏](https://zhuanlan.zhihu.com/p/700167692)
- **RAG/Retrieval（文档加载器、文本分割器、Embedding、向量数据库、检索）**：[专栏](https://zhuanlan.zhihu.com/p/706889931)
- **Tools/Agents（工具、function call、agent）**：[专栏](https://zhuanlan.zhihu.com/p/712459598)
