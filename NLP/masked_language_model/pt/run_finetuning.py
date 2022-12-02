import os
from collections import OrderedDict
import torch
from transformers.models.bert.modeling_bert import BertModel, BertConfig


def main(model_path):
    model = BertModel(config=BertConfig.from_pretrained(os.path.join(model_path, 'config.json')))
    checkpoint = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')

    # 筛选出`BertModel`部分的权重，并提出权重名称的前缀`bert`
    bert_state_dict = OrderedDict()
    for key in checkpoint['model_state_dict']:
        if key.startswith('bert.'):
            bert_state_dict[key[5:]] = checkpoint['model_state_dict'][key]

    model.load_state_dict(bert_state_dict)
    print(model)


class YourModel(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()

        self.bert = BertModel(config=BertConfig.from_pretrained(os.path.join(model_path, 'config.json')))
        checkpoint = torch.load(os.path.join(model_path, 'pytorch_model.bin'), map_location='cpu')
        self.bert.load_state_dict(checkpoint)
        print(self.bert)


if __name__ == '__main__':
    main('../cache/bert-base-chinese-pretrained')
    # or
    model = YourModel('../cache/bert-base-chinese-pretrained')
