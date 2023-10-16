import torch
from typing import Optional
from dataclasses import dataclass
from torch.nn import CrossEntropyLoss
from transformers.models.bert.modeling_bert import BertForPreTraining, BertForPreTrainingOutput
from transformers import AdamW, get_polynomial_decay_schedule_with_warmup


@dataclass
class XBertForPreTrainingOutput(BertForPreTrainingOutput):
    masked_lm_loss: Optional[torch.FloatTensor] = None
    next_sentence_loss: Optional[torch.FloatTensor] = None


class XBertForPreTraining(BertForPreTraining):
    def __init__(self, config):
        super().__init__(config)

    def forward(
            self,
            input_ids=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            next_sentence_label=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
    ):
        r"""
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the masked language modeling loss. Indices should be in `[-100, 0, ...,
                config.vocab_size]` (see `input_ids` docstring) Tokens with indices set to `-100` are ignored (masked),
                the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`
            next_sentence_label (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
                Labels for computing the next sequence prediction (classification) loss. Input should be a sequence
                pair (see `input_ids` docstring) Indices should be in `[0, 1]`:

                - 0 indicates sequence B is a continuation of sequence A,
                - 1 indicates sequence B is a random sequence.
            kwargs (`Dict[str, any]`, optional, defaults to *{}*):
                Used to hide legacy arguments that have been deprecated.

        Returns:

        Example:

        ```python
        >>> from transformers import BertTokenizer, BertForPreTraining
        >>> import torch

        >>> tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        >>> model = BertForPreTraining.from_pretrained('bert-base-uncased')

        >>> inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")
        >>> outputs = model(**inputs)

        >>> prediction_logits = outputs.prediction_logits
        >>> seq_relationship_logits = outputs.seq_relationship_logits
        ```
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output, pooled_output = outputs[:2]
        prediction_scores, seq_relationship_score = self.cls(sequence_output, pooled_output)
        seq_relationship_score.view(-1, 2)

        total_loss = None
        masked_lm_loss = None
        next_sentence_loss = None

        # masked_lm labels is necessary when training
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(prediction_scores.view(-1, self.config.vocab_size), labels.view(-1))
            total_loss = masked_lm_loss

        if next_sentence_label is not None:
            next_sentence_loss = loss_fct(seq_relationship_score.view(-1, 2), next_sentence_label.view(-1))
            total_loss += next_sentence_loss

        if not return_dict:
            output = (prediction_scores, seq_relationship_score) + outputs[2:]
            return ((total_loss,) + output) if total_loss is not None else output

        return XBertForPreTrainingOutput(
            loss=total_loss,
            prediction_logits=prediction_scores,
            seq_relationship_logits=seq_relationship_score,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            masked_lm_loss=masked_lm_loss,
            next_sentence_loss=next_sentence_loss
        )


def build_optimizer(model, args):

    no_decay = ["bias", "LayerNorm.weight"]
    model_parameters = model.named_parameters()
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model_parameters if not any(nd in n for nd in no_decay)],
         'weight_decay': args.weight_decay},
        {'params': [p for n, p in model_parameters if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=args.learning_rate, eps=args.adam_epsilon)
    scheduler = get_polynomial_decay_schedule_with_warmup(optimizer,
                                                          args.num_warmup_steps,
                                                          args.num_train_steps)
    return optimizer, scheduler
