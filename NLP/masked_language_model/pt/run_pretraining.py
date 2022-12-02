import os
from functools import partial
import torch
import shutil
from torch.utils.data import DataLoader
from tfrecord.torch.dataset import MultiTFRecordDataset

from model import XBertForPreTraining, build_optimizer
from export import download
from config import parse_args


def validate(model, val_dataloader, vocab_size):
    model.eval()

    masked_lm_total = 0
    masked_lm_correct = 0
    samples_total = 0
    masked_lm_loss = 0.
    next_sentence_loss = 0.
    next_sentence_correct = 0

    with torch.no_grad():
        for batch in val_dataloader:
            output = model(**batch)

            samples_total += output['prediction_logits'].shape[0]

            prediction_scores = output['prediction_logits'].view(-1, vocab_size)
            labels = batch['labels'].view(-1).to(prediction_scores.device)
            select_index = torch.nonzero(labels != -100, as_tuple=False).squeeze()
            masked_lm_prediction = torch.index_select(prediction_scores, dim=0, index=select_index)
            masked_lm_prediction = torch.argmax(masked_lm_prediction, dim=1)
            masked_lm_labels = torch.index_select(labels, dim=0, index=select_index)
            masked_lm_correct += (masked_lm_prediction == masked_lm_labels).float().sum().cpu()
            masked_lm_loss += output['masked_lm_loss'].mean().cpu() * masked_lm_prediction.shape[0]
            masked_lm_total += masked_lm_prediction.shape[0]

            if 'next_sentence_label' in batch:
                next_sentence_loss += output['next_sentence_loss'].mean() * len(batch)
                next_sentence_correct += (batch['next_sentence_label'].view(-1).to(prediction_scores.device) ==
                                          torch.argmax(output['seq_relationship_logits'], dim=1)
                                          ).float().sum().cpu()

    return {'total_loss': masked_lm_loss / masked_lm_total + next_sentence_loss / samples_total,
            'masked_lm_loss': masked_lm_loss / masked_lm_total,
            'next_sentence_loss': next_sentence_loss / samples_total,
            'masked_lm_accuracy': masked_lm_correct / masked_lm_total,
            'next_sentence_accuracy': next_sentence_correct / samples_total,
            }


def create_dataloader(tfrecord_dir, batch_size, args):
    splits = {file.rsplit('.', 1)[0]: 1.0
              for file in os.listdir(tfrecord_dir) if file.endswith('.tfrecord')}
    tfrecord_pattern = os.path.join(tfrecord_dir, '{}.tfrecord')
    index_pattern = os.path.join(tfrecord_dir, '{}.index')
    description = {"input_ids": "int",
                   "attention_mask": "int",
                   "token_type_ids": "int",
                   "labels": "int"}
    if args.sentence_order_prediction or args.random_next_sentence:
        description["next_sentence_label"] = "int"
    dataset = MultiTFRecordDataset(tfrecord_pattern, index_pattern, splits,
                                   description=description,
                                   infinite=False,
                                   batch_size=batch_size)

    if args.num_workers > 0:
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=args.num_workers,
                                   prefetch_factor=args.prefetch)
    else:
        # single-thread reading does not support prefetch_factor arg
        dataloader_class = partial(DataLoader, pin_memory=True, num_workers=0)
    dataloader = dataloader_class(dataset,
                                  batch_size=batch_size,
                                  num_workers=args.num_workers,
                                  prefetch_factor=args.prefetch
                                  )

    return dataloader


def validate_with_early_stop(model, val_dataloader, vocab_size, best_perf, save_type_tuple,
                             early_stop=False, max_steps_without_increase=None):
    print('='*20, 'begin validating', '='*20)
    val_metrics = validate(model, val_dataloader, vocab_size)
    print('\n'.join([f'{k}: {v}' for k, v in val_metrics.items()]))

    if early_stop:
        key_name = 'masked_lm_accuracy'
        _name = 'steps_without_increase'
        if best_perf.get(key_name) is None or val_metrics[key_name] > best_perf[key_name]:
            best_perf[key_name] = val_metrics[key_name]
            best_perf[_name] = 0  # reset `steps_without_increase`
        else:
            if best_perf.get(_name) is None or best_perf[_name] < max_steps_without_increase - 1:
                best_perf[_name] = best_perf.get(_name, 0) + 1
                print(
                    f'No increase in metric "{key_name}" for {save_type_tuple[1]} {save_type_tuple[0]}, skip saving model')
                return True
            else:
                print(f'No increase in metric "{key_name}" which is greater than or equal to max steps '
                      f'({max_steps_without_increase}) configured for early stopping.')
                print(f'Requesting early stopping at {save_type_tuple[0]} {save_type_tuple[1]}')
                return False

    masked_lm_accuracy = val_metrics['masked_lm_accuracy']
    state_dict = model.module.state_dict() if args.device == 'cuda' else model.state_dict()
    model_path = f'{args.output_dir}/bert_{save_type_tuple[0]}_{save_type_tuple[1]}_mlm_acc_{masked_lm_accuracy}.bin'
    torch.save({f'{save_type_tuple[0]}': save_type_tuple[1],
                'model_state_dict': state_dict,
                'masked_lm_accuracy': masked_lm_accuracy},
               model_path)
    best_perf['best_model_path'] = model_path

    return True


def main(args):
    num_devices = 1

    model = XBertForPreTraining.from_pretrained(args.model_name, cache_dir=args.cache_dir)
    vocab_size = model.config.vocab_size
    optimizer, scheduler = build_optimizer(model, args)
    if torch.cuda.is_available() and args.device == 'cuda':
        if args.device_ids is None:
            device_ids = list(range(torch.cuda.device_count()))
        else:
            device_ids = [int(i) for i in args.device_ids.split(',')]
        num_devices = len(device_ids)
        model = torch.nn.parallel.DataParallel(model.to('cuda'), device_ids=device_ids)

    os.makedirs(args.output_dir, exist_ok=True)

    train_dataloader = create_dataloader(args.train_tfrecord_dir,
                                         args.train_batch_size * num_devices,
                                         args)
    val_dataloader = create_dataloader(args.eval_tfrecord_dir,
                                       args.eval_batch_size * num_devices,
                                       args)

    best_perf = {}
    step = 1
    for epoch in range(args.max_epochs):
        model.train()
        for batch in train_dataloader:
            output = model(**batch)
            output['loss'] = output['loss'].mean()
            output['loss'].backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)

            optimizer.step()
            scheduler.step()

            if step % args.print_steps == 0:
                print(f'step: {step}')
                print(' | '.join(['{}: {}'.format(key, output[key].item())
                                  for key in ['loss', 'masked_lm_loss', 'next_sentence_loss']
                                  if output[key] is not None]))

            if not args.save_checkpoints_per_epoch and step % args.save_checkpoints_steps == 0:
                flag = validate_with_early_stop(model, val_dataloader, vocab_size, best_perf,
                                                save_type_tuple=('step', step),
                                                early_stop=args.early_stop,
                                                max_steps_without_increase=args.max_steps_without_increase)
                if not flag:
                    return best_perf

            step += 1

        print('*'*20 + f' epoch: {epoch} end ' + '*'*20)

        if args.save_checkpoints_per_epoch:
            flag = validate_with_early_stop(model, val_dataloader, vocab_size, best_perf,
                                            save_type_tuple=('epoch', epoch),
                                            early_stop=args.early_stop,
                                            max_steps_without_increase=args.max_steps_without_increase)
            if not flag:
                return best_perf

    return best_perf


def merge_best_model(args, best_model_path):
    print(f'best_model_path: {best_model_path}')
    model_path = download(args.model_name, cache_folder=args.cache_dir, ignore_files='pytorch_model.bin')
    if model_path is not None:
        if os.path.exists(model_path+'-pretrained'):
            dest = model_path
        else:
            dest = model_path+'-pretrained'
            os.rename(model_path, dest)
        shutil.copy(best_model_path, os.path.join(dest, 'pytorch_model.bin'))
        print(f'pretrained model export to {dest}')


if __name__ == '__main__':
    args = parse_args()
    perf = main(args)
    merge_best_model(args, perf['best_model_path'])
