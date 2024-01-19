import math
import torch
import torch.nn.functional as F
from unicore import metrics, utils
from unicore.losses import UnicoreLoss, register_loss
import torch.nn as nn
import torch.distributed as dist


def get_loss(logits_decoder, decoder_target, padding_idx):
    decoder_target = decoder_target[:, 1:]
    logits_decoder = logits_decoder[:, :-1]
    decode_tokens = decoder_target.ne(padding_idx)
    decoder_sample_size = decode_tokens.long().sum()

    decoder_loss = F.nll_loss(
        F.log_softmax(
            logits_decoder[decode_tokens], dim=-1, dtype=torch.float32),
        decoder_target[decode_tokens].view(-1),
        ignore_index=padding_idx,
        reduction='mean',
    )

    decoder_pred = torch.argmax(
        logits_decoder[decode_tokens], dim=-1)
    decoder_hit = (decoder_pred == decoder_target[decode_tokens]).long().sum()
    decoder_cnt = decoder_sample_size

    acc_sentence_count = []
    for i in range(decoder_target.shape[0]):
        decoder_cnt_per_sen = decode_tokens[i].long().sum()
        decoder_pred_per_sen = torch.argmax(
            logits_decoder[i][decode_tokens[i]], dim=-1)
        decoder_hit_per_sen = (decoder_pred_per_sen ==
                               decoder_target[i][decode_tokens[i]]).long().sum()
        acc_sentence_count.append(decoder_hit_per_sen == decoder_cnt_per_sen)
    acc_sentence_count = (sum(acc_sentence_count), len(acc_sentence_count))
    return decoder_loss, decoder_hit, decoder_cnt, acc_sentence_count


@register_loss("NAG2GF")
class NAG2GFLoss(UnicoreLoss):
    def __init__(self, task):
        super().__init__(task)
        self.padding_idx = task.dictionary.pad()

    def forward(self, model, sample, reduce=True):
        def inner_forward(input_key='net_input', target_key='target'):
            logits_encoder, logits_decoder, cl_out, vae_kl_loss = model(
                **sample[input_key], features_only=True)

            loss = torch.tensor(0.0)
            logging_output = {
                "sample_size": 1,
                "bsz": sample[input_key]['src_tokens'].size(0),
                "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
            }
            if logits_decoder is not None:
                decoder_target = sample[input_key]['decoder_src_tokens']
                decoder_loss, decoder_hit, decoder_cnt, acc_sentence_count = get_loss(
                    logits_decoder, decoder_target, self.padding_idx)
                loss = decoder_loss * self.args.decoder_loss
                logging_output = {
                    "sample_size": 1,
                    "bsz": sample[input_key]['src_tokens'].size(0),
                    "seq_len": sample[input_key]['src_tokens'].size(1) * sample[input_key]['src_tokens'].size(0),
                    "decoder_loss": decoder_loss.data,
                    "decoder_hit": decoder_hit.data,
                    "decoder_cnt": decoder_cnt.data,
                    "acc_sentence_hit": acc_sentence_count[0],
                    "acc_sentence_cnt": acc_sentence_count[1],
                }

            logging_output['loss'] = loss.data
            return loss, 1, logging_output, cl_out

        loss, sample_size, logging_output, cls_repr = inner_forward()
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs, split='valid') -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        bsz = sum(log.get("bsz", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        seq_len = sum(log.get("seq_len", 0) for log in logging_outputs)
        metrics.log_scalar(
            "loss", loss_sum / sample_size, sample_size, round=5
        )
        metrics.log_scalar(
            "seq_len", seq_len / bsz, 1, round=3
        )
        decoder_loss = sum(log.get('decoder_loss', 0)
                           for log in logging_outputs)
        if decoder_loss > 0:
            metrics.log_scalar('decoder_loss', decoder_loss /
                               sample_size, sample_size, round=5)
            decoder_acc = sum(log.get('decoder_hit', 0) for log in logging_outputs) / \
                sum(log.get('decoder_cnt', 1) for log in logging_outputs)
            if decoder_acc > 0:
                metrics.log_scalar(
                    'decoder_acc', decoder_acc, sample_size, round=5)

            decoder_cnt_t = sum(log.get('decoder_cnt', 1)
                                for log in logging_outputs)
            decoder_ppl = math.exp(min(decoder_loss / decoder_cnt_t, 100))
            if decoder_ppl > 0:
                metrics.log_scalar(
                    'decoder_ppl', decoder_ppl, sample_size, round=5)

            acc_sentence_count = sum(log.get('acc_sentence_hit', 0) for log in logging_outputs)
            acc_sentence_count = acc_sentence_count / \
                sum(log.get('acc_sentence_cnt', 0) for log in logging_outputs)
            metrics.log_scalar('acc_sentence_percentage',
                               acc_sentence_count, sample_size, round=5)

    @staticmethod
    def logging_outputs_can_be_summed(is_train) -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
