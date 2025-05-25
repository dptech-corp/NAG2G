import math
from typing import Dict, List, Optional
import sys

import torch
import torch.nn as nn
from torch import Tensor

# from . import move_to_cuda, strip_pad

import logging
import math
import sys
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# from dpaie import search, utils
# from fairseq.data import data_utils
# from fairseq.models import FairseqIncrementalDecoder
# from dpaie.ngram_repeat_block import NGramRepeatBlock

logger = logging.getLogger(__name__)

"""
Translate algorithms, whitch supprot a batch input.
    algorithms:
     - greedy search (when args.beam_size <= 0)
     - beam search (when args.beam_size > 0. Support to adjust 
                    these parameters: beam_size and length_penalty)
    
    inputs:
     - src_tokens: (batch_size, src_len)
    outputs:
     - gen_seqs: (batch_size, max_seq_len/tgt_len.max()) (related to the stop rules)
    
"""

"""
Referenced from facebookresearch/XLM,
 at https://github.com/facebookresearch/XLM/blob/master/xlm/model/transformer.py
"""


class BeamHypotheses(object):
    def __init__(self, n_hyp, length_penalty):
        """
        Initialize n-best list of hypotheses.
        """
        self.length_penalty = length_penalty
        self.n_hyp = n_hyp
        self.hyp = []
        self.worst_score = 1e9

    def __len__(self):
        """
        Number of hypotheses in the list.
        """
        return len(self.hyp)

    def add(self, hyp, sum_logprobs):
        """
        Add a new hypothesis to the list.
        """

        lp = len(hyp) ** self.length_penalty  # deafult length penalty
        # lp = (5 + len(hyp)) ** self.length_penalty / (5 + 1) ** self.length_penalty # Google GNMT's length penalty
        score = sum_logprobs / lp
        # score = sum_logprobs
        if len(self) < self.n_hyp or score > self.worst_score:
            self.hyp.append((score, hyp))
            if len(self) > self.n_hyp:
                sorted_scores = sorted(
                    [(s, idx) for idx, (s, _) in enumerate(self.hyp)]
                )
                # delete the worst hyp in beam
                del self.hyp[sorted_scores[0][1]]
                # update worst score with the sencond worst hyp
                self.worst_score = sorted_scores[1][0]
            else:
                self.worst_score = min(score, self.worst_score)

    def is_done(self, best_sum_logprobs, cur_len):
        """
        If there are enough hypotheses and that none of the hypotheses being generated
        can become better than the worst one in the heap, then we are done with this sentence.
        """

        if len(self) < self.n_hyp:
            return False
        else:
            # return self.worst_score >= best_sum_logprobs
            # / cur_len ** self.length_penalty
            return (
                self.worst_score >= best_sum_logprobs / cur_len**self.length_penalty
            )


class SimpleGenerator(nn.Module):
    def __init__(
        self,
        model,
        dict,
        output_num_return_sequences_per_batch=5,
        max_seq_len=512,
        beam_size=5,
        temperature=1.0,
        match_source_len=False,
        len_penalty=1.0,
        args=None,
        eos=None,
    ):
        super().__init__()
        self.model = model
        self.dict = dict
        self.output_num_return_sequences_per_batch = beam_size
        self.max_seq_len = max_seq_len
        self.beam_size = beam_size
        self.temperature = temperature
        self.match_source_len = match_source_len
        self.pad_idx = self.dict.pad()
        self.bos_idx = self.dict.bos()
        self.eos_idx = eos if eos is not None else self.dict.eos()
        # self.eos_idx2 = 247
        self.len_penalty = len_penalty
        self.args = args

    @torch.no_grad()
    def _generate(self, sample):
        self.model.eval()
        src_tokens = self.model.get_src_tokens(sample)
        batch_size = src_tokens.size(0)

        decoder_kwargs = {}
        encoder_kwargs = {}
        for k, v in sample["net_input"].items():
            if "decoder" in k:
                decoder_kwargs[k] = v
            else:
                encoder_kwargs[k] = v

        if self.args.use_class_encoder:
            assert self.args.N_vnode == 2
            encoder_kwargs["cls_embedding"] = self.model.decoder_embed_tokens(decoder_kwargs["decoder_src_tokens"][:, 1])
            
        encoder_result = self.model.forward_encoder(**encoder_kwargs)
        encoder_out = encoder_result["encoder_rep"]
        padding_mask = encoder_result["padding_mask"]
        masked_tokens = encoder_result["masked_tokens"]

        src_padding_mask = padding_mask
        if src_padding_mask is None:
            src_padding_mask = torch.zeros(
                [encoder_out.shape[0], encoder_out.shape[1]]
            ).to(encoder_out.device)

        # expand to beam size the source latent representations
        encoder_out = encoder_out.repeat_interleave(
            self.beam_size, dim=0
        )  # (B x beam_size) x T x C
        src_padding_mask = src_padding_mask.repeat_interleave(
            self.beam_size, dim=0
        )  # (B x beam_size) x T x C
        # generated sentences (batch with beam current hypotheses)
        generated = src_tokens.new(batch_size * self.beam_size, self.max_seq_len).fill_(
            self.pad_idx
        )  # upcoming output
        generated[:, 0].fill_(self.bos_idx)
        # generated hypotheses
        generated_hyps = [
            BeamHypotheses(self.beam_size * 2, self.len_penalty)
            for _ in range(batch_size)
        ]  # 为每个输入句子定义维护其beam search序列的类实例

        # scores for each sentence in the beam
        beam_scores = encoder_out.new(batch_size, self.beam_size).fill_(
            0
        )  # 定义scores向量，保存累加的log_probs
        beam_scores[:, 1:] = -1e9  # 需要初始化为-inf
        # current position
        cur_len = 1
        # done sentences
        done = [False] * batch_size  # 标记每个输入句子的beam search是否完成

        while cur_len < self.max_seq_len:
            tgt_padding_mask = generated[:, :cur_len].eq(self.pad_idx)
            scores, avg_attn_scores = self.model.forward_decoder(
                decoder_src_tokens=generated[:, :cur_len],
                encoder_cls=encoder_out,
                temperature=self.temperature,
                encoder_padding_mask=src_padding_mask,
            )  # (batch_size * beam_size, n_tgt_words)
            n_tgt_words = scores.size(-1)
            # - scores: (batch_size * beam_size, n_tgt_words)

            # if self.args.N_left > 0:  # and self.eos_idx2 in generated[:, :cur_len]:
            #     score_tmp, _ = torch.topk(
            #         scores, self.args.N_left, dim=-1, largest=True, sorted=True
            #     )
            #     score_tmp = score_tmp[:, -1].unsqueeze(1)
            #     scores[scores < score_tmp] = -100

            _scores = scores + beam_scores.view(batch_size * self.beam_size, 1)
            # (batch_size, beam_size * vocab_size)
            _scores = _scores.view(batch_size, self.beam_size * n_tgt_words)
            next_scores, next_words = torch.topk(
                _scores, 2 * self.beam_size, dim=-1, largest=True, sorted=True
            )
            # - next_scores, next_words: (batch_size, 2 * beam_size)
            # next batch beam content
            # list of (batch_size * beam_size) tuple(next hypothesis score, next word, current position in the batch)
            next_batch_beam = []
            # for each sentence
            for sent_id in range(batch_size):
                # if we are done with this sentence
                done[sent_id] = done[sent_id] or generated_hyps[sent_id].is_done(
                    next_scores[sent_id].max().item(), cur_len
                )
                if done[sent_id]:
                    next_batch_beam.extend(
                        [(0, self.pad_idx, 0)] * self.beam_size
                    )  # pad the batch
                    continue
                # next sentence beam content
                next_sent_beam = []
                # next words for this sentence
                for idx, value in zip(next_words[sent_id], next_scores[sent_id]):
                    # get beam and word IDs

                    beam_id = idx // n_tgt_words
                    word_id = idx % n_tgt_words

                    # end of sentence, or next word
                    effective_beam_id = sent_id * self.beam_size + beam_id
                    if word_id == self.eos_idx or cur_len + 1 == self.max_seq_len:
                        generated_hyps[sent_id].add(
                            generated[effective_beam_id, :cur_len].clone(), value.item()
                        )
                    else:
                        next_sent_beam.append((value, word_id, effective_beam_id))
                    # the beam for next step is full
                    if len(next_sent_beam) == self.beam_size:
                        break
                # update next beam content
                if len(next_sent_beam) == 0:
                    next_sent_beam = [
                        (0, self.pad_idx, 0)
                    ] * self.beam_size  # pad the batch
                next_batch_beam.extend(next_sent_beam)
            # prepare next batch
            beam_scores = beam_scores.new([x[0] for x in next_batch_beam])
            beam_words = generated.new([x[1] for x in next_batch_beam])
            beam_idx = src_tokens.new([x[2] for x in next_batch_beam])
            # re-order batch and internal states
            generated = generated[beam_idx, :]
            generated[:, cur_len] = beam_words
            # update current length
            cur_len = cur_len + 1
            if all(done):
                break
        # select the best hypotheses
        tgt_len = src_tokens.new(
            batch_size * self.output_num_return_sequences_per_batch
        )
        best = []
        score = []
        for i, hypotheses in enumerate(generated_hyps):
            sorted_hyps = sorted(hypotheses.hyp, key=lambda x: x[0])

            for j in range(self.output_num_return_sequences_per_batch):
                effective_batch_idx = self.output_num_return_sequences_per_batch * i + j
                hyps_pop = sorted_hyps.pop()
                best_score = hyps_pop[0]
                best_hyp = hyps_pop[1]
                tgt_len[effective_batch_idx] = len(best_hyp)
                best.append(best_hyp)
                score.append(best_score)

        # generate target batch
        gen_seqs = src_tokens.new(
            batch_size * self.output_num_return_sequences_per_batch,
            tgt_len.max().item(),
        ).fill_(self.pad_idx)
        for i, hypo in enumerate(best):
            gen_seqs[i, : tgt_len[i]] = hypo
        tgt_lengths = torch.sum(gen_seqs.ne(self.pad_idx), dim=1)
        return gen_seqs, tgt_lengths, score
