# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional, List
import math
from unicore.optim.lr_scheduler import UnicoreLRScheduler, register_lr_scheduler


@register_lr_scheduler("tensor2tensor_decay")
class Tensor2tensorDecayLRSchedule(UnicoreLRScheduler):
    """Decay the LR on the tensor2tensor schedule."""

    def __init__(self, args, optimizer, total_train_steps):
        super().__init__(args, optimizer, total_train_steps)
        if self.args.warmup_ratio > 0:
            # if warmup_ratio > 0, use external train steps
            assert total_train_steps is not None
            self.warmup_updates = int(
                self.args.warmup_ratio * total_train_steps)
            self.total_num_update = total_train_steps
        else:
            assert args.total_num_update > 0
            self.warmup_updates = args.warmup_updates
            self.total_num_update = args.total_num_update
        self.lr = args.lr[0]
        if self.warmup_updates > 0:
            self.warmup_factor = 1.0 / self.warmup_updates
        else:
            self.warmup_factor = 1
        self.total_num_update = total_train_steps
        self.end_learning_rate = args.end_learning_rate
        self.power = args.power
        self.optimizer.set_lr(self.warmup_factor * self.lr)

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        parser.add_argument('--force-anneal', '--fa', type=int, metavar='N',
                            help='force annealing at specified epoch')
        parser.add_argument('--warmup-updates', default=0, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-ratio', default=-1.0, type=float, metavar='N',
                            help='warmup the learning rate linearly for the first N-percent updates')
        parser.add_argument('--start-learning-rate', default=2.0, type=float)
        parser.add_argument('--end-learning-rate', default=0.0, type=float)
        parser.add_argument('--power', default=1.0, type=float)
        parser.add_argument('--total-num-update', default=1000000, type=int)

    def get_next_lr(self, epoch):
        lrs = self.args.lr
        if self.args.force_anneal is None or epoch < self.args.force_anneal:
            # use fixed LR schedule
            next_lr = lrs[min(epoch, len(lrs) - 1)]
        else:
            # annneal based on lr_shrink
            next_lr = self.optimizer.get_lr()
        return next_lr

    def step_begin_epoch(self, epoch):
        """Update the learning rate at the beginning of the given epoch."""
        self.lr = self.get_next_lr(epoch)
        self.optimizer.set_lr(self.warmup_factor * self.lr)
        return self.optimizer.get_lr()

    def step_update(self, num_updates):
        """Update the learning rate after each update."""

        if num_updates == 0:
            t_num_updates = math.pow(1, -0.5)
        else:
            t_num_updates = math.pow(num_updates, -0.5)
        if t_num_updates < num_updates * math.pow(self.warmup_updates, -1.5):
            lr = self.args.start_learning_rate * \
                (math.pow(self.args.encoder_embed_dim, -0.5) * t_num_updates)
        else:
            lr = self.args.start_learning_rate * \
                (math.pow(self.args.encoder_embed_dim, -0.5) *
                 num_updates * math.pow(self.warmup_updates, -1.5))
        self.optimizer.set_lr(lr)
        return self.optimizer.get_lr()
