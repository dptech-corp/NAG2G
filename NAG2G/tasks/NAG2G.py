import logging
import os
import time

import contextlib
from typing import Optional

import numpy as np
from unicore.data import (
    Dictionary,
    MaskTokensDataset,
    NestedDictionaryDataset,
    NumelDataset,
    NumSamplesDataset,
    LMDBDataset,
    AppendTokenDataset,
    PrependTokenDataset,
    RightPadDataset,
    EpochShuffleDataset,
    RawLabelDataset,
    TokenizeDataset,
    data_utils,
    RightPadDataset2D,
    SortDataset,
    FromNumpyDataset,
)
from NAG2G.data import (
    CustomizedUnicoreDataset,
    MaskPointsDataset,
    MaskPointsPocketDataset,
    DistanceDataset,
    EdgeTypeDataset,
    CrossDistanceDataset,
    RandomDataset,
    RandomLabelDataset,
    KeyDataset,
    SizeDataset,
    ReorderDataset,
    RightPadDatasetCoord,
    RightPadDatasetCross2D,
    ListShuffleDataset,
    RandomSmilesDataset,
    BartTokenDataset,
)

from unicore.tasks import register_task, UnicoreTask
import selfies as sf
import rdkit.Chem as Chem

logger = logging.getLogger(__name__)


@register_task("NAG2GF")
class NAG2GFTask(UnicoreTask):
    """Task for training transformer auto-encoder models."""

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        NAG2GFTask.add_other_args(parser)
        NAG2GFTask.add_default_encoder_args(parser)

    @staticmethod
    def add_other_args(parser):
        parser.add_argument(
            "data",
            help="colon separated path to data directories list, \
                            will be iterated upon during epochs in round-robin manner",
        )

        parser.add_argument(
            "--dict-name",
            default="tokenizer-smiles-bart/dict.txt",
            help="dict name",
        )

        parser.add_argument(
            "--bart-dict-path",
            default="tokenizer-smiles-bart",
            help="bart dict path",
        )

        parser.add_argument(
            "--use-old-dataset", action="store_true", help="use-old-dataset"
        )

        parser.add_argument(
            "--task-type",
            default="retrosynthetic",
            choices=["retrosynthetic", "synthesis"],
            help="noise type in coordinate noise",
        )

        parser.add_argument(
            "--use-selfies",
            default=0,
            type=int,
            help="coordinate noise for masked atoms",
        )
        parser.add_argument(
            "--use-class-embedding",
            default=0,
            type=int,
            help="class embedding or not",
        )
        parser.add_argument(
            "--results-smi-path",
            metavar="RESDIR",
            type=str,
            default=None,
            help='path to save eval smile results (optional)"',
        )
        parser.add_argument(
            "--results-smi-file",
            metavar="RESDIR",
            type=str,
            default=None,
            help='file to save eval smile results (optional)"',
        )
        parser.add_argument(
            "--training-shuffle", action="store_true", help="disable progress bar"
        )

    @staticmethod
    def add_default_encoder_args(parser):
        parser.add_argument(
            "--mask-prob",
            default=0.25,
            type=float,
            help="probability of replacing a token with mask",
        )
        parser.add_argument(
            "--leave-unmasked-prob",
            default=0.05,
            type=float,
            help="probability that a masked token is unmasked",
        )
        parser.add_argument(
            "--random-token-prob",
            default=0.05,
            type=float,
            help="probability of replacing a token with a random token",
        )
        parser.add_argument(
            "--noise-type",
            default="normal",
            choices=["trunc_normal", "uniform", "normal", "none"],
            help="noise type in coordinate noise",
        )
        parser.add_argument(
            "--noise",
            default=1,
            type=float,
            help="coordinate noise for masked atoms",
        )

    def __init__(self, args, dictionary):
        super().__init__(args)
        self.dictionary = dictionary
        self.seed = args.seed
        # add mask token
        self.use_selfies = args.use_selfies
        self.use_class_embedding = args.use_class_embedding
        self.mask_idx = dictionary.add_symbol("[MASK]", is_special=True)
        self.task_type = args.task_type

        # self.src_sizes = np.array([])
        # self.tgt_sizes = np.array([])
        # self.indices = np.array([])

    @classmethod
    def setup_task(cls, args, **kwargs):
        dictionary = Dictionary.load(os.path.join(args.data, args.dict_name))
        logger.info("dictionary: {} types".format(len(dictionary)))
        return cls(args, dictionary)

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def get_size(self, sizes):
        tmp_size_data = np.array([0] * len(sizes))
        for i in range(len(sizes)):
            tmp_size_data[i] = sizes[i]
        return tmp_size_data

    def ordered_indices(self, dataset_length):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        indices = np.arange(dataset_length, dtype=np.int64)
        if self.tgt_sizes is not None:
            indices = indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
            indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
        else:
            indices = np.arange(dataset_length, dtype=np.int64)
        return indices

    def one_dataset_old(self, raw_dataset, coord_seed, mask_seed, **kwargs):
        # src_sizes = KeyDataset(raw_dataset, 'tar_size')
        # tgt_sizes = KeyDataset(raw_dataset, 'rec_size')

        # self.src_sizes = self.get_size(src_sizes)
        # self.tgt_sizes = self.get_size(tgt_sizes)
        # self.indices = self.ordered_indices(len(raw_dataset))
        flag_label_wanted = (
            kwargs["flag_label_wanted"]
            if "flag_label_wanted" in kwargs.keys()
            else True
        )
        if self.use_selfies == 0:
            if self.task_type == "retrosynthetic":
                input_name, output_name = "smiles_target", "smiles_train"
            elif self.task_type == "synthetic":
                input_name, output_name = "smiles_train", "smiles_target"

            # dataset = ReorderDataset(raw_dataset, coord_seed, 'smi', 'atoms','coordinates')

        else:
            if self.task_type == "retrosynthetic":
                input_name, output_name = "selfies_target", "selfies_train"
            elif self.task_type == "synthetic":
                input_name, output_name = "selfies_train", "selfies_target"
            # dataset = ReorderDataset(raw_dataset, coord_seed, 'selfies', 'atoms','coordinates')
        token_dataset = KeyDataset(raw_dataset, input_name)
        if flag_label_wanted:
            token_target_dataset = KeyDataset(raw_dataset, output_name)

        # if self.use_class_embedding:
        #     class_dataset = KeyDataset(raw_dataset, 'class')
        # else:
        #     class_dataset = KeyDataset(raw_dataset, 'class')

        token_dataset = TokenizeDataset(
            token_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
        )
        if flag_label_wanted:
            token_target_dataset = TokenizeDataset(
                token_target_dataset, self.dictionary, max_seq_len=self.args.max_seq_len
            )

        # src_dataset, tgt_dataset = MaskTokensDataset.apply_mask(
        #     token_dataset,
        #     self.dictionary,
        #     pad_idx=self.dictionary.pad(),
        #     mask_idx=self.mask_idx,
        #     seed=mask_seed,
        #     mask_prob=self.args.mask_prob,
        #     leave_unmasked_prob=self.args.leave_unmasked_prob,
        #     random_token_prob=self.args.random_token_prob,
        # )

        def PrependAndAppend(dataset, pre_token, app_token):
            dataset = PrependTokenDataset(dataset, pre_token)
            return AppendTokenDataset(dataset, app_token)

        src_dataset = token_dataset
        # tgt_dataset = token_dataset

        src_dataset = PrependAndAppend(
            src_dataset, self.dictionary.bos(), self.dictionary.eos()
        )
        # tgt_dataset = PrependAndAppend(
        #     tgt_dataset, self.dictionary.pad(), self.dictionary.pad())

        if flag_label_wanted:
            decoder_src_dataset = PrependAndAppend(
                token_target_dataset, self.dictionary.bos(), self.dictionary.eos()
            )
        # decoder_target = PrependAndAppend(
        #     token_target_dataset, self.dictionary.bos(), self.dictionary.eos())
        if flag_label_wanted:
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "decoder_src_tokens": RightPadDataset(
                    decoder_src_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                # "reaction_type": RawLabelDataset(class_dataset),
            }, {
                # 'encoder': RightPadDataset(
                #     tgt_dataset,
                #     pad_idx=self.dictionary.pad()
                #     ),
                # 'decoder':RightPadDataset(
                #     decoder_target,
                #     pad_idx=self.dictionary.pad()
                #     )
            }
        else:
            return {
                "src_tokens": RightPadDataset(
                    src_dataset,
                    pad_idx=self.dictionary.pad(),
                )
            }, {}

    def one_dataset(self, raw_dataset, coord_seed, mask_seed, **kwargs):
        flag_label_wanted = (
            kwargs["flag_label_wanted"]
            if "flag_label_wanted" in kwargs.keys()
            else True
        )
        if self.use_selfies == 0:
            if self.task_type == "retrosynthetic":
                input_name, output_name = "smiles_target_list", "smiles_reactant_list"
            elif self.task_type == "synthetic":
                input_name, output_name = "smiles_reactant_list", "smiles_target_list"
        else:
            raise
            if self.task_type == "retrosynthetic":
                input_name, output_name = "selfies_target_list", "selfies_reactant_list"
            elif self.task_type == "synthetic":
                input_name, output_name = "selfies_reactant_list", "selfies_target_list"

        token_dataset = KeyDataset(raw_dataset, input_name)
        if flag_label_wanted:
            token_target_dataset = KeyDataset(raw_dataset, output_name)

        def data_aug_dataset(dataset):
            bart_dict_path = os.path.join(self.args.data, self.args.bart_dict_path)
            if kwargs["split"] in ["valid", "test"]:
                prob = 0
            else:
                prob = 0.9
            dataset = ListShuffleDataset(dataset, prob=prob)
            dataset = RandomSmilesDataset(dataset, prob=prob)
            dataset = BartTokenDataset(
                dataset, bart_dict_path, max_seq_len=self.args.max_seq_len
            )
            return dataset

        token_dataset = data_aug_dataset(token_dataset)
        if flag_label_wanted:
            token_target_dataset = data_aug_dataset(token_target_dataset)

            return {
                "src_tokens": RightPadDataset(
                    token_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
                "decoder_src_tokens": RightPadDataset(
                    token_target_dataset,
                    pad_idx=self.dictionary.pad(),
                ),
            }, {}
        else:
            return {
                "src_tokens": RightPadDataset(
                    token_dataset,
                    pad_idx=self.dictionary.pad(),
                )
            }, {}

    def load_dataset(self, split, combine=False, **kwargs):
        """Load a given dataset split.
        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        split_path = os.path.join(self.args.data, split + ".lmdb")
        # todo: load data from example data
        raw_dataset = LMDBDataset(split_path)
        if not self.args.use_old_dataset:
            net_input, target = self.one_dataset(
                raw_dataset, self.args.seed, self.args.seed, split=split
            )
        else:
            net_input, target = self.one_dataset_old(
                raw_dataset, self.args.seed, self.args.seed
            )
        dataset = {"net_input": net_input, "target": target}

        dataset = NestedDictionaryDataset(dataset)
        if split in ["train", "train.small"]:
            # self.args.training_shuffle and
            dataset = EpochShuffleDataset(dataset, len(dataset), self.args.seed)

        self.datasets[split] = dataset

    def test_step(self, args, sample, model, loss, step, seed):
        tgt_tokens = sample["net_input"]["decoder_src_tokens"]
        pred, vae_kl_loss = model._generate(sample)
        tgt_tokens = tgt_tokens.cpu().numpy()
        beam_size = model.beam_size
        for i in range(len(tgt_tokens)):
            a = tgt_tokens[i]
            mol_str = []
            for t in a[1:]:
                tt = self.dictionary.vec_index(t)
                if tt == "[SEP]":
                    break
                mol_str.append(tt)
            mol_str = "".join(mol_str)

            if self.use_selfies == 1:
                mol_str = sf.decoder(mol_str)

            file_path = args.results_smi_path
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            file_check = os.path.join(file_path, args.results_smi_file)

            with open(file_check, "a") as w:
                w.write(mol_str + "\t" + "target" + "\n")
                for j in range(beam_size):
                    mol_smi = []
                    mol_str2 = []
 
                    a = pred[j + i * beam_size].cpu().numpy()
                    for t in a[1:]:
                        tt = self.dictionary.vec_index(t)
                        if tt == "[PAD]":
                            break
                        mol_str2.append(tt)
                    mol_str2 = "".join(mol_str2)

                    if self.use_selfies == 1:
                        smiles = sf.decoder(mol_str2)
                        # 加一个rdkit转分子对象能否成功的判断
                        # if Chem.MolFromSmiles(smiles) is not None:
                        w.write(smiles + "\t" + "predicted" + str(j) + "\n")
                        mol_smi.append(smiles)
                    else:
                        # 加一个rdkit转分子对象能否成功的判断
                        smiles = mol_str2
                        # if Chem.MolFromSmiles(smiles) is not None:
                        w.write(smiles + "\t" + "predicted" + str(j) + "\n")
                        mol_smi.append(smiles)

        return pred, {
            "vae_kl_loss": vae_kl_loss,
            "sample_size": 1,
            "bsz": sample["net_input"]["decoder_src_tokens"].size(0),
            "seq_len": sample["net_input"]["decoder_src_tokens"].size(1)
            * sample["net_input"]["decoder_src_tokens"].size(0),
        }

    def infer_step(self, sample, model, **kwargs):
        pred, _ = model._generate(sample)
        beam_size = model.beam_size
        mol_smi = []
        for j in range(beam_size):
            mol_str2 = []
            a = pred[j].cpu().numpy()
            for t in a[1:]:
                tt = self.dictionary.vec_index(t)
                if tt == "[PAD]":
                    break
                mol_str2.append(tt)
            mol_str2 = "".join(mol_str2)
            mol_smi.append(mol_str2)
        return mol_smi


@register_task("NAG2GF_RS")
class NAG2GFTask_RS(NAG2GFTask):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.task_type = "retrosynthetic"


@register_task("NAG2GF_S")
class NAG2GFTask_S(NAG2GFTask):
    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)
        self.task_type = "synthesis"
