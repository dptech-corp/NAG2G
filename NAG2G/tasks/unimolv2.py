# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from unimol import __version__

if __version__ == "2.0.0":
    import logging
    import os
    import numpy as np
    from unicore.data import (
        LMDBDataset,
        RightPadDataset,
        TokenizeDataset,
        RightPadDataset2D,
        NestedDictionaryDataset,
        EpochShuffleDataset,
    )
    from unimol.data import (
        KeyDataset,
        ConformerPCQSampleDataset,
        Unimolv2Features,
    )
    from NAG2G.data import (
        CsvGraphormerDataset,
        SmilesDataset,
        GraphormerDataset,
        ShuffleGraphormerDataset,
        SeqGraphormerDataset,
        RightPadDataset3D,
        ReorderGraphormerDataset,
        RandomSmilesDataset,
        ReorderSmilesDataset,
        ReorderCoordDataset,
        BpeTokenizeDataset,
    )
    from unicore.tasks import UnicoreTask, register_task
    from .transformer_m import G2GMTask

    logger = logging.getLogger(__name__)

    @register_task("G2G_unimolv2")
    class G2GUnimolv2Task(G2GMTask):
        """Task for training transformer auto-encoder models."""

        def load_dataset(self, split, **kwargs):
            """Load a given dataset split.
            Args:
                split (str): name of the data scoure (e.g., bppp)
            """
            split_path = os.path.join(self.args.data, split + ".csv")
            if os.path.exists(split_path):
                raw_dataset = CsvGraphormerDataset(split_path)
            else:
                split_path = os.path.join(self.args.data, split + ".lmdb")
                raw_dataset = LMDBDataset(split_path)

            is_train = "train" in split
            flag_aftsep2 = "aftspe2" in self.args.bpe_tokenizer_path
            sample_dataset = ConformerPCQSampleDataset(
                raw_dataset,
                self.seed,
                "target_coordinates",
                None,
            )
            raw_coord_dataset = KeyDataset(sample_dataset, "coordinates")
            map_coord_dataset = KeyDataset(sample_dataset, "target_map")

            dataset = KeyDataset(raw_dataset, "rxn_smiles")

            dataset = SmilesDataset(dataset)

            reactant_dataset = KeyDataset(dataset, "reactant_smiles")
            product_dataset = KeyDataset(dataset, "product_smiles")

            if is_train and self.args.shufflegraph == "randomsmiles":
                product_dataset = RandomSmilesDataset(product_dataset)
                if self.args.use_reorder:
                    reactant_dataset = ReorderSmilesDataset(product_dataset, reactant_dataset)
                else:
                    reactant_dataset = RandomSmilesDataset(reactant_dataset)


            reactant_smiles_dataset = reactant_dataset
            product_smiles_dataset = product_dataset
            if not self.args.no_reactant:
                reactant_dataset = GraphormerDataset(reactant_dataset)
            product_dataset = GraphormerDataset(product_dataset)

            if (not self.args.no_reactant) and self.args.use_reorder:
                reorder_dataset = ReorderGraphormerDataset(
                    product_dataset,
                    reactant_dataset,
                    align_base="product",
                )
                product_dataset = KeyDataset(reorder_dataset, "product")
                reactant_dataset = KeyDataset(reorder_dataset, "reactant")
            raw_coord_dataset = ReorderCoordDataset(
                raw_coord_dataset, map_coord_dataset, product_dataset
            )

            if self.args.use_class:
                class_dataset = KeyDataset(raw_dataset, "class")
            else:
                class_dataset = None
            if not self.args.no_reactant:
                reactant_dataset = SeqGraphormerDataset(
                    reactant_dataset,
                    class_dataset,
                    min_node=self.args.laplacian_pe_dim,
                    want_attn=self.args.decoder_attn_from_loader,
                    want_charge_h=self.args.want_charge_h,
                    # max_seq_len=self.args.max_seq_len,
                    sumto2=not self.args.not_sumto2,
                    use_sep2=self.args.use_sep2 or flag_aftsep2,
                    want_h_degree=self.args.want_h_degree,
                    idx_type=self.args.idx_type,
                    charge_h_last=self.args.charge_h_last,
                )

                seq_reactant_dataset = KeyDataset(reactant_dataset, "seq")

                seq_reactant_dataset = TokenizeDataset(
                    seq_reactant_dataset,
                    self.dictionary,
                    max_seq_len=self.args.max_seq_len + 1,
                )
                if self.args.bpe_tokenizer_path != "none":
                    seq_reactant_dataset = BpeTokenizeDataset(
                        seq_reactant_dataset,
                        self.args.bpe_tokenizer_path,
                        flag_aftsep2=flag_aftsep2,
                    )
            product_dataset = Unimolv2Features(
                product_dataset,
                raw_coord_dataset,
                None,
                is_train=is_train,
                label_prob=1,
                mid_prob=self.args.mid_prob,
                mid_lower=self.args.mid_lower,
                mid_upper=self.args.mid_upper,
                noise=self.args.noise_scale,
                seed=self.seed + 2,
                N_vnode=self.args.N_vnode,
            )

            net_input = {
                "batched_data": product_dataset,
            }
            if not self.args.no_reactant:
                net_input["decoder_src_tokens"] = RightPadDataset(
                    seq_reactant_dataset,
                    pad_idx=self.dictionary.pad(),
                )

            if self.args.decoder_attn_from_loader:
                reactant_degree_attn_mask_dataset = KeyDataset(
                    reactant_dataset, "degree_attn_mask"
                )
                if self.args.laplacian_pe_dim > 0:
                    reactant_laplacian_attn_mask_dataset = KeyDataset(
                        reactant_dataset, "laplacian_attn_mask"
                    )
                    net_input["decoder_laplacian_attn_mask"] = RightPadDataset3D(
                        reactant_laplacian_attn_mask_dataset,
                        pad_idx=0,
                    )
                net_input["decoder_degree_attn_mask"] = RightPadDataset2D(
                    reactant_degree_attn_mask_dataset,
                    pad_idx=0,
                )

            nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": net_input,
                    "target": {
                        "reactant_smiles": reactant_smiles_dataset,
                        "product_smiles": product_smiles_dataset,
                    },
                },
            )
            if split in ["train", "train.small"]:
                nest_dataset = EpochShuffleDataset(
                    nest_dataset, len(nest_dataset), self.seed
                )
            self.datasets[split] = nest_dataset
