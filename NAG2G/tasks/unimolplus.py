# Copyright (c) DP Techonology, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from unimol import __version__

if __version__ == "2.5.0":
    import logging
    import os
    import numpy as np
    from unicore.data import (
        LMDBDataset,
        RightPadDataset,
        # TokenizeDataset,
        RightPadDataset2D,
        NestedDictionaryDataset,
        EpochShuffleDataset,
    )
    from unimol.data import (
        KeyDataset,
        ConformerSampleDataset,
        NormalizeDataset,
        UnimolPlusFeatureDataset,
    )
    from NAG2G.data import (
        CsvGraphormerDataset,
        SmilesDataset,
        SeqGraphormerDataset,
        RightPadDataset3D,
        ReorderGraphormerDataset,
        RandomSmilesDataset,
        ReorderSmilesDataset,
        ReorderCoordDataset,
        BpeTokenizeDataset,
        TokenizeDataset,
        CoordDataset,
        AddMapDataset,
        MoleculeFeatureDataset,
        IndexAtomDataset,
    )
    from unicore.tasks import UnicoreTask, register_task
    from .transformer_m import G2GMTask

    logger = logging.getLogger(__name__)

    @register_task("G2G_unimolplus")
    class G2GUnimolv2Task(G2GMTask):
        """Task for training transformer auto-encoder models."""

        def load_dataset(self, split, **kwargs):
            """Load a given dataset split.
            Args:
                split (str): name of the data scoure (e.g., bppp)
            """
            # 0:smiles 1:label
            flag_label_wanted = True
            # flag_label_wanted = (True, False)
            split_path = os.path.join(self.args.data, split + ".csv")
            if os.path.exists(split_path):
                raw_dataset = CsvGraphormerDataset(split_path)
                file_type = "csv"
            else:
                split_path = os.path.join(self.args.data, split + ".lmdb")
                raw_dataset = LMDBDataset(split_path)
                file_type = "lmdb"

            self.load_dataset_detail(
                raw_dataset=raw_dataset,
                split=split,
                flag_label_wanted=flag_label_wanted,
                file_type=file_type,
                **kwargs
            )

        def load_dataset_detail(
            self, raw_dataset, split, flag_label_wanted=True, **kwargs
        ):
            if self.args.use_stereoisomerism:
                raise
            if isinstance(flag_label_wanted, tuple):
                 flag_label_smiles_wanted, flag_label_wanted = flag_label_wanted
            elif flag_label_wanted is True:
                flag_label_smiles_wanted = True
            elif flag_label_wanted is False:
                flag_label_smiles_wanted = False
            is_train = "train" in split
            flag_aftsep2 = "aftspe2" in self.args.bpe_tokenizer_path
            if self.args.task_type == "retrosynthesis":
                input_name, output_name = "product_smiles", "reactant_smiles"
                coordinates_name = "target_coordinates"
                map_name = "target_map"
            elif self.args.task_type == "synthesis":
                raise
                input_name, output_name = "reactant_smiles", "product_smiles"
            else:
                print(self.args.task_type)
                raise
            dataset = KeyDataset(raw_dataset, "rxn_smiles")
            dataset = SmilesDataset(dataset)
            if flag_label_wanted or flag_label_smiles_wanted:
                reactant_dataset = KeyDataset(dataset, output_name)
                reactant_smiles_dataset = reactant_dataset
            product_dataset = KeyDataset(dataset, input_name)
            product_dataset = AddMapDataset(product_dataset)

            file_type = kwargs["file_type"] if "file_type" in kwargs.keys() else "lmdb"
            if file_type == "lmdb":
                map_coord_dataset = KeyDataset(raw_dataset, map_name)

                raw_coord_dataset = ConformerSampleDataset(
                    raw_dataset, self.seed, "target_atoms", coordinates_name
                )
                raw_coord_dataset = NormalizeDataset(
                    raw_coord_dataset, "coordinates", normalize_coord=True
                )
                raw_coord_dataset = KeyDataset(raw_coord_dataset, "coordinates")
            elif file_type == "csv":
                coor_dataset = CoordDataset(product_dataset)
                raw_coord_dataset = KeyDataset(coor_dataset, "coordinates")
                map_coord_dataset = KeyDataset(coor_dataset, map_name)
            else:
                raise

            if is_train and self.args.shufflegraph == "randomsmiles":
                product_dataset = RandomSmilesDataset(
                    product_dataset, prob=self.args.random_smiles_prob
                )
                if not self.args.use_reorder:
                    reactant_dataset = RandomSmilesDataset(
                        reactant_dataset, prob=self.args.random_smiles_prob
                    )

            product_smiles_dataset = product_dataset
            product_dataset = MoleculeFeatureDataset(
                dataset=product_dataset,
                drop_feat_prob=0.0,
                seed=self.seed,
                smi_key=None,
            )
            origin_token_dataset = IndexAtomDataset(
                product_smiles_dataset,
            )

            if flag_label_wanted:
                reactant_smiles_dataset = reactant_dataset
                if self.args.use_reorder:
                    reactant_dataset = ReorderSmilesDataset(
                        product_smiles_dataset, reactant_dataset
                    )
                    mol_reactant_dataset = KeyDataset(reactant_dataset, "mol")
                    reactant_dataset = KeyDataset(reactant_dataset, "smiles")
                    reactant_smiles_dataset = reactant_dataset
                    if self.args.use_stereoisomerism:
                        # reactant_dataset = GraphormerDataset(reactant_dataset)
                        reactant_dataset = MoleculeFeatureDataset(
                            dataset=mol_reactant_dataset,
                            drop_feat_prob=0.0,
                            seed=self.seed,
                            smi_key=None,
                        )
                    else:
                        # reactant_dataset = GraphormerDataset(reactant_dataset)
                        reactant_dataset = MoleculeFeatureDataset(
                            dataset=reactant_dataset,
                            drop_feat_prob=0.0,
                            seed=self.seed,
                            smi_key=None,
                        )
                        reorder_dataset = ReorderGraphormerDataset(
                            product_dataset,
                            reactant_dataset,
                            align_base="product",
                            task_type=self.args.task_type,
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
            if flag_label_wanted:
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
                    multi_gap=self.args.multi_gap,
                    use_stereoisomerism=self.args.use_stereoisomerism,
                    want_re=self.args.want_re,
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

            product_dataset = UnimolPlusFeatureDataset(
                smi_dataset=product_smiles_dataset,
                token_dataset=origin_token_dataset,
                mask_token_prob=0.0,
                pad_idx=self.dictionary.pad(),
                mask_idx=127,
                src_pos_dataset=raw_coord_dataset,
                mask_pos_prob=0.0,
                noise=0.0,
                noise_type="uniform",
                molecule_dataset=product_dataset,
                seed=self.seed,
            )

            net_input = {
                "batched_data": product_dataset,
            }
            if flag_label_wanted:
                net_input["decoder_src_tokens"] = RightPadDataset(
                    seq_reactant_dataset,
                    pad_idx=self.dictionary.pad(),
                )
            if self.args.decoder_attn_from_loader:
                assert flag_label_wanted is True
                reactant_degree_attn_mask_dataset = KeyDataset(
                    reactant_dataset, "degree_attn_mask"
                )
                net_input["decoder_degree_attn_mask"] = RightPadDataset2D(
                    reactant_degree_attn_mask_dataset,
                    pad_idx=0,
                )
                if self.args.laplacian_pe_dim > 0:
                    reactant_laplacian_attn_mask_dataset = KeyDataset(
                        reactant_dataset, "laplacian_attn_mask"
                    )
                    net_input["decoder_laplacian_attn_mask"] = RightPadDataset3D(
                        reactant_laplacian_attn_mask_dataset,
                        pad_idx=0,
                    )

                if self.args.want_bond_attn:
                    reactant_bond_attn_mask_dataset = KeyDataset(
                        reactant_dataset, "bond_attn_mask"
                    )
                    net_input["decoder_bond_attn_mask"] = RightPadDataset2D(
                        reactant_bond_attn_mask_dataset,
                        pad_idx=0,
                    )
            target_dict = {
                "product_smiles": product_smiles_dataset,
            }
            if flag_label_wanted or flag_label_smiles_wanted:
                target_dict["reactant_smiles"] = reactant_smiles_dataset

            nest_dataset = NestedDictionaryDataset(
                {
                    "net_input": net_input,
                    "target": target_dict,
                },
            )
            if split in ["train", "train.small"]:
                nest_dataset = EpochShuffleDataset(
                    nest_dataset, len(nest_dataset), self.seed
                )
            self.datasets[split] = nest_dataset
