import torch

try:
    from unimol.models import UniMolModel

    class CustomizedUniMolModel(UniMolModel):
        def forward(
            self,
            src_tokens,
            src_distance,
            src_coord,
            src_edge_type,
            encoder_masked_tokens=None,
            features_only=False,
            classification_head_name=None,
            **kwargs
        ):

            if classification_head_name is not None:
                features_only = True

            padding_mask = src_tokens.eq(self.padding_idx)
            if not padding_mask.any():
                padding_mask = None
            x = self.embed_tokens(src_tokens)

            def get_dist_features(dist, et):
                n_node = dist.size(-1)
                gbf_feature = self.gbf(dist, et)
                gbf_result = self.gbf_proj(gbf_feature)
                graph_attn_bias = gbf_result
                graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
                graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
                return graph_attn_bias

            graph_attn_bias = get_dist_features(src_distance, src_edge_type)
            (
                encoder_rep,
                encoder_pair_rep,
                delta_encoder_pair_rep,
                x_norm,
                delta_encoder_pair_rep_norm,
            ) = self.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias)
            encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0

            encoder_distance = None
            encoder_coord = None

            logits = encoder_rep

            if not features_only:
                if self.args.masked_token_loss > 0:
                    logits = self.lm_head(encoder_rep, encoder_masked_tokens)
                if self.args.masked_coord_loss > 0:
                    coords_emb = src_coord
                    if padding_mask is not None:
                        atom_num = (
                            torch.sum(1 - padding_mask.type_as(x), dim=1) - 1
                        ).view(-1, 1, 1, 1)
                    else:
                        atom_num = src_coord.shape[1] - 1
                    delta_pos = coords_emb.unsqueeze(1) - coords_emb.unsqueeze(2)
                    attn_probs = self.pair2coord_proj(delta_encoder_pair_rep)
                    coord_update = delta_pos / atom_num * attn_probs
                    coord_update = torch.sum(coord_update, dim=2)
                    encoder_coord = coords_emb + coord_update
                if self.args.masked_dist_loss > 0:
                    encoder_distance = self.dist_head(encoder_pair_rep)

            if classification_head_name is not None:
                logits = self.classification_heads[classification_head_name](
                    encoder_rep
                )

            return (
                logits,
                encoder_distance,
                encoder_coord,
                x_norm,
                delta_encoder_pair_rep_norm,
            )

except:
    print("Cannot import unimol")
