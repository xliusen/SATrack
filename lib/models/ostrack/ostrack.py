"""
Basic OSTrack model.
"""
import math
import os
from typing import List
import numpy as np
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
# from torchinfo import summary

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce, vit_tiny_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh

###################################################
from transformers import BertTokenizer, BertModel
from transformers.models.bert.modeling_bert import BertConfig, BertEmbeddings
from lib.models.ostrack import utils as utils
from transformers import logging

logging.set_verbosity_error()

class PositionEmbeddingLearned(nn.Module):
    """
    Absolute pos embedding, learned.
    """
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.embed = nn.Embedding(in_dim, out_dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.uniform_(self.embed.weight)

    def forward(self, x):
        # x: (B, 5, C) or (B, mask, C) or (B, bbox+mask, C)
        n = x.size(1)
        i = torch.arange(n, device=x.device)
        pos = self.embed(i).unsqueeze(0) # (N,C) --> (1,N,C) --> (B,N,C)
        return pos

class OSTrack(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER", token_len=1):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head

        self.aux_loss = aux_loss
        self.head_type = head_type
        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)

        # track query: save the history information of the previous frame 轨迹查询:保存前一帧的历史信息
        self.track_query = None
        self.token_len = token_len

        #################################################################
        # with torch.no_grad():
        #     # BERT
        #     self.bert_model = BertModel.from_pretrained("bert-base-uncased")  # .cuda()
        # Only Use BERT tokenizer
        bert_config = BertConfig(
            vocab_size=30522,
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=12 * 4,
            max_position_embeddings=40,
            hidden_dropout_prob=0.1,
            attention_probs_dropout_prob=0.1,
        )
        # self.language_embeddings = BertEmbeddings(bert_config)
        # self.language_embeddings.apply(utils.init_weights)
        self.tokenizer = BertTokenizer.from_pretrained('/data/data6T/xls/projects/SeamTrack-kd/bert-base-uncased')
        self.descript_embedding = BertEmbeddings(bert_config)
        self.descript_embedding.apply(utils.init_weights)
        self.description_patch_pos_embed = PositionEmbeddingLearned(768, 768)

    ###################################################################
    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                language = None,
                # phrase_ids: torch.Tensor,  # 32*40
                # phrase_attnmask: torch.Tensor,  # 32*40
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                ):
        assert isinstance(search, list), "The type of search is not List"

        out_dict = []
        for i in range(len(search)):

            descript_id = self.tokenizer(language, add_special_tokens=True, truncation=True, pad_to_max_length=True, max_length=40)['input_ids']
            descript_id_tensor = torch.tensor(descript_id)
            language_embeddings = self.descript_embedding(descript_id_tensor.to('cuda'))
            language_embeddings += self.description_patch_pos_embed(language_embeddings)

            x, aux_dict = self.backbone(z=template.copy(), x=search[i],
                                        language_embeddings=language_embeddings,
                                        ce_template_mask=ce_template_mask,
                                        ce_keep_rate=ce_keep_rate,
                                        return_last_attn=return_last_attn, track_query=self.track_query,
                                        token_len=self.token_len)

            feat_last = x
            if isinstance(x, list):
                feat_last = x[-1] # 如果x是一个列表，那么feat_last将被设置为x的最后一个元素

            enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)

            # # self.track_query在前向之后进行更新以保存时间令牌
            # if self.backbone.add_cls_token:
            #     self.track_query = (x[:, :self.token_len].clone()).detach()  # stop grad  (B, N, C)
            #     #从张量x中提取每个序列的前self.token_len个元素，并将这些元素存储在一个新的张量self.track_query中，同时确保这些元素不会参与后续的梯度计算

            att = torch.matmul(enc_opt, x[:, :1].transpose(1, 2))  # (B, HW, N)
            opt = (enc_opt.unsqueeze(-1) * att.unsqueeze(-2)).permute(
                (0, 3, 2, 1)).contiguous()  # (B, HW, C, N) --> (B, N, C, HW)

            # Forward head
            out = self.forward_head(opt, None)

            out.update(aux_dict)
            out['backbone_feat'] = x

            # self.track_query在前向之后进行更新以保存时间令牌
            query = (x[:, :self.token_len].clone()).detach()
            if self.backbone.add_cls_token:
                if self.track_query is None:
                    self.track_query = torch.zeros_like(query)

                max_score, idx = torch.max(out['score_map'].flatten(1), dim=1, keepdim=True)

                # self.track_query = (query.clone()).detach()  # stop grad  (B, N, C)
                # 获取需要更新的位置索引（布尔 mask）
                update_mask = (max_score > 0.7).squeeze(1)  # shape: [B]

                # 将 query 中满足条件的位置复制给 self.track_query
                self.track_query[update_mask] = query[update_mask].clone().detach()
                #从张量x中提取每个序列的前self.token_len个元素，并将这些元素存储在一个新的张量self.track_query中，同时确保这些元素不会参与后续的梯度计算

            out_dict.append(out)

        return out_dict

    def forward_head(self, opt, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        # enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        # opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_ostrack(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    # pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    pretrained_path = os.path.join(current_dir, 'pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('ODTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                        add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                        attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE, )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224':
        backbone = vit_large_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                         add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                         attn_type=cfg.MODEL.BACKBONE.ATTN_TYPE,
                                         )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_tiny_patch16_224_ce':
        backbone = vit_tiny_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                           )

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            add_cls_token=cfg.MODEL.BACKBONE.ADD_CLS_TOKEN,
                                            )

    else:
        raise NotImplementedError
    hidden_dim = backbone.embed_dim
    patch_start_index = 1

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)

    model = OSTrack(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
        token_len=cfg.MODEL.BACKBONE.TOKEN_LEN,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

        print("missing keys:", missing_keys)
        print("unexpected keys:", unexpected_keys)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)


    return model
