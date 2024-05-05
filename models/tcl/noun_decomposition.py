import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import MODELS
from models.tcl.encoders import CLIPImageFeatureEncoder
from models.tcl.mi import InfoNCE, ExtendedInfoNCE
from models.tcl.pamr import PAMR
from models.tcl.masker import Masker
import us

import sclip
from sclip.model import CLIP

from datasets.templates import full_imagenet_templates as custom_template

import random

from models.tcl.tcl import TCL, tv_loss, AreaTCLLoss

from models.tcl.prompter import CLIPPrompter


@MODELS.register_module()
class ImageDecomposition(TCL):
    def __init__(
        self, clip_model, ie_freeze, ie_ignore_last_attn, masker,
        w_tcl, w_tv,
        w_pos_area, w_neg_area, pos_area,
        w_kg, use_region_highlighting_prompt,
        **kwargs
    ):
        nn.Module.__init__(self)

        self.w_tcl = w_tcl
        self.w_tv = w_tv

        pamr_iter = 10
        pamr_kernel = [1, 2, 4, 8, 12, 24]
        self.pamr = PAMR(pamr_iter, pamr_kernel)

        self.frozen_clip: CLIP = sclip.load(clip_model, "cpu")[0]

        self.clip_text_encoder = CLIPPrompter(clip_model)
        assert ie_freeze >= 1, f"for now, ie_freeze >= 1 is required, but {ie_freeze} is given."
        self.clip_image_encoder = CLIPImageFeatureEncoder(
            clip_model,
            feature_extract_index=ie_freeze-1,
            ignore_last_attn=ie_ignore_last_attn,
        )
        self.patch_size = self.clip_image_encoder.patch_size

        masker_backbone = self.clip_image_encoder.clone_masker_backbone(
            ie_freeze)
        masker_backbone.patch_size = self.patch_size
        image_proj = self.clip_image_encoder.clone_proj()
        self.masker = Masker(
            backbone=masker_backbone,
            image_proj=image_proj,
            ignore_last_attn=ie_ignore_last_attn,
            **masker
        )

        self.tcli_loss = InfoNCE()
        self.tclf_loss = ExtendedInfoNCE()

        self.w_pos_area = w_pos_area
        self.w_neg_area = w_neg_area
        self.area_loss = AreaTCLLoss(pos_area)
        self.neg_area_loss = AreaTCLLoss(0.)

        self.ust = False
        self.w_kg = w_kg

        self.learnable_image_bg = nn.Parameter(
            torch.zeros(1, 3, 224, 224),
            requires_grad=use_region_highlighting_prompt
        )

    def train(self, mode=True):
        """Override the default train() to freeze CLIP backbone
        """
        super().train(mode)
        # CLIP encoders are always frozen
        self.clip_image_encoder.eval()
        self.clip_text_encoder.train()

        self.pamr.eval()
        self.frozen_clip.eval()

        if self.ust:
            # Masker IE backbone is frozen in UST phase
            self.masker.image_encoder.backbone.eval()

    def set_train(self, decoder_only: bool):
        """Update requires_grad_ and train/eval mode by `decoder_only` flag.
        """
        self.ust = decoder_only

        # set train mode
        self.train()

        # freeze clip encoders
        self.clip_image_encoder.requires_grad_(False)
        self.clip_text_encoder.prompt_learner.clip_model.requires_grad_(False)
        self.clip_text_encoder.text_encoder.requires_grad_(False)

        self.pamr.requires_grad_(False)
        self.frozen_clip.requires_grad_(False)

        # masker is learnable
        self.masker.image_encoder.requires_grad_(True)

        # self.clip_text_encoder.prompt_learner.ctx.requires_grad_(False)

        if decoder_only:
            self.masker.image_encoder.backbone.requires_grad_(False)

    def cal_iseg_loss(self, image, masks, feature_map, text_emb):
        ret = {}
        ret["mask"] = masks["soft_pos"].detach()
        ret["neg_mask"] = masks["soft_neg"].detach()

        pos_mask = masks["soft_pos"]
        neg_mask = masks["soft_neg"]
        mask = masks["soft_all"]

        # Feature-level contrastive loss
        image_emb = self.masked_pool(feature_map, mask)  # [BNC]
        feat_contrstive_loss = self.tclf_loss(image_emb, text_emb)
        ret["hlf_noun_loss"] = feat_contrstive_loss * self.w_tcl

        # Total variation (TV) regularization loss (Page 5 of TCL's paper)
        tv_img_loss = tv_loss(feature_map)
        ret["tv_img_loss"] = tv_img_loss * self.w_tv
        tv_mask_loss = tv_loss(mask)
        ret["tv_mask_loss"] = tv_mask_loss * self.w_tv

        # Area losses
        ret["area_loss"] = self.area_loss(pos_mask) * self.w_pos_area
        ret["neg_area_loss"] = self.neg_area_loss(neg_mask) * self.w_neg_area

        # Image-level contrastive loss
        pos_mask = F.interpolate(masks["pos"], size=image.shape[2:])
        bg_img = F.interpolate(self.learnable_image_bg,
                               size=image.shape[2:], mode="bilinear")

        fg_img = pos_mask * image + (1-pos_mask) * bg_img
        fg_img_embs = self.clip_image_encoder.clip_forward(fg_img)
        fg_img_embs = us.normalize(fg_img_embs, dim=-1)

        tcli_loss = self.tcli_loss(fg_img_embs, text_emb)
        ret["hli_noun_loss"] = tcli_loss * self.w_tcl

        return ret, fg_img_embs

    @torch.no_grad()
    def get_kg_embedding(self, all_nouns):
        """ Calculate the knowledge guided text embeddings. """
        sentences = [
            random.choice(custom_template).format(noun)
            for noun in all_nouns
        ]
        kg_text_emb = self.clip_text_encoder.wo_prompt_learning(sentences)
        kg_text_emb = us.normalize(kg_text_emb, dim=-1)
        return kg_text_emb

    def cal_kg_loss(self, noun_embs, all_nouns):
        kg_embs = self.get_kg_embedding(all_nouns)
        sims = torch.einsum("bc,bc->b", noun_embs, kg_embs).mean()
        return 1 - sims

    def decode_feature_map(self, image: torch.Tensor):
        with torch.no_grad():
            _, feature_maps_all = self.clip_image_encoder.tcl_forward(
                image, ret_feats=True)

            # feature_maps: (197, batch_size, features)
            # 197 = 16^2 + 1 = patch^2 + class_token
            feature_maps = feature_maps_all[0]

        #  decoded_feat_map [BCHW]: spatial embedding
        decoded_feat_map, _ = self.masker.image_encoder(
            image, feature_maps, ret_feats=True)
        decoded_feat_map = us.normalize(decoded_feat_map, dim=1)
        return decoded_feat_map

    def forward(self, image: torch.Tensor, noun_lists: list[list[str]], **kwargs):
        """
            image: (B, 3, H, W)
            noun_lists: 
                len(noun_lists) == N {the number of selected noun in each sentence}
                len(noun_lists[0]) == B {batch size}
        """

        # Process nouns
        num_nouns = len(noun_lists)
        all_nouns = sum((noun_list for noun_list in noun_lists), [])

        ret = {}  # losses + logs

        decoded_feat_map = self.decode_feature_map(image)

        decoded_feat_map = torch.cat([decoded_feat_map] * num_nouns, dim=0)
        image = torch.cat([image] * num_nouns, dim=0)

        # Build noun embeddings
        noun_embs = self.clip_text_encoder(all_nouns)

        ret["kg_loss"] = self.w_kg * self.cal_kg_loss(noun_embs, all_nouns)

        masks = self.masker(decoded_feat_map, noun_embs)

        new_ret, _ = self.cal_iseg_loss(
            image,
            masks,
            decoded_feat_map,
            noun_embs,
        )

        ret.update(new_ret)

        records = {
            "image": image,
            "masks": masks,
        }

        return ret, records

    def apply_pamr(self, image, mask):
        """ Post-processing (Override from TCL) """
        image = F.interpolate(
            image, mask.shape[-2:], mode="bilinear", align_corners=True)
        mask = self.pamr(image, mask)
        return mask

    @torch.no_grad()
    def build_text_embedding_from_noun(self, classnames: list):
        """
        Args:
            # classnames (List): [classname_0, classname_1, ...]

        Returns:
            text_embs
        """
        text_emb = self.clip_text_encoder(classnames)
        return text_emb

    @torch.no_grad()
    def build_text_embedding_from_text_tokens(self, all_sentences):
        """
        Args:
            text (torch.Tensor): [NUM_CLASSES, NUM_TEMPLATES, CONTEXT_LENGTH] text tokens

        Returns:
            text_embs
        """
        # chunked inference for memory limitation
        text_embs = torch.stack([
            self.clip_text_encoder.wo_prompt_learning(sentences)
            for sentences in all_sentences
        ], dim=0)
        # [N, T, C]
        text_embs = us.normalize(text_embs, dim=-1)
        # [N, C]
        text_embs = text_embs.mean(dim=1)
        text_embs = us.normalize(text_embs, dim=-1)

        return text_embs
