import torch
import torch.nn as nn
import torch.nn.functional as F

from models.builder import MODELS
from models.tcl.mi import InfoNCE
import us

from models.tcl.noun_decomposition import ImageDecomposition
from models.tcl.decoders import TextDecoder
from sclip import tokenize


def handle_padded_tokens(
    masks: torch.Tensor,
    text_indices: torch.Tensor,
    *,
    sos_token_value=None,
    eos_token_value=None,
    padded_token_value=None
):
    """ Handling

    params:
        masks: torch.Tensor of shape (B, L), the original mask
        text_indices: torch.Tensor of shape (B, ), the indices of eos token
        sos_token_value: the value of the start of sentence token to be updated
        eos_token_value: the value of the end of sentence token to be updated
        padded_token_value: the value of the tokens after the eos token to be updated
    returns:
        The updated mask
    """
    if sos_token_value is not None:
        update_mask = torch.zeros_like(masks)
        update_mask[:, 0] = 1.
        masks = masks * (1 - update_mask) + sos_token_value * update_mask

    if eos_token_value is not None:
        update_mask = torch.zeros_like(masks)
        for i, text_index in enumerate(text_indices):
            update_mask[i, text_index] = 1.
        masks = masks * (1 - update_mask) + eos_token_value * update_mask

    if padded_token_value is not None:
        update_mask = torch.zeros_like(masks)
        for i, text_index in enumerate(text_indices):
            update_mask[i, text_index + 1:] = 1.
        masks = masks * (1 - update_mask) + eos_token_value * update_mask

    return masks


def highlight_txt(tokens, txt_mask, bg_txt):
    """ Highlighting word process

    params:
        tokens: (L, B, C)
        txt_mask: (L, B)
        bg_txt: (L, 1, C)
    """
    fgm = txt_mask[:, :, None]
    output = tokens * fgm + bg_txt * (1 - fgm)
    return output


@MODELS.register_module()
class ImageTextCoDecomposition(ImageDecomposition):
    def __init__(
        self,
        w_hcl,
        w_tseg,
        use_word_highlighting_prompt,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.text_decoder = TextDecoder()
        self.text_mask_w = nn.Parameter(torch.full([], 10.0))
        self.text_mask_b = nn.Parameter(torch.full([], 0.0))

        self.bg_text = nn.Parameter(
            torch.zeros(77, 1, 512),
            requires_grad=use_word_highlighting_prompt
        )

        self.w_hcl = w_hcl
        self.w_tseg = w_tseg

        self.hcl_loss = InfoNCE()

    @torch.no_grad()
    def encode_text_features(self, caption):
        text_token_ids = tokenize(caption, context_length=77, truncate=True)
        text_token_ids = text_token_ids.cuda()

        _, text_hidden_embs = self.frozen_clip.encode_text(text_token_ids, True)

        text_tokens, text_indices = self.frozen_clip.get_word_tokens(
            text_token_ids)
        text_tokens = text_tokens.permute(1, 0, 2)
        return {
            "text_hidden_embs": text_hidden_embs,
            "text_tokens": text_tokens,
            "text_indices": text_indices,
        }

    def get_highlighted_text_emb(
        self,
        text_mask,
        text_tokens,
        text_indices,
    ):
        highlighting_mask = handle_padded_tokens(
            text_mask,
            text_indices,
            sos_token_value=1,
            eos_token_value=1,
            padded_token_value=1,
        )
        highlighted_tokens = highlight_txt(
            text_tokens,
            highlighting_mask.permute(1, 0),
            self.bg_text
        )
        txt_emb = self.frozen_clip.encode_text_from_wordemb(
            highlighted_tokens.permute(1, 0, 2), text_indices)
        txt_emb = us.normalize(txt_emb, dim=-1)
        return txt_emb

    def get_text_masks(self, dense_text_emb, prompt_emb, num_nouns):
        # dense_text_emb: LBC
        # prompt_emb: BC
        L, B, _ = dense_text_emb.shape

        simmap = torch.einsum("lbc,bc->bl", dense_text_emb, prompt_emb)

        # (TOTAL_BS, TEXT_LEN)
        text_logits = simmap * self.text_mask_w + self.text_mask_b
        text_logits = text_logits.view(num_nouns, -1, L)

        # Softmax
        text_logits = torch.cat([
            torch.zeros_like(text_logits[:1]),
            text_logits
        ], dim=0)
        test_mask = F.softmax(text_logits, dim=0)
        test_mask = test_mask[1:].view(B, L)

        masks = {
            "probs": test_mask,
            "logits": text_logits.contiguous().permute(1, 0, 2),
        }

        return masks

    def codecomposition(self, noun_prompt, fg_image_emb, caption, noun_lists, pseudo_text_mask):
        num_nouns = len(noun_lists)
        encoded_text = self.encode_text_features(caption)
        text_hidden_embs = encoded_text['text_hidden_embs']
        text_tokens = encoded_text['text_tokens']
        text_indices = encoded_text['text_indices']

        # (LEN, batch_size, features)
        text_dense_emb = self.text_decoder(text_hidden_embs)
        text_dense_emb = us.normalize(text_dense_emb, dim=-1)

        #
        text_dense_emb = torch.cat([text_dense_emb] * num_nouns, dim=1)
        text_indices = torch.cat([text_indices] * num_nouns, dim=0)
        text_tokens = torch.cat([text_tokens] * num_nouns, dim=1)

        # masks
        text_masks = self.get_text_masks(
            text_dense_emb,
            noun_prompt,
            num_nouns,
        )

        text_mask = text_masks["probs"]
        logits = text_masks["logits"]

        fg_txt_emb = self.get_highlighted_text_emb(
            text_mask,
            text_tokens,
            text_indices,
        )

        # Area losses
        tseg_loss = F.cross_entropy(
            logits,
            pseudo_text_mask,
            ignore_index=-1,
            weight=torch.tensor([0.5] + [1.] * num_nouns).cuda()
        )

        return {
            "tseg_loss": self.w_tseg * tseg_loss,
            "hcl_loss": self.w_hcl * self.hcl_loss(fg_image_emb, fg_txt_emb)
        }

    def forward(self, image, noun_lists, caption, pseudo_text_mask, use_pamr=False):
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

        new_ret, fg_image_emb = self.cal_iseg_loss(
            image,
            masks,
            decoded_feat_map,
            noun_embs,
        )

        ret.update(new_ret)

        new_ret = self.codecomposition(
            noun_embs,
            fg_image_emb,
            caption,
            noun_lists,
            pseudo_text_mask
        )

        ret.update(new_ret)

        if use_pamr:
            masks["soft_pos"] = self.apply_pamr(image, masks["soft_pos"])

        records = {
            "image": image,
            "masks": masks,
        }

        return ret, records
