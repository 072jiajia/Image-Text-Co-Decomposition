import torch
import torch.nn as nn

from models.tcl.clip_builder import get_clip_textenc
from sclip import tokenize

class TextEncoder(nn.Module):
    def __init__(self, clip_model):
        super().__init__()
        self.transformer = clip_model.transformer
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final
        self.text_projection = clip_model.text_projection
        # self.dtype = clip_model.dtype

    def forward(self, prompts, eos_indices):
        x = prompts + self.positional_embedding # .type(self.dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x) #.type(self.dtype)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = x[torch.arange(x.shape[0]), eos_indices] @ self.text_projection

        return x


class PromptLearner(nn.Module):
    def __init__(self, clip_model, n_ctx=16):
        super().__init__()
        self.clip_model = clip_model
        self.n_ctx = n_ctx

        embedding = tokenize("")
        self.sos = embedding[:, 0:1]
        self.eos = embedding[:, 1:2]

        # dtype = clip_model.dtype
        # ctx_dim = clip_model.ln_final.weight.shape[0]

        with torch.no_grad():
            embedding = clip_model.token_embedding(self.sos)#.type(dtype)
        self.register_buffer("token_prefix", embedding)
        # self.register_buffer("token_suffix", embedding[:, 1 + n_ctx:, :])

        # use given words to initialize context vectors
        prompt = tokenize("A photo of" + " x" * 50)
        with torch.no_grad():
            embedding = clip_model.token_embedding(prompt)#.type(dtype)
        ctx_vectors = embedding[0, 1: 1 + n_ctx, :]

        self.ctx = nn.Parameter(ctx_vectors, requires_grad=True)
    
    def encode_sentence(self, text):
        tokens = tokenize(text, context_length=77, truncate=True)
        tokens = tokens.cuda()

        embedding = self.clip_model.token_embedding(tokens)
        eos_indices = tokens.argmax(dim=-1)

        return embedding, eos_indices

    def forward(self, text):
        """
            tokens: [B, L]
        """
        tokens = tokenize(text, context_length=77-self.n_ctx, truncate=True)
        tokens = tokens[:, 1:].to(self.ctx.device)

        B, L = tokens.shape

        embedding = self.clip_model.token_embedding(tokens)
        eos_indices = (self.n_ctx + 1) + tokens.argmax(dim=-1)

        ctx = self.ctx.expand(B, -1, -1)
        prefix = self.token_prefix.expand(B, -1, -1)

        prompts = torch.cat(
            [
                prefix,  # (n_cls, 1, dim)
                ctx,     # (n_cls, n_ctx, dim)
                embedding,  # (n_cls, *, dim)
            ],
            dim=1,
        )

        return prompts, eos_indices


class CLIPPrompter(nn.Module):
    def __init__(self, clip_model, n_ctx=16):
        super().__init__()
        clip_model = get_clip_textenc(clip_model)

        self.prompt_learner = PromptLearner(clip_model, n_ctx)
        self.text_encoder = TextEncoder(clip_model)

    def train(self, mode=True):
        """Override the default train() to freeze CLIP backbone
        """
        super().train(mode)
        # CLIP encoders are always frozen
        self.prompt_learner.clip_model.eval()
        self.text_encoder.eval()

    def forward(self, text, normalize=True) -> torch.Tensor:
        prompts, eos_indices = self.prompt_learner(text)
        text_emb = self.text_encoder(prompts, eos_indices)
        if normalize:
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb

    def wo_prompt_learning(self, text, normalize=True):
        prompts, eos_indices = self.prompt_learner.encode_sentence(text)
        text_emb = self.text_encoder(prompts, eos_indices)
        if normalize:
            text_emb = text_emb / text_emb.norm(dim=-1, keepdim=True)
        return text_emb
