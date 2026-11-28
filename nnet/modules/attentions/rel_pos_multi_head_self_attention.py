# Copyright 2025, Maxime Burchi.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# PyTorch
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

# Neural Nets
from nnet.modules import layers
from nnet.modules import embeddings
from nnet.modules.attentions import MultiHeadAttention
from nnet.modules.attentions.multi_head_attention import fp32_scores, to_fp32
from nnet.modules.attentions.shared_kv_cache import SharedKVCache
from nnet import structs

class RelPosMultiHeadSelfAttention(MultiHeadAttention):

    """Multi-Head Self-Attention Layer with Relative Sinusoidal Positional Encodings

    Args:
        dim_model: model feature dimension
        num_heads: number of attention heads
        causal: whether the attention is causal or unmasked
        max_pos_encoding: maximum relative distance between elements

    References: 
        Transformer-XL: Attentive Language Models Beyond a Fixed-Length Context, Dai et al.
        https://arxiv.org/abs/1901.02860

    """

    def __init__(
        self, 
        dim_model, 
        num_heads, 
        attn_drop_rate, 
        max_pos_encoding, 
        causal, 
        weight_init="scaled_uniform", 
        bias_init="zeros", 
        output_proj=True,
        apply_bias=True,
        channels_last=True,
        query_chunk_size=None,
        query_chunk_checkpoint=True
    ):
        super(RelPosMultiHeadSelfAttention, self).__init__(
            dim_model=dim_model, 
            num_heads=num_heads, 
            attn_drop_rate=attn_drop_rate, 
            weight_init=weight_init, 
            bias_init=bias_init, 
            output_proj=output_proj,
            channels_last=channels_last
        )

        # Position Embedding Layer
        self.pos_layer = layers.Linear(self.dim_model, self.dim_model)
        self.causal = causal

        # Global content and positional bias
        self.apply_bias = apply_bias
        if self.apply_bias:
            self.u = nn.Parameter(torch.Tensor(self.dim_model)) # Content bias
            nn.init.zeros_(self.u)
            self.v = nn.Parameter(torch.Tensor(self.dim_model)) # Pos bias
            nn.init.zeros_(self.v)

        # Relative Sinusoidal Positional Encodings
        self.rel_pos_enc = embeddings.RelSinPosEncoding(max_pos_encoding, self.dim_model, self.causal)

        # Query chunking: compute attention for query_chunk_size queries at a time (causal + mask only).
        # With query_chunk_checkpoint, each chunk is recomputed in backward (same dropout mask, RNG state preserved)
        # so the (B, H, T, Th + T) attention scores / weights / dropout masks are never stored for backward.
        self.query_chunk_size = query_chunk_size
        self.query_chunk_checkpoint = query_chunk_checkpoint

    def rel_to_abs(self, att_scores):

        """Relative to absolute position indexing

        Args:
            att_scores: absolute-by-relative indexed attention scores of shape 
            (B, H, T, Th + 2*T-1) for full context and (B, H, T, Th + T) for causal context

        Return:
            att_scores: absolute-by-absolute indexed attention scores of shape (B, H, T, Th + T)

        References: 
            causal context:
            Music Transformer, Huang et al.
            https://arxiv.org/abs/1809.04281
            
            full context:
            Attention Augmented Convolutional Networks, Bello et al.
            https://arxiv.org/abs/1904.09925

        """

        # Causal Context
        if self.causal:

            # Att Scores (B, H, T, Th + T)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, 1 + Th + T)
            att_scores = F.pad(att_scores, pad=(1, 0), value=0)

            # Flatten (B, H, T + TTh + TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # Start Padding (B, H, Th + T + TTh + TT)
            att_scores = F.pad(att_scores, pad=(seq_length2 - seq_length1, 0), value=0)

            # Reshape (B, H, 1 + T, Th + T)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, 1:]

        # Full Context
        else:

            # Att Scores (B, H, T, Th + 2*T-1)
            batch_size, num_heads, seq_length1, seq_length2 = att_scores.size()

            # Column Padding (B, H, T, Th + 2*T)
            att_scores = F.pad(att_scores, pad=(0, 1), value=0)

            # Flatten (B, H, TTh + 2*TT)
            att_scores = att_scores.reshape(batch_size, num_heads, -1)

            # End Padding (B, H, TTh + 2*TT + Th + T - 1)
            att_scores = F.pad(att_scores, pad=(0, seq_length2 - seq_length1), value=0)

            # Reshape (B, H, T + 1, Th + 2*T-1)
            att_scores = att_scores.reshape(batch_size, num_heads, 1 + seq_length1, seq_length2)

            # Slice (B, H, T, Th + T)
            att_scores = att_scores[:, :, :seq_length1, seq_length1-1:]

        return att_scores
    
    def forward(self, x, mask=None, hidden=None, return_hidden=False, return_att_w=False):
        return self.forwardQKV(Q=x, K=x, V=x, mask=mask, hidden=hidden, return_hidden=return_hidden, return_att_w=return_att_w)

    def forwardQKV(self, Q, K, V, mask=None, hidden=None, return_hidden=False, return_att_w=False, detach_hidden=True):

        """Scaled Dot-Product Self-Attention with relative sinusoidal position encodings

        Args:
            Q: Query of shape (B, T, D)
            K: Key of shape (B, T, D)
            V: Value of shape (B, T, D)
            mask: Optional position mask of shape (1 or B, 1 or H, 1 or T, 1 or T)
            hidden: Optional Key and Value hidden states for decoding
        
        Return:
            O: Attention output of shape (B, T, D)
            att_w: Attention weights of shape (B, H, T, Th + T)
            hidden: Key and value hidden states

        """

        # Shared KV cache imagination (mask is built from the cache)
        if isinstance(hidden, SharedKVCache):
            return self.forward_shared_kv(Q, K, V, cache=hidden, return_hidden=return_hidden, return_att_w=return_att_w)

        # Input Linear Layers
        Q, K, V = self.forward_inputs(Q, K, V)

        # Batch size B
        batch_size = Q.size(0)

        # Hidden State Provided
        if hidden is not None:
            K = torch.cat([hidden[0], K], dim=1)
            V = torch.cat([hidden[1], V], dim=1)

        # Update Hidden State
        if return_hidden:
            hidden = (K.detach(), V.detach()) if detach_hidden else (K, V)

        # Add Bias
        if self.apply_bias:
            Qu = Q + self.u
            Qv = Q + self.v
        else:
            Qu = Q
            Qv = Q

        # Relative Positional Embeddings (B, Th + 2*T-1, D) / (B, Th + T, D)
        E = self.pos_layer(self.rel_pos_enc(batch_size, Q.size(1), K.size(1) - Q.size(1)))

        # Reshape and Transpose (B, T, D) -> (B, H, T, d)
        Qu = Qu.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        Qv = Qv.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        
        # Reshape and Transpose (B, Th + T, D) -> (B, H, Th + T, d)
        K = K.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)
        V = V.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Reshape and Transpose (B, Th + 2*T-1, D) -> (B, H, Th + 2*T-1, d) / (B, Th + T, D) -> (B, H, Th + T, d)
        E = E.reshape(batch_size, -1, self.num_heads, self.dim_head).transpose(1, 2)

        # Query chunked attention (B, H, T, d), see forward_query_chunks
        if self.use_query_chunks(Q.size(1), mask, return_att_w):
            O = self.forward_query_chunks(Qu, Qv, K, V, E, mask)

        else:

            # att_scores and weights (B, H, T, Th + T), float32 (see fp32_scores)
            with fp32_scores(Qu.device):
                att_scores_K = to_fp32(Qu).matmul(to_fp32(K).transpose(2, 3))
                att_scores_E = self.rel_to_abs(to_fp32(Qv).matmul(to_fp32(E).transpose(2, 3)))
                att_scores = (att_scores_K + att_scores_E) / K.shape[-1]**0.5

                # Apply mask
                if mask is not None:
                    # (B, H, L, Context)
                    att_scores += (mask.logical_not() * -1e9)

                # Att weights (B, H, T, Th + T)
                att_w = att_scores.softmax(dim=-1)

            # Att Dropout
            att_w = self.dropout(att_w)

            # Att output (B, H, T, d)
            O = att_w.to(V.dtype).matmul(V)

        # Transpose and Reshape (B, H, T, d) -> (B, T, D)
        O = O.transpose(1, 2).reshape(batch_size, -1,  self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        # Format Outputs
        outputs = structs.AttrDict(x=O)
        if return_hidden:
            outputs.hidden = hidden
        if return_att_w:
            outputs.att_w = att_w

        return outputs

    def use_query_chunks(self, seq_len, mask, return_att_w):

        # Causal with explicit mask only: masked positions hide the out of range relative positions
        return self.query_chunk_size is not None and self.causal and mask is not None and not return_att_w and seq_len > self.query_chunk_size

    def forward_query_chunks(self, Qu, Qv, K, V, E, mask):

        """Causal relative position attention computed query_chunk_size queries at a time

        Same function and gradients as the full computation in forwardQKV: every query row has its own softmax,
        so splitting rows does not change the result. With query_chunk_checkpoint, the chunk internals
        (scores, weights, dropout mask) are recomputed in backward with the preserved RNG state,
        reproducing the dropout mask used in forward.

        Args:
            Qu, Qv: (B, H, T, d) queries with content / position bias
            K, V: (B, H, Th + T, d)
            E: (B, H, Th + T, d) relative position embeddings, row r = distance Th + T - 1 - r
            mask: (1 or B, 1, T, Th + T)

        Return:
            O: (B, H, T, d)
        """

        T = Qu.size(2)

        # Contiguous once, avoids a (B, H, Th + T, d) copy per chunk matmul
        Qu, Qv, K, V, E = Qu.contiguous(), Qv.contiguous(), K.contiguous(), V.contiguous(), E.contiguous()

        checkpoint = self.query_chunk_checkpoint and torch.is_grad_enabled()

        O = []
        for start in range(0, T, self.query_chunk_size):
            end = min(T, start + self.query_chunk_size)
            args = (Qu[:, :, start:end], Qv[:, :, start:end], K, V, E, mask[:, :, start:end], start, T)
            if checkpoint:
                O.append(torch.utils.checkpoint.checkpoint(self.attend_query_chunk, *args, use_reentrant=False, preserve_rng_state=True))
            else:
                O.append(self.attend_query_chunk(*args))

        return torch.cat(O, dim=2)

    def attend_query_chunk(self, Qu, Qv, K, V, E, mask, start, T):

        """Attention for queries start .. start + c of a length T sequence (B, H, c, d)"""

        B, H, c, d = Qu.shape
        Tk = K.size(2)

        # Position scores: query i (key index Th + i) and key j, distance Th + i - j, E row (T - 1 - i) + j
        # Equivalent to rel_to_abs on the full matrix; out of range (j > Th + i) entries are causally masked
        i = torch.arange(start, start + c, device=Qu.device)
        j = torch.arange(Tk, device=Qu.device)
        rows = (j.unsqueeze(0) + (T - 1) - i.unsqueeze(1)).clamp(0, Tk - 1)

        # Scores and weights in float32 (see fp32_scores)
        with fp32_scores(Qu.device):

            # Content scores (B, H, c, Th + T)
            att_scores_K = to_fp32(Qu).matmul(to_fp32(K).transpose(2, 3))

            # Position scores (B, H, c, Th + T)
            att_scores_E = to_fp32(Qv).matmul(to_fp32(E).transpose(2, 3)).gather(-1, rows.expand(B, H, c, Tk))

            att_scores = (att_scores_K + att_scores_E) / d**0.5

            # Apply mask (1 or B, 1, c, Th + T)
            att_scores += (mask.logical_not() * -1e9)

            # Att weights
            att_w = att_scores.softmax(dim=-1)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, H, c, d)
        return att_w.to(V.dtype).matmul(V)

    def attend_shared_kv_chunk(self, Qu, Qv, E, Kp, Vp, cache, a, b):

        """Shared kv attention for start points a .. b-1

        Args:
            Qu, Qv: (B, H, L, d) queries of all start points
            E: (C+1, H, d) relative position embeddings indexed by distance
            Kp, Vp: imagined keys / values, lists of (B, H, L, d), oldest first, current token last
            cache: SharedKVCache
            a, b: start point range

        Return:
            O: (B, H, b-a, d)
            att_w: (B, H, b-a, S + P) attention weights, S observe keys of the chunk band then P imagined keys
        """

        B, H, L, d = Qu.shape
        C, h = cache.context, cache.step
        c = b - a

        # Scores and weights in float32 (see fp32_scores), values stay in the autocast dtype
        with fp32_scores(Qu.device):
            att_w, S = self.shared_kv_chunk_weights(to_fp32(Qu[:, :, a:b]), to_fp32(Qv[:, :, a:b]), to_fp32(E), Kp, cache, a, b)

        # Att Dropout
        att_w = self.dropout(att_w)

        # Att output (B, H, c, d)
        Vs = cache.shared_v[:, :, b - S:b]
        O = att_w[..., :S].to(Vs.dtype).matmul(Vs)
        for j, v in enumerate(Vp):
            O = O + att_w[..., S + j : S + j + 1] * v[:, :, a:b]

        return O, att_w

    def shared_kv_chunk_weights(self, Qu, Qv, E, Kp, cache, a, b):

        """Attention weights of attend_shared_kv_chunk (B, H, c, S + P) and the number S of observe keys"""

        B, H, c, d = Qu.shape
        C, h = cache.context, cache.step

        # Positional scores for every distance (B, H, c, C+1), einsum batches over heads (no expand of E over B)
        P = torch.einsum("bhld,chd->bhlc", Qv, E)

        ###############################################################################
        # Observe (shared) keys: band s in [a+1-C+h, b-1], positions outside are masked anyway
        ###############################################################################

        k1 = b
        k0 = min(max(0, a + 1 - C + h), k1)
        Ks = to_fp32(cache.shared_k[:, :, k0:k1]) # (B, H, S, d)
        S = k1 - k0

        # Query start t, key observe position s: distance t + 1 + h - s (c, S)
        t_idx = torch.arange(a, b, device=Qu.device)
        s_idx = torch.arange(k0, k1, device=Qu.device)
        dist = t_idx.unsqueeze(1) + 1 + h - s_idx.unsqueeze(0)

        # Content + position scores (B, H, c, S)
        scores_s = Qu.matmul(Ks.transpose(2, 3)) + P.gather(-1, dist.clamp(0, C).expand(B, H, c, S))

        # Band + episode mask: max(t+1-C+h, ep_start) <= s <= t  (B, 1, c, S)
        lower = torch.maximum(t_idx + 1 - C + h, cache.ep_start[:, a:b]) # (B, c)
        mask_s = (s_idx.view(1, 1, S) >= lower.unsqueeze(-1)) & (dist >= 1 + h).unsqueeze(0)
        mask_s = mask_s.unsqueeze(1)

        ###############################################################################
        # Imagined (private) keys, one per start point
        ###############################################################################

        # Per key dot products avoid stacking (and saving) a new (B*L, P, D) tensor each step
        n = len(Kp) - 1
        scores_p = torch.stack([(Qu * to_fp32(k[:, :, a:b])).sum(dim=-1) + P[..., n - j] for j, k in enumerate(Kp)], dim=-1) # (B, H, c, P)

        ###############################################################################
        # Attention
        ###############################################################################

        # (B, H, c, S + P)
        att_scores = torch.cat([scores_s, scores_p], dim=-1) / d**0.5
        att_scores[..., :S] += (mask_s.logical_not() * -1e9)

        # Att weights
        att_w = att_scores.softmax(dim=-1)

        return att_w, S

    def forward_shared_kv(self, Q, K, V, cache, return_hidden=False, return_att_w=False):

        """Single step attention for B*L imagination rollouts sharing the observe keys / values

        Numerically equivalent to forwardQKV with the per start point (B*L, C, D) hidden states
        built by the original implementation, without materializing them.

        Args:
            Q, K, V: current imagined token of shape (B*L, 1, D)
            cache: SharedKVCache with observe keys / values (B, L, D)

        Return:
            O: Attention output of shape (B*L, 1, D)
            att_w: Attention weights of shape (B, H, L, S + P), observe keys band then imagined keys (start_chunk_size=None only)
            hidden: SharedKVCache for the next step
        """

        assert self.causal, "shared kv cache requires causal attention"

        # Input Linear Layers (B*L, 1, D)
        Q, K, V = self.forward_inputs(Q, K, V)

        # Sizes
        B, L, C, h = cache.batch_size, cache.seq_len, cache.context, cache.step
        H, d = self.num_heads, self.dim_head
        assert cache.num_heads == H
        assert Q.shape[0] == B * L and Q.shape[1] == 1, "expected ({}, 1, D) queries, got {}".format(B * L, tuple(Q.shape))
        assert C < self.rel_pos_enc.max_len, "att context {} exceeds max_pos_encoding {}".format(C, self.rel_pos_enc.max_len)

        # Add Bias
        if self.apply_bias:
            Qu = Q + self.u
            Qv = Q + self.v
        else:
            Qu = Q
            Qv = Q

        # (B*L, 1, D) -> (B, H, L, d), start points t laid out along the query axis
        to_heads = lambda x: x.reshape(B, L, H, d).transpose(1, 2)
        Qu, Qv = to_heads(Qu), to_heads(Qv)

        # Relative position embeddings indexed by distance E[dist], dist = 0..C (C+1, D)
        R = self.rel_pos_enc.pos_encoding[0, self.rel_pos_enc.max_len - 1 - C : self.rel_pos_enc.max_len].flip(0)
        E = self.pos_layer(R).reshape(C + 1, H, d) # (C+1, H, d)

        # Imagined (private) keys / values, previous tokens (detached) + current token, distances n..0
        Kp = [to_heads(k) for k in cache.private_k] + [to_heads(K)]
        Vp = [to_heads(v) for v in cache.private_v] + [to_heads(V)]

        # Attention by chunks of start points (B, H, L, d)
        chunk = cache.start_chunk_size or L
        assert not (return_att_w and chunk < L), "return_att_w requires start_chunk_size=None"
        O, att_w = [], None
        for a in range(0, L, chunk):
            O_c, att_w = self.attend_shared_kv_chunk(Qu, Qv, E, Kp, Vp, cache, a, min(L, a + chunk))
            O.append(O_c)
        O = torch.cat(O, dim=2) if len(O) > 1 else O[0]

        # (B, H, L, d) -> (B*L, 1, D)
        O = O.transpose(1, 2).reshape(B * L, 1, self.dim_model)

        # Output linear layer
        O = self.forward_outputs(O)

        # Format Outputs
        outputs = structs.AttrDict(x=O)
        if return_hidden:
            outputs.hidden = cache.append(K, V)
        if return_att_w:
            outputs.att_w = att_w

        return outputs