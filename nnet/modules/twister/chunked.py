# Memory efficient chunked computations with activation checkpointing.
#
# Both functions compute exactly the same values and gradients as their full counterparts:
# the computation is split along an axis whose elements are independent (frames for the CNNs,
# rows of the InfoNCE similarity matrix), and each chunk is recomputed in backward instead of
# storing its intermediate activations.

# PyTorch
import torch
import torch.utils.checkpoint


def chunked_checkpoint(fn, x, chunk_size):

    """fn(x) computed by chunks of chunk_size elements along dim 0 with activation checkpointing

    fn must act independently on every element of dim 0 (e.g. a CNN on a batch of frames)
    and be deterministic (no dropout), so that the backward recomputation matches the forward.
    Without grad or when x is small, fn(x) is computed directly.
    """

    if chunk_size is None or x.shape[0] <= chunk_size or not torch.is_grad_enabled():
        return fn(x)

    return torch.cat([
        torch.utils.checkpoint.checkpoint(fn, x[i:i + chunk_size], use_reentrant=False)
    for i in range(0, x.shape[0], chunk_size)], dim=0)


def _info_nce_rows(features_x, features_y, start):

    # Similarities of rows start .. start + c with all columns (c, B')
    features = features_x.matmul(features_y.transpose(0, 1))
    rows = torch.arange(features.shape[0], device=features.device)

    # Positives on the global diagonal, InfoNCE = pos - logsumexp
    info_nce = features[rows, start + rows] - torch.logsumexp(features, dim=-1)

    return info_nce, features.argmax(dim=-1)


def info_nce_loss(features_x, features_y, chunk_size=None):

    """InfoNCE between features_x[i] and features_y[i], all other features_y[j] as negatives

    Same values as the TWISTER compute_contrastive_loss, optionally by chunks of rows with checkpointing:
    only (chunk_size, B') similarities exist at a time instead of storing (B', B') for backward.

    Args:
        features_x: (B, L, D)
        features_y: (B, L, D)
        chunk_size: rows per chunk, None = full (B', B') matrix

    Return:
        info_nce_loss: (B',) with B' = B * L
        acc_con: contrastive accuracy (scalar, cpu)
    """

    # Flatten (B', D)
    features_x = features_x.flatten(start_dim=0, end_dim=1)
    features_y = features_y.flatten(start_dim=0, end_dim=1)
    N = features_x.shape[0]

    if chunk_size is None or N <= chunk_size:
        info_nce, argmax = _info_nce_rows(features_x, features_y, 0)

    else:
        checkpoint = torch.is_grad_enabled()
        info_nce, argmax = [], []
        for start in range(0, N, chunk_size):
            args = (features_x[start:start + chunk_size], features_y, start)
            if checkpoint:
                nce_c, argmax_c = torch.utils.checkpoint.checkpoint(_info_nce_rows, *args, use_reentrant=False)
            else:
                nce_c, argmax_c = _info_nce_rows(*args)
            info_nce.append(nce_c)
            argmax.append(argmax_c)
        info_nce, argmax = torch.cat(info_nce), torch.cat(argmax)

    # Accuracy Contrastive
    acc_con = torch.mean(torch.where(argmax.cpu() == torch.arange(0, N), 1.0, 0.0))

    return info_nce, acc_con
