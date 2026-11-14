"""Recognize only batch1, non-speculative default causal RPC context."""
import torch


def normalize_prefix_context(mask, positions, *, batch_size, sequence_length, start_position=0,
                             speculative=False):
    if speculative or batch_size != 1 or sequence_length <= 0:
        return mask, positions
    absent_positions = positions is None or (torch.is_tensor(positions) and positions.numel() == 0)
    default_positions = absent_positions
    if torch.is_tensor(positions) and positions.numel() == sequence_length:
        if positions.ndim == 1 or tuple(positions.shape) == (1, sequence_length):
            expected = torch.arange(start_position, start_position + sequence_length,
                                    device=positions.device, dtype=positions.dtype)
            default_positions = bool(torch.equal(positions.reshape(-1), expected))
    # Never discard an attention mask when actual positions are not the default.
    if not default_positions:
        return mask, positions
    absent_mask = mask is None or (torch.is_tensor(mask) and mask.numel() == 0)
    full_ones = (torch.is_tensor(mask) and tuple(mask.shape) == (1, sequence_length)
                 and bool(torch.all(mask == 1).item()))
    if absent_mask or full_ones:
        return None, None
    return mask, positions
