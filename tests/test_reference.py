import torch
import torch.nn.functional as F
import pytest


from reference.gqa_reference import gqa_reference


@pytest.fixture(autouse=True)
def seed():
    torch.manual_seed(42)




def test_output_shape_basic():
    """Output should be [B, H_q, S, D], same as Q."""
    B, H_q, H_kv, S, D = 2, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)
    out = gqa_reference(Q, K, V)
    assert out.shape == (B, H_q, S, D)


def test_output_shape_single_kv_head():
    """Single KV head (MQA case)."""
    B, H_q, H_kv, S, D = 1, 16, 1, 64, 32
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)
    out = gqa_reference(Q, K, V)
    assert out.shape == (B, H_q, S, D)




def test_matches_sdpa_no_causal():
    """Compare against PyTorch SDPA with expanded KV, no causal mask."""
    B, H_q, H_kv, S, D = 2, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)

    group_size = H_q // H_kv
    K_exp = K.repeat_interleave(group_size, dim=1)
    V_exp = V.repeat_interleave(group_size, dim=1)
    expected = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=False)

    out = gqa_reference(Q, K, V, causal=False)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max().item()}"


def test_matches_sdpa_causal():
    """Compare against PyTorch SDPA with causal mask."""
    B, H_q, H_kv, S, D = 2, 8, 2, 128, 64
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)

    group_size = H_q // H_kv
    K_exp = K.repeat_interleave(group_size, dim=1)
    V_exp = V.repeat_interleave(group_size, dim=1)
    expected = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)

    out = gqa_reference(Q, K, V, causal=True)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max().item()}"


def test_mha_case():
    """When H_q == H_kv, GQA should reduce to standard MHA."""
    B, H, S, D = 2, 8, 64, 32
    Q = torch.randn(B, H, S, D)
    K = torch.randn(B, H, S, D)
    V = torch.randn(B, H, S, D)

    expected = F.scaled_dot_product_attention(Q, K, V, is_causal=False)
    out = gqa_reference(Q, K, V, causal=False)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max().item()}"


def test_mqa_case():
    """When H_kv == 1, GQA should reduce to MQA."""
    B, H_q, S, D = 2, 8, 64, 32
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, 1, S, D)
    V = torch.randn(B, 1, S, D)

    K_exp = K.repeat_interleave(H_q, dim=1)
    V_exp = V.repeat_interleave(H_q, dim=1)
    expected = F.scaled_dot_product_attention(Q, K_exp, V_exp, is_causal=True)

    out = gqa_reference(Q, K, V, causal=True)
    assert torch.allclose(out, expected, atol=1e-5), \
        f"Max diff: {(out - expected).abs().max().item()}"




def test_seq_len_1():
    """Sequence length of 1 — causal mask should be trivial."""
    B, H_q, H_kv, S, D = 1, 4, 2, 1, 16
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)
    out = gqa_reference(Q, K, V, causal=True)

    V_exp = V.repeat_interleave(H_q // H_kv, dim=1)
    assert torch.allclose(out, V_exp, atol=1e-5)


def test_causal_no_future_leakage():
    """Verify position 0 attends only to itself, not future positions."""
    B, H_q, H_kv, S, D = 1, 4, 2, 32, 16
    Q = torch.randn(B, H_q, S, D)
    K = torch.randn(B, H_kv, S, D)
    V = torch.randn(B, H_kv, S, D)

    out_causal = gqa_reference(Q, K, V, causal=True)
  
    V_modified = V.clone()
    V_modified[:, :, 1:, :] = torch.randn_like(V_modified[:, :, 1:, :])
    out_modified = gqa_reference(Q, K, V_modified, causal=True)

    assert torch.allclose(out_causal[:, :, 0, :], out_modified[:, :, 0, :], atol=1e-6), \
        "Position 0 output changed when future V values changed — causal mask is broken"


def test_group_size_assertion():
    """H_q must be divisible by H_kv."""
    Q = torch.randn(1, 7, 32, 16)
    K = torch.randn(1, 3, 32, 16)
    V = torch.randn(1, 3, 32, 16)
    with pytest.raises((AssertionError, ValueError)):
        gqa_reference(Q, K, V)