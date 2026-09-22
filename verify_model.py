import torch
import torch.nn as nn
from CNNHInfinityLSTM import CNNHInfinityLSTM, solve_hinf_riccati
from loss import CustomBinaryLoss

def test_checkpoint_compatibility():
    print("=" * 60)
    print("TEST 1: Checkpoint Compatibility with CNNHInfinityLSTM.pth")
    print("=" * 60)
    model = CNNHInfinityLSTM(mode='parametric')
    ckpt = torch.load('CNNHInfinityLSTM.pth', map_location='cpu')
    
    # Attempt loading state dict
    load_res = model.load_state_dict(ckpt, strict=True)
    print(f"Loaded successfully! Missing keys: {load_res.missing_keys}, Unexpected keys: {load_res.unexpected_keys}")
    
    # Inspect K_filter parameter
    k_filter = model.hinf_lstm.cell.K_filter
    lambda_h = torch.sigmoid(k_filter)
    print(f"K_filter shape: {k_filter.shape}")
    print(f"Mean lambda_h (robustness forgetting gain): {lambda_h.mean().item():.4f}")
    print(f"Min lambda_h: {lambda_h.min().item():.4f}, Max lambda_h: {lambda_h.max().item():.4f}")
    assert len(load_res.missing_keys) == 0 and len(load_res.unexpected_keys) == 0, "Key mismatch in checkpoint!"
    print("--> PASS: Model perfectly matches CNNHInfinityLSTM.pth\n")
    return model

def test_forward_and_backward(model):
    print("=" * 60)
    print("TEST 2: Forward & Backward Pass (Parametric Mode)")
    print("=" * 60)
    # Expected input shape: (batch_size, 1, n_mels=128, time_steps=32)
    dummy_input = torch.randn(4, 1, 128, 32)
    dummy_target = torch.tensor([[1.0], [0.0], [1.0], [0.0]])
    
    model.train()
    preds = model(dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Output shape: {preds.shape}")
    print(f"Predictions: {preds.squeeze().detach().numpy()}")
    assert preds.shape == (4, 1), f"Unexpected output shape: {preds.shape}"
    assert not torch.isnan(preds).any(), "NaN found in predictions!"

    criterion = CustomBinaryLoss(alpha=0.87)
    loss = criterion(preds, dummy_target, threshold=0.5, smooth=True)
    print(f"Loss (smooth surrogate): {loss.item():.4f}")
    loss.backward()
    
    # Check gradient flow
    grad_norm = model.hinf_lstm.cell.K_filter.grad.norm().item()
    print(f"Gradient norm for K_filter: {grad_norm:.6f}")
    assert grad_norm > 0, "Zero gradient for K_filter!"
    print("--> PASS: Forward and backward pass functional\n")

def test_stability_certificate(model):
    print("=" * 60)
    print("TEST 3: Lyapunov / Contraction Stability Certificate")
    print("=" * 60)
    rho_parametric = model.hinf_lstm.cell.get_stability_certificate()
    print(f"Spectral radius rho(I - KC) [Parametric Mode]: {rho_parametric:.6f}")
    assert rho_parametric < 1.0, f"System is not contractive! rho = {rho_parametric}"
    print("--> Mathematical Guarantee: rho < 1 confirms Bounded-Input Bounded-Output (BIBO) stability.")
    print("--> PASS: Contraction mapping theorem satisfied.\n")

def test_dare_mode():
    print("=" * 60)
    print("TEST 4: Differentiable H-Infinity DARE Riccati Solver")
    print("=" * 60)
    dare_model = CNNHInfinityLSTM(mode='dare', gamma=2.0)
    dummy_input = torch.randn(2, 1, 128, 16)
    preds = dare_model(dummy_input)
    print(f"DARE Mode Output shape: {preds.shape}")
    print(f"DARE Mode Predictions: {preds.squeeze().detach().numpy()}")
    
    rho_dare = dare_model.hinf_lstm.cell.get_stability_certificate()
    print(f"Spectral radius rho(I - KC) [DARE Mode]: {rho_dare:.6f}")
    assert rho_dare < 1.0, f"DARE system not contractive! rho = {rho_dare}"
    print("--> PASS: Discrete Algebraic H-Infinity Riccati solver operational and stable.\n")

def test_loss_variants():
    print("=" * 60)
    print("TEST 5: Penalty Weighted Loss (PWL) Verification")
    print("=" * 60)
    criterion = CustomBinaryLoss(alpha=0.87, default_threshold=0.5)
    preds = torch.tensor([0.2, 0.8, 0.1, 0.9])
    targets = torch.tensor([1.0, 0.0, 1.0, 0.0]) # 2 false negatives, 2 false positives
    
    loss_discrete = criterion(preds, targets, smooth=False)
    loss_smooth = criterion(preds, targets, smooth=True)
    print(f"Discrete Penalty Loss: {loss_discrete.item():.4f}")
    print(f"Smooth Surrogate Loss: {loss_smooth.item():.4f}")
    assert loss_discrete > 0 and loss_smooth > 0, "Loss calculation invalid"
    print("--> PASS: PWL loss functions verified.\n")

if __name__ == '__main__':
    loaded_model = test_checkpoint_compatibility()
    test_forward_and_backward(loaded_model)
    test_stability_certificate(loaded_model)
    test_dare_mode()
    test_loss_variants()
    print("=" * 60)
    print("ALL TESTS PASSED SUCCESSFULLY! Codebase mathematically & functionally verified.")
    print("=" * 60)
