# I-Con Playground: Testing Quick Reference

## Quick Test Commands

### 1. Run All Unit Tests (2-3 seconds)
```bash
cd "/Users/siddharthmanne/Coding Projects/ICon"
pytest tests/test_divergences.py \
       tests/test_config_integration.py \
       tests/test_device_handling.py \
       -v
```

**Expected**: 44 passed, 2 skipped

---

### 2. Run Integration Test (~20 seconds)
```bash
pytest tests/test_integration_pure_pytorch.py::TestIntegrationPurePyTorch::test_minimal_training_cpu -v
```

**Expected**: 1 passed

---

### 3. Test CLI with New Flags (~20 seconds)
```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --icon_mode simclr_like \
  --divergence js \
  --optimizer adam \
  --weight_decay 0.001 \
  --epochs 1 \
  --batch_size 32 \
  --num_workers 0 \
  --cpu \
  --no_viz \
  --no_probe
```

**Expected**: "Device: CPU (forced)" and "Experiment complete!"

---

### 4. Test GPU Requirement Error
```bash
# Only run this on a machine WITHOUT GPU
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --epochs 1 \
  --gpu
```

**Expected** (if no GPU): `RuntimeError: GPU requested (--gpu) but no GPU available.`

---

### 5. Test MPS (Apple Silicon GPU) - If Available
```bash
python -m playground.playground_cli \
  --dataset mnist \
  --backbone mlp \
  --divergence rkl \
  --optimizer adamw \
  --epochs 1 \
  --num_workers 0 \
  --gpu
```

**Expected**: "Device: Apple Silicon GPU (MPS)" or "Device: CUDA GPU"

---

### 6. Test Different Divergences
```bash
# KL divergence (default)
python -m playground.playground_cli --dataset mnist --backbone mlp --divergence kl --epochs 1 --cpu --no_viz --no_probe

# Reverse KL
python -m playground.playground_cli --dataset mnist --backbone mlp --divergence rkl --epochs 1 --cpu --no_viz --no_probe

# Jensen-Shannon
python -m playground.playground_cli --dataset mnist --backbone mlp --divergence js --epochs 1 --cpu --no_viz --no_probe
```

**Expected**: Different final loss values for each divergence

---

### 7. Test Different Optimizers
```bash
# Adam
python -m playground.playground_cli --dataset mnist --backbone mlp --optimizer adam --epochs 1 --cpu --no_viz --no_probe

# AdamW (default)
python -m playground.playground_cli --dataset mnist --backbone mlp --optimizer adamw --epochs 1 --cpu --no_viz --no_probe

# SGD
python -m playground.playground_cli --dataset mnist --backbone mlp --optimizer sgd --epochs 1 --cpu --no_viz --no_probe
```

**Expected**: All complete successfully

---

### 8. Test Recipe (longer, ~2-3 minutes)
```bash
python -m playground.playground_cli \
  --recipe simclr_vs_sne \
  --dataset mnist \
  --epochs 2 \
  --batch_size 64 \
  --num_workers 0 \
  --cpu \
  --no_viz
```

**Expected**: Two experiments run, comparison plot generated

---

## Verification Checklist

Use this checklist to verify implementation:

- [ ] All unit tests pass (44/44)
- [ ] Integration test passes
- [ ] CLI with `--divergence js` works
- [ ] CLI with `--optimizer adam` works
- [ ] CLI with `--weight_decay 0.01` works
- [ ] CPU mode (`--cpu`) works
- [ ] GPU mode (`--gpu`) works (if GPU available)
- [ ] GPU requirement error shows clear message (if no GPU)
- [ ] Device is logged explicitly
- [ ] Recipe system works
- [ ] Embeddings are saved
- [ ] Model checkpoint is saved

---

## File Locations

After running experiments, check these files exist:

```
playground_runs/
└── mnist_mlp_simclr_like_TIMESTAMP/
    ├── config.json           # Configuration used
    ├── embeddings.npz        # Learned embeddings
    ├── logs.json            # Training logs
    └── final_model.pt       # Model checkpoint
```

---

## Common Issues

### Issue: "No module named pytest"
**Solution**: `pip install pytest`

### Issue: MPS warnings about pin_memory
**Solution**: This is expected, not an error. Use `--num_workers 0` to reduce warnings.

### Issue: Recipe takes too long
**Solution**: Use `--epochs 1` and `--batch_size 64` for faster testing.

### Issue: CUDA out of memory
**Solution**: Reduce `--batch_size` or use `--cpu`.

---

## Success Criteria

✅ All tests pass
✅ CLI works with new flags
✅ Device selection is explicit
✅ GPU request without GPU raises error (not silent fallback)
✅ Divergences can be selected
✅ Optimizers can be selected
✅ Recipes work end-to-end
✅ Results saved correctly
