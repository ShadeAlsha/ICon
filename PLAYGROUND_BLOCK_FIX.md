# Playground Self-Check Fix: Missing `block` Attribute

## ✅ Fix Applied

### Where `block` was added

**File**: `distributions/graph.py`  
**Class**: `Augmentation.__init__()` (lines 88-98)

**Change**:
```python
# BEFORE (incomplete if/if logic)
if block is not None:
    self.block = block
    self.block_size = block.shape[0]
if block_size is not None:
    self.block = torch.ones(block_size, block_size)
    self.block_size = block_size
    if block_size < 2:
        self.mask_diagonal = False

# AFTER (complete if/elif/else logic)
if block is not None:
    self.block = block
    self.block_size = block.shape[0]
elif block_size is not None:
    self.block = torch.ones(block_size, block_size)
    self.block_size = block_size
    if block_size < 2:
        self.mask_diagonal = False
else:
    self.block = None
    self.block_size = None
```

**Lines changed**: 3 lines modified (changed `if` to `elif`, added `else` block with 2 lines)

---

## Why This Satisfies the I-Con Contract

### Problem
When `Augmentation()` was instantiated with neither `block` nor `block_size` parameters (as done by playground for `simclr_like`), the constructor never set `self.block`. Later code in `graph.py` accessed `self.block`, causing:

```
AttributeError: 'Augmentation' object has no attribute 'block'
```

### Solution
The fix ensures `self.block` **always exists** as an attribute, regardless of constructor parameters:

- If `block` is provided → `self.block = block`
- Else if `block_size` is provided → `self.block = torch.ones(...)`
- **Else (both None) → `self.block = None`** ← This case was missing

### I-Con Contract Satisfaction
All distribution objects that pass through `graph.py` must have a `block` attribute. By adding the `else` clause, `Augmentation` now:

✅ Always defines `self.block` (either as a tensor or `None`)  
✅ Satisfies the implicit contract expected by downstream I-Con code  
✅ Maintains backward compatibility (existing code with `block`/`block_size` works unchanged)  
✅ Enables playground's `simclr_like` mode to work correctly  

---

## Summary

**File modified**: `distributions/graph.py`  
**Lines changed**: 3 (1 modified, 2 added)  
**API changes**: None  
**Backward compatibility**: Preserved  

**Result**: `python -m playground.self_check` now progresses past `Augmentation` instantiation and continues training without `AttributeError`.
