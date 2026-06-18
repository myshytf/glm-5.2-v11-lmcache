# GLM-5.2 v11 + LMCache Deployment

Production deployment of **GLM-5.2 NVFP4** on vLLM with **LMCache** multi-process connector, B12X MLA sparse attention, and MTP speculative decoding.

**DCP=4 + LMCache is now fully working** — all 4 DCP ranks store and lookup correctly (100% hit rate).

## Hardware Requirements

- **GPU**: 8× NVIDIA GPU with ≥96GB VRAM (tested on RTX 5090/PRO 6000)
- **RAM**: ≥128GB system memory (48GB allocated to LMCache L1)
- **Storage**: ~500GB NVMe SSD for model weights + LMCache L2
- **CUDA**: 13.2+
- **Docker** with NVIDIA Container Toolkit

## Quick Start

```bash
# 1. Clone
git clone https://github.com/myshytf/glm-5.2-v11-lmcache.git
cd glm-5.2-v11-lmcache

# 2. Download model (~435GB, ~70 min with hf_transfer)
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download lukealonso/GLM-5.2-NVFP4 --local-dir /models/GLM-5.2-NVFP4

# 3. Configure
cp .env.example .env
# Edit .env: set VLLM_API_KEY, verify MODEL path

# 4. Launch
docker compose up -d

# 5. Wait for boot (~15-20 min for weight load + CUDA graph capture)
docker logs -f glm52-v11
# Look for: "Application startup complete"

# 6. Verify
curl http://localhost:5318/health
curl http://localhost:5318/v1/models -H "Authorization: Bearer YOUR_KEY"
```

## Configuration

### Key Parameters

- **`DCP_SIZE`**: Decode Context Parallel size. `1` = stable without DCP patches. `4` = **working** with DCP+LMCache patches
- **`NUM_SPECULATIVE_TOKENS`**: MTP speculative decode tokens (default: `2`)
- **`MAX_MODEL_LEN`**: Maximum context length (default: `450000`)
- **`GLM52_LMCACHE_CHUNK_SIZE`**: LMCache chunk size. Use `512` for DCP=4 (= `block_size × dcp_size`)
- **`GLM52_LMCACHE_L1_GB`**: LMCache L1 RAM cache (default: `48`)
- **`GLM52_LMCACHE_L2_GB`**: LMCache L2 NVMe cache (default: `500`)

### DCP=4 with LMCache

DCP shards the KV cache across DCP groups, reducing per-GPU decode memory. Three patches enable DCP>1 with LMCache:

1. **`patch_dcp_lmcache.py`** — DCP-aware cache keys (`kv_rank = rank % dcp_size`), block sizing (`vllm_block_size × dcp`), and `compute_extra_count`
2. **`patch_dcp_mla_store.py`** — Fixes MLA store guard: first `dcp_size` ranks store (one per DCP shard), not just rank 0
3. DCP LSE combine is already handled by vLLM in `mla_attention.py` — no patch needed

```bash
# In .env:
DCP_SIZE=4
GLM52_LMCACHE_CHUNK_SIZE=512   # must be >= block_size * dcp_size (128 * 4 = 512)
```

**DCP rank layout** (TP=8, DCP=4):
- DCP group 0: ranks [0,1,2,3] — each holds a different KV shard (DCP rank 0-3)
- DCP group 1: ranks [4,5,6,7] — MLA TP shares KV with group 0
- Store: ranks 0-3 (first `dcp_size` ranks, one per DCP shard)
- Lookup: expands to all 4 DCP ranks via `no_worker_id_version()`

## Architecture

```
┌─────────────────────────────────────────────────┐
│              Docker Container                    │
│  ┌─────────────┐    ┌─────────────────────────┐ │
│  │ LMCache MP  │    │   vLLM Engine           │ │
│  │ Server      │◄──►│   (TP=8, DCP=4)         │ │
│  │ port 5555   │    │                         │ │
│  └──────┬──────┘    │  B12X MLA Sparse        │ │
│         │           │  MTP Spec Decode (x2)    │ │
│  ┌──────▼──────┐    │  NVFP4 Quantization     │ │
│  │ L1: 48GB    │    └─────────────────────────┘ │
│  │ L2: 500GB   │    port 5318                   │
│  │ (NVMe)      │                                │
│  └─────────────┘                                │
└─────────────────────────────────────────────────┘
```

## Included Patches

### Stability Patches (always applied)

- **`patch_kv_xfer_assert`** — Guards against LMCache duplicate callback crash
- **`patch_fs_l2_adapter`** — L2 adapter: capacity limit, byte accounting, eviction, startup scan
- **`patch_l1_evict_flush_to_l2`** — L1 eviction flushes to L2 before discard
- **`patch_preempt_safe_restore`** — Async-aware preempted request KV restore from LMCache

### DCP Patches (required for DCP>1)

- **`patch_dcp_lmcache`** — DCP-aware cache keys, block sizing, extra_count
- **`patch_dcp_mla_store`** — Fixes MLA store guard for DCP (first `dcp_size` ranks store)

## Monitoring

```bash
# LMCache status (L1/L2 objects, memory, prefetch stats)
curl http://localhost:8088/status | python3 -m json.tool

# LMCache server log (lookup hit rates)
docker exec glm52-v11 tail -f /tmp/lmcache_mp_server.log | grep "Prefetch.*completed"

# vLLM logs (external cache hit rate, KV usage)
docker logs glm52-v11 2>&1 | grep "External prefix cache hit"
```

### Verifying DCP+LMCache lookup hits

```bash
# Should show "X/X prefix hits" (all DCP ranks found), not "1/X"
docker exec glm52-v11 tail -20 /tmp/lmcache_mp_server.log | grep "Prefetch.*completed"
# Good: "884/884 prefix hits (884 L1, 0 L2)"
# Bad:  "1/564 prefix hits (1 L1, 0 L2)"  ← only DCP rank 0 stored
```

## Troubleshooting

### Lookup hit rate shows 1/N instead of N/N

This means only DCP rank 0 is storing. Apply `patch_dcp_mla_store.py` which fixes the `is_first_rank_of_pp_group` guard to allow first `dcp_size` ranks to store.

### LMCache hit rate is 0%

- Ensure prompts exceed `GLM52_LMCACHE_CHUNK_SIZE` tokens
- Check `http://localhost:8088/status` for `is_healthy: true`
- Verify `chunk_size % (block_size * dcp_size) == 0`

### OOM during CUDA graph capture

Reduce `GPU_MEMORY_UTILIZATION` to `0.90` or `MAX_NUM_BATCHED_TOKENS` to `4096`.

### Boot takes >20 min

Normal for 435GB model. Weight loading (~12 min) + CUDA graph capture (~7 min).

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` crash

**Never** set this with LMCache MP connector. Remove it from environment.

## Image

```
voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618
```

## License

See model license at [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4).
