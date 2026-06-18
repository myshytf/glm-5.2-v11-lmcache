# GLM-5.2 v11 + LMCache Deployment

Production deployment of **GLM-5.2 NVFP4** on vLLM with **LMCache** multi-process connector, B12X MLA sparse attention, and MTP speculative decoding.

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
curl http://localhost:5318/v1/models -H "Authorization: Bearer YOUR_API_KEY"
```

## Configuration

### Key Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DCP_SIZE` | `1` | Decode Context Parallel size. `1` = stable with LMCache. `4` = experimental (requires DCP patches) |
| `NUM_SPECULATIVE_TOKENS` | `2` | MTP speculative decode tokens |
| `MAX_MODEL_LEN` | `450000` | Maximum context length |
| `MAX_NUM_SEQS` | `16` | Max concurrent sequences |
| `GPU_MEMORY_UTILIZATION` | `0.94` | GPU memory utilization |
| `GLM52_LMCACHE_CHUNK_SIZE` | `256` | LMCache chunk size (tokens). Use `512+` for DCP>1 |
| `GLM52_LMCACHE_L1_GB` | `48` | LMCache L1 (CPU RAM) size |
| `GLM52_LMCACHE_L2_GB` | `500` | LMCache L2 (NVMe) size |

### DCP > 1 (Experimental)

DCP shards KV cache across GPUs, reducing per-GPU memory usage. The included patches enable DCP>1 with LMCache:

1. `patch_dcp_lmcache.py` — DCP-aware cache keys, block sizing, and extra_count
2. `patch_dcp_combine.py` — B12X MLA sparse attention DCP LSE combine (fixes output quality)

```bash
# In .env:
DCP_SIZE=4
GLM52_LMCACHE_CHUNK_SIZE=512   # must be >= block_size * dcp_size (128 * 4 = 512)
```

**Note**: DCP>1 is experimental. Tool call quality may be affected. DCP=1 with LMCache is the stable production configuration.

## Architecture

```
┌─────────────────────────────────────────────┐
│              Docker Container                │
│  ┌─────────────┐    ┌─────────────────────┐ │
│  │ LMCache MP  │    │   vLLM Engine       │ │
│  │ Server      │◄──►│   (TP=8, DCP=1)     │ │
│  │ port 5555   │    │                     │ │
│  └──────┬──────┘    │  B12X MLA Sparse    │ │
│         │           │  MTP Spec Decode    │ │
│  ┌──────▼──────┐    │  NVFP4 Quantization │ │
│  │ L1: 48GB    │    └─────────────────────┘ │
│  │ L2: 500GB   │    port 5318               │
│  │ (NVMe)      │                            │
│  └─────────────┘                            │
└─────────────────────────────────────────────┘
```

## Included Patches

| Patch | Purpose |
|-------|---------|
| `patch_kv_xfer_assert` | Guards against LMCache duplicate callback crash |
| `patch_fs_l2_adapter` | Fixes L2 adapter: capacity limit, byte accounting, eviction, startup scan |
| `patch_l1_evict_flush_to_l2` | L1 eviction flushes to L2 before discard |
| `patch_preempt_safe_restore` | Async-aware preempted request KV restore from LMCache |
| `patch_dcp_lmcache` | **(DCP>1)** DCP-aware cache keys, block sizing, extra_count |
| `patch_dcp_combine` | **(DCP>1)** B12X MLA sparse attention LSE combine across DCP ranks |

## Monitoring

```bash
# LMCache status
curl http://localhost:8088/status | python3 -m json.tool

# LMCache metrics (hit rate, L1/L2 usage)
curl http://localhost:8088/metrics | grep lmcache_mp_lookup

# vLLM logs (prefix cache hit, KV usage)
docker logs glm52-v11 2>&1 | grep "External prefix cache hit"
```

## Troubleshooting

### OOM during CUDA graph capture
Reduce `GPU_MEMORY_UTILIZATION` to `0.90` or `MAX_NUM_BATCHED_TOKENS` to `4096`.

### LMCache hit rate is 0%
- Ensure prompts exceed `GLM52_LMCACHE_CHUNK_SIZE` tokens
- Check `http://localhost:8088/status` for `is_healthy: true`
- Verify `PYTHONHASHSEED=0` in `.env`

### Boot takes >20 min
Normal for 435GB model. Weight loading (~12 min) + CUDA graph capture (~7 min) + LMCache pip install (~30s).

### `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` crash
**Never** set this with LMCache MP connector. Remove it from environment.

## Image

```
voipmonitor/vllm:glm52-v11-darkdevotion-vllma86f74e-b12x5b2e018-cu132-20260618
```

## License

See model license at [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4).
