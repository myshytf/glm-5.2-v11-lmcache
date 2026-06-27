# GLM-5.2 v12 Dark Devotion + LMCache Deployment

Production deployment of **GLM-5.2 NVFP4** on vLLM with **LMCache** multi-process connector, B12X MLA sparse attention, DCP=4, MTP speculative decoding, and **O_DIRECT aligned NVMe L2 cache**.

## What's New (2026-06-27)

- **O_DIRECT aligned buffer patch** (`patch_odirect_aligned.py`): posix_memalign-based aligned buffers for O_DIRECT L2 read/write. LMCache 0.4.6's built-in O_DIRECT was broken (Python `bytes` buffer not aligned to block size). This patch uses `posix_memalign` + padding to filesystem block size, enabling `use_odirect=true` with any `chunk_size`.
- **DCP=4 fully working** with LMCache: all 4 DCP ranks store and lookup correctly (100% store success, verified L1/L2 hit).
- **chunk_size=1024** with `use_odirect=true`: O_DIRECT pads I/O to 4096 block size automatically.
- **L2 capacity 2.8TB** on RAID0 NVMe (XFS).
- **MAX_NUM_BATCHED_TOKENS=8192** (was 4096 — 4096 caused decode starvation during prefill).

## Hardware Requirements

- **GPU**: 8× NVIDIA GPU with >=96GB VRAM (tested on RTX PRO 6000 Blackwell)
- **RAM**: >=128GB system memory (48GB allocated to LMCache L1)
- **Storage**: ~435GB for model weights + 2.8TB NVMe for LMCache L2
- **CUDA**: 13.2+
- **Docker** with NVIDIA Container Toolkit

## Quick Start

```bash
# 1. Clone
git clone https://github.com/myshytf/glm-5.2-v11-lmcache.git
cd glm-5.2-v11-lmcache

# 2. Download model (~435GB)
pip install huggingface_hub hf_transfer
HF_HUB_ENABLE_HF_TRANSFER=1 hf download lukealonso/GLM-5.2-NVFP4 --local-dir /models/GLM-5.2-NVFP4

# 3. Configure
cp .env.example .env
# Edit .env: set VLLM_API_KEY, verify paths

# 4. Launch
docker compose up -d

# 5. Wait for boot (~15-20 min)
docker logs -f glm52-v12
# Look for: "Application startup complete"

# 6. Verify
curl http://localhost:5318/health
curl http://localhost:5318/v1/models -H "Authorization: Bearer YOUR_KEY"
```

## Configuration

### Key Parameters

| Parameter | Default | Notes |
|---|---|---|
| `DCP_SIZE` | `4` | DCP=4 requires `patch_dcp_lmcache.py` + `patch_dcp_mla_store.py` |
| `GLM52_LMCACHE_CHUNK_SIZE` | `1024` | Tokens per chunk. O_DIRECT pads to 4096 block size |
| `GLM52_LMCACHE_L1_GB` | `48` | L1 RAM cache |
| `GLM52_LMCACHE_L1_INIT_GB` | `8` | Initial L1 allocation (grows on demand) |
| `GLM52_LMCACHE_L2_GB` | `2800` | L2 NVMe cache capacity |
| `MAX_NUM_BATCHED_TOKENS` | `8192` | Was 4096 — 4096 caused decode starvation |
| `MAX_MODEL_LEN` | `1024000` | 1M context window |
| `ATTN_BACKEND` | `B12X_MLA_SPARSE` | B12X sparse MLA attention |

### O_DIRECT L2 Cache

The `patch_odirect_aligned.py` patch enables `use_odirect=true` in the LMCache FS L2 adapter:

- Uses `posix_memalign` for 4096-aligned memory buffers
- Pads I/O to filesystem block size (non-aligned chunk sizes handled automatically)
- Handles `memoryview`/`bytearray`/`bytes` sources safely
- Verified: 100% store success rate, 0 errors, L2 read/write both working

```json
// L2 adapter config in serve.sh:
{"type":"fs","base_path":"/lmcache/l2","use_odirect":true,
 "eviction":{"eviction_policy":"LRU","trigger_watermark":0.8,"eviction_ratio":0.1}}
```

### DCP=4 with LMCache

DCP shards the KV cache across DCP groups. Two patches enable DCP>1 with LMCache:

1. **`patch_dcp_lmcache.py`** — DCP-aware cache keys, block sizing, `compute_extra_count`
2. **`patch_dcp_mla_store.py`** — Fixes MLA store guard: first `dcp_size` ranks store (one per DCP shard)

**DCP rank layout** (TP=8, DCP=4):
- DCP group 0: ranks [0,1,2,3] — each holds a different KV shard (DCP rank 0-3)
- DCP group 1: ranks [4,5,6,7] — MLA TP shares KV with group 0
- Store: ranks 0-3 (first `dcp_size` ranks)
- Lookup: expands to all 4 DCP ranks via `no_worker_id_version()`

## Architecture

```
┌─────────────────────────────────────────────────────┐
│              Docker Container (glm52-v12)            │
│  ┌─────────────┐    ┌─────────────────────────┐     │
│  │ LMCache MP  │    │   vLLM Engine           │     │
│  │ Server      │◄──►│   (TP=8, DCP=4, MTP=1)  │     │
│  │ port 8088   │    │                         │     │
│  └──────┬──────┘    │  B12X MLA Sparse        │     │
│         │           │  NVFP4 Quantization     │     │
│  ┌──────▼──────┐    └────────────┬────────────┘     │
│  │ L1: 48GB    │                 port 5318          │
│  │ L2: 2.8TB   │                                    │
│  │ (NVMe O_DIR)│                                    │
│  └─────────────┘                                    │
└─────────────────────────────────────────────────────┘
```

## Included Patches

Applied in order by `serve.sh` during container startup:

### Stability Patches (always applied)

| Patch | Purpose |
|---|---|
| `patch_kv_xfer_assert_v10.py` | Guards against LMCache duplicate callback crash |
| `patch_fs_l2_adapter.py` | L2 adapter: capacity limit, byte accounting, eviction, startup scan |
| `patch_l1_evict_flush_to_l2.py` | L1 eviction flushes to L2 via `store_objects_sync()` before discard |
| `patch_preempt_safe_restore.py` | Async-aware preempted request KV restore from LMCache |

### DCP Patches (required for DCP>1)

| Patch | Purpose |
|---|---|
| `patch_dcp_lmcache.py` | DCP-aware cache keys, block sizing, `compute_extra_count` |
| `patch_dcp_mla_store.py` | Fixes MLA store guard for DCP (first `dcp_size` ranks store) |

### O_DIRECT Patch

| Patch | Purpose |
|---|---|
| `patch_odirect_aligned.py` | `posix_memalign`-based aligned buffers for O_DIRECT L2 read/write |

## Monitoring

```bash
# LMCache status (L1/L2 objects, memory, prefetch stats)
curl http://localhost:8088/status | python3 -m json.tool

# LMCache metrics
curl -s http://localhost:8088/metrics | grep -E 'l1_read|l1_write|l2_store|l2_prefetch'

# L2 O_DIRECT errors (should be 0)
docker exec glm52-v12 bash -lc 'grep -ciE "O_DIRECT|Invalid argument|Failed" /tmp/lmcache_mp_server.log'

# Prefetch hits (should show N/N, not 1/N)
docker exec glm52-v12 tail -20 /tmp/lmcache_mp_server.log | grep "Prefetch.*completed"

# vLLM metrics
curl -s http://localhost:5318/metrics | grep -E 'prompt_tokens|external_prefix|gpu_prefix'
```

### Cache Hit Rate Verification

```bash
# After container restart (L1 cleared, L2 persisted):
# 1. Send a long-prefix request
# 2. Check L2 prefetch hits:
docker exec glm52-v12 tail -5 /tmp/lmcache_mp_server.log | grep "Prefetch.*completed"
# Good: "40/40 prefix hits (0 L1, 40 L2)" — L2 O_DIRECT read working
```

## Troubleshooting

### O_DIRECT errors (Invalid argument)

If `use_odirect=true` without `patch_odirect_aligned.py`, LMCache 0.4.6's built-in O_DIRECT fails because Python `bytes` buffers are not memory-aligned. Apply the patch and ensure it's mounted in `docker-compose.yml`.

### Lookup hit rate shows 1/N instead of N/N

Only DCP rank 0 is storing. Apply `patch_dcp_mla_store.py`.

### LMCache hit rate is 0%

- Ensure prompts exceed `GLM52_LMCACHE_CHUNK_SIZE` tokens
- Check `http://localhost:8088/status` for `is_healthy: true`
- Verify `chunk_size % (block_size * dcp_size) == 0`

### Decode starvation (1-2 tok/s during prefill)

Increase `MAX_NUM_BATCHED_TOKENS` from 4096 to 8192. 4096 causes the scheduler to prioritize prefill over decode.

### OOM during CUDA graph capture

Reduce `GPU_MEMORY_UTILIZATION` to `0.90` or `MAX_CUDAGRAPH_CAPTURE_SIZE` to `64`.

## Docker Image

```
voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622
```

## License

See model license at [lukealonso/GLM-5.2-NVFP4](https://huggingface.co/lukealonso/GLM-5.2-NVFP4).
