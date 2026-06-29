#!/usr/bin/env bash
# GLM-5.2 v12 Dark Devotion — DCP=4 + MTP3 + LMCache MP connector.
# Based on https://github.com/local-inference-lab/rtx6kpro/blob/master/models/glm5.2_v12.md
# LMCache/DCP patch set adapted from myshytf/glm-5.2-v11-lmcache (same as v13).
set -euo pipefail

unset NCCL_GRAPH_FILE NCCL_GRAPH_DUMP_FILE VLLM_B12X_MLA_EXTEND_MAX_CHUNKS VLLM_PREFIX_CACHE_RETENTION_INTERVAL 2>/dev/null || true

PY=/opt/venv/bin/python
PIP=/opt/venv/bin/pip

# ── LMCache install (one-time per container start) ──────────────────────
LMCACHE_PIN="${GLM52_LMCACHE_VERSION:-0.4.6}"
if ! ${PIP} show lmcache 2>/dev/null | grep -q "^Version: ${LMCACHE_PIN}$"; then
  echo "[glm52-v12] Installing LMCache==${LMCACHE_PIN} (one-time per container start)..."
  ${PIP} install "lmcache==${LMCACHE_PIN}" --quiet 2>&1 | tail -5
  echo "[glm52-v12] LMCache installed"
fi

# ── Apply LMCache/DCP patches (same patch set as v13) ───────────────────
for patch in \
  /opt/patch_kv_xfer_assert_v10.py \
  /opt/patch_fs_l2_adapter.py \
  /opt/patch_l1_evict_flush_to_l2.py \
  /opt/patch_preempt_safe_restore.py \
  /opt/patch_dcp_lmcache.py \
  /opt/patch_dcp_mla_store.py \
  /opt/patch_odirect_aligned.py \
  /opt/patch_ttl_flush_to_l2.py \
  /opt/patch_empty_tools.py; do
  if [[ -f "${patch}" ]]; then
    echo "[glm52-v12] Applying $(basename "${patch}")"
    "${PY}" "${patch}"
  fi
done

# ── Config ──────────────────────────────────────────────────────────────
MODEL="${MODEL:-/models/GLM-5.2-NVFP4}"
SERVED_MODEL_NAMES="${SERVED_MODEL_NAMES:-${SERVED_MODEL_NAME:-GLM-5.2-v13-lmcache}}"
read -r -a SERVED_MODEL_NAME_ARGS <<< "${SERVED_MODEL_NAMES}"
PORT="${PORT:-5318}"
TP_SIZE="${TP_SIZE:-8}"
DCP_SIZE="${DCP_SIZE:-4}"
MTP="${MTP:-1}"
NUM_SPECULATIVE_TOKENS="${NUM_SPECULATIVE_TOKENS:-3}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.94}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-1024000}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-16}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
MAX_CUDAGRAPH_CAPTURE_SIZE="${MAX_CUDAGRAPH_CAPTURE_SIZE:-128}"
KV_CACHE_DTYPE="${KV_CACHE_DTYPE:-fp8}"
ATTENTION_BACKEND="${ATTENTION_BACKEND:-${ATTN_BACKEND:-B12X_MLA_SPARSE}}"
MOE_BACKEND="${MOE_BACKEND:-b12x}"
QUANTIZATION="${QUANTIZATION:-modelopt_fp4}"
GLM52_INDEX_TOPK_PATTERN="${GLM52_INDEX_TOPK_PATTERN:-FFFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSSFSSS}"
HF_OVERRIDES=$(printf '{"use_index_cache":true,"index_topk_pattern":"%s"}' "${GLM52_INDEX_TOPK_PATTERN}")

echo "[glm52-v12] DCP=${DCP_SIZE} TP=${TP_SIZE} MTP=${MTP} spec_tokens=${NUM_SPECULATIVE_TOKENS} max-num-seqs=${MAX_NUM_SEQS} max-model-len=${MAX_MODEL_LEN} gpu-mem=${GPU_MEMORY_UTILIZATION} LMCACHE=${GLM52_ENABLE_LMCACHE:-0}"
echo "[glm52-v12] HF_OVERRIDES=${HF_OVERRIDES}"

# ── vLLM serve args (v12 dark-devotion flags) ───────────────────────────
args=(
  -m vllm.entrypoints.cli.main serve "${MODEL}"
  --served-model-name "${SERVED_MODEL_NAME_ARGS[@]}"
  --host 0.0.0.0
  --port "${PORT}"
  --trust-remote-code
  --tensor-parallel-size "${TP_SIZE}"
  --pipeline-parallel-size 1
  --decode-context-parallel-size "${DCP_SIZE}"
  --dcp-comm-backend ag_rs
  --dcp-kv-cache-interleave-size 1
  --quantization "${QUANTIZATION}"
  --kv-cache-dtype "${KV_CACHE_DTYPE}"
  --attention-backend "${ATTENTION_BACKEND}"
  --moe-backend "${MOE_BACKEND}"
  --load-format fastsafetensors
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --max-model-len "${MAX_MODEL_LEN}"
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
  --max-cudagraph-capture-size "${MAX_CUDAGRAPH_CAPTURE_SIZE}"
  --async-scheduling
  -cc.pass_config.fuse_allreduce_rms=True
  --enable-chunked-prefill
  --enable-prefix-caching
  --enable-auto-tool-choice
  --tool-call-parser glm47
  --reasoning-parser glm45
  --default-chat-template-kwargs '{"reasoning_effort":"max","clear_thinking":false}'
  --hf-overrides "${HF_OVERRIDES}"
)

if [[ -n "${VLLM_API_KEY:-}" ]]; then
  args+=( --api-key "${VLLM_API_KEY}" )
fi

# ── LMCache MP server ───────────────────────────────────────────────────
if [[ "${GLM52_ENABLE_LMCACHE:-0}" == "1" ]]; then
  LMCACHE_MP_HOST="${GLM52_LMCACHE_MP_HOST:-localhost}"
  LMCACHE_MP_PORT="${GLM52_LMCACHE_MP_PORT:-5555}"
  LMCACHE_HTTP_PORT="${GLM52_LMCACHE_HTTP_PORT:-8088}"
  LMCACHE_PROM_PORT="${GLM52_LMCACHE_PROMETHEUS_PORT:-9091}"
  LMCACHE_CHUNK_SIZE="${GLM52_LMCACHE_CHUNK_SIZE:-1024}"
  LMCACHE_L1_GB="${GLM52_LMCACHE_L1_GB:-48}"
  LMCACHE_L1_INIT_GB="${GLM52_LMCACHE_L1_INIT_GB:-8}"
  LMCACHE_L2_GB="${GLM52_LMCACHE_L2_GB:-2800}"
  LMCACHE_DISK_PATH="${GLM52_LMCACHE_DISK_PATH:-/lmcache/l2}"
  LMCACHE_LOG="${GLM52_LMCACHE_LOG:-/tmp/lmcache_mp_server.log}"

  mkdir -p "${LMCACHE_DISK_PATH}"
  echo "[glm52-v12] Starting LMCache MP server: tcp://${LMCACHE_MP_HOST}:${LMCACHE_MP_PORT}, L1=${LMCACHE_L1_GB}GB, L2=${LMCACHE_L2_GB}GB, disk=${LMCACHE_DISK_PATH}, chunk=${LMCACHE_CHUNK_SIZE}"
  rm -f "${LMCACHE_LOG}"
  lmcache server \
    --host "${LMCACHE_MP_HOST}" \
    --port "${LMCACHE_MP_PORT}" \
    --chunk-size "${LMCACHE_CHUNK_SIZE}" \
    --l1-size-gb "${LMCACHE_L1_GB}" \
    --l1-init-size-gb "${LMCACHE_L1_INIT_GB}" \
    --eviction-policy LRU \
    --l2-adapter "{\"type\":\"fs\",\"base_path\":\"${LMCACHE_DISK_PATH}\",\"relative_tmp_dir\":\"tmp\",\"max_capacity_gb\":${LMCACHE_L2_GB},\"use_odirect\":true,\"eviction\":{\"eviction_policy\":\"LRU\",\"trigger_watermark\":0.8,\"eviction_ratio\":0.1}}" \
    --http-port "${LMCACHE_HTTP_PORT}" \
    --prometheus-port "${LMCACHE_PROM_PORT}" \
    >"${LMCACHE_LOG}" 2>&1 &
  LMCACHE_MP_PID=$!
  trap 'kill ${LMCACHE_MP_PID:-} 2>/dev/null || true' EXIT

  _lmcache_ready=0
  for _i in $(seq 1 120); do
    if ! kill -0 "${LMCACHE_MP_PID}" 2>/dev/null; then
      echo "[glm52-v12] LMCache MP server exited during startup; log follows:" >&2
      sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
      exit 1
    fi
    if [[ -f "${LMCACHE_LOG}" ]] && grep -q "ZMQ cache server is running" "${LMCACHE_LOG}"; then
      _lmcache_ready=1
      break
    fi
    sleep 1
  done
  if [[ "${_lmcache_ready}" != "1" ]]; then
    echo "[glm52-v12] LMCache MP server did not become ready; log follows:" >&2
    sed -n '1,220p' "${LMCACHE_LOG}" >&2 || true
    exit 1
  fi
  echo "[glm52-v12] LMCache MP server ready"

  _kv_transfer_config=$(printf '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both","kv_connector_extra_config":{"lmcache.mp.host":"tcp://%s","lmcache.mp.port":%s,"lmcache.mp.mq_timeout":30,"lmcache.mp.heartbeat_interval":5}}' "${LMCACHE_MP_HOST}" "${LMCACHE_MP_PORT}")
  args+=(
    --disable-hybrid-kv-cache-manager
    --kv-transfer-config "${_kv_transfer_config}"
  )
fi

# ── MTP speculative decoding ────────────────────────────────────────────
if [[ "${MTP}" == "1" ]]; then
  SPEC_CONFIG=$(printf '{"model":"%s","method":"mtp","num_speculative_tokens":%s,"moe_backend":"%s","draft_sample_method":"probabilistic"}' "${MODEL}" "${NUM_SPECULATIVE_TOKENS}" "${MOE_BACKEND}")
  args+=( --speculative-config "${SPEC_CONFIG}" )
fi

echo "[glm52-v12] exec: ${PY} -m vllm.entrypoints.cli.main serve ${MODEL} --served-model-name ${SERVED_MODEL_NAMES} --port ${PORT} --tensor-parallel-size ${TP_SIZE} --decode-context-parallel-size ${DCP_SIZE} --kv-transfer-config <lmcache> ${VLLM_API_KEY:+--api-key <redacted>}"
exec "${PY}" "${args[@]}"
