#!/usr/bin/env bash
set -euo pipefail

OLD_DIR="/home/g0san/glm52-v13-lmcache"
NEW_DIR="/home/g0san/glm52-v12"
OLD_NAME="glm52-v13-lmcache"
NEW_NAME="glm52-v12"
PORT="5318"
MODEL_NAME="GLM-5.2-v13-lmcache"
LOG="/home/g0san/glm52-v12/swap_$(date +%Y%m%d_%H%M%S).log"

exec > >(tee -a "$LOG") 2>&1

log() { echo "[$(date '+%F %T')] $*"; }

rollback() {
  local rc=$?
  log "ERROR: swap failed with rc=${rc}; rolling back to v13"
  set +e
  cd "$NEW_DIR" && docker compose down --timeout 60
  cd "$OLD_DIR" && docker compose up -d
  log "rollback command issued; current containers:"
  docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'glm52|NAMES' || true
  log "rollback log saved at $LOG"
  exit "$rc"
}
trap rollback ERR

log "=== preflight ==="
[[ -d "$OLD_DIR" && -d "$NEW_DIR" ]]
[[ -f "$NEW_DIR/docker-compose.yml" && -f "$NEW_DIR/serve_glm52_v12.sh" && -f "$NEW_DIR/.env" ]]
[[ -f "/mnt/models/GLM-5.2-NVFP4/config.json" ]]
[[ -d "/mnt/lmcache-l2/l2" ]]

docker images --format '{{.Repository}}:{{.Tag}}' | grep -q 'voipmonitor/vllm:glm52-dark-devotion-release-vllmec65667-b12xaaf1891-scale-fix-cu132-20260622'
cd "$NEW_DIR" && docker compose config --quiet

log "v12 critical config:"
cd "$NEW_DIR" && docker compose config | grep -E 'GPU_MEMORY_UTILIZATION:|MAX_MODEL_LEN:|MAX_NUM_SEQS:|DCP_SIZE:|NUM_SPECULATIVE_TOKENS:|PORT:|SERVED_MODEL_NAMES:|GLM52_ENABLE_LMCACHE:' || true

log "current GLM containers:"
docker ps --format 'table {{.Names}}\t{{.Status}}\t{{.Ports}}' | grep -E 'glm52|NAMES' || true

log "=== stopping v13 ==="
cd "$OLD_DIR"
docker compose down --timeout 120

log "waiting for port ${PORT} to free"
for i in $(seq 1 60); do
  if ! ss -tlnp | grep -q ":${PORT} "; then
    break
  fi
  sleep 1
done
if ss -tlnp | grep -q ":${PORT} "; then
  log "port ${PORT} still busy"
  ss -tlnp | grep ":${PORT} " || true
  exit 1
fi

log "=== starting v12 ==="
cd "$NEW_DIR"
docker compose up -d

log "waiting for v12 container to stay running"
for i in $(seq 1 120); do
  status="$(docker inspect -f '{{.State.Status}}' "$NEW_NAME" 2>/dev/null || true)"
  if [[ "$status" == "running" ]]; then
    break
  fi
  if [[ "$status" == "exited" || "$status" == "dead" ]]; then
    log "v12 container status=$status"
    docker logs --tail 200 "$NEW_NAME" || true
    exit 1
  fi
  sleep 1
done
status="$(docker inspect -f '{{.State.Status}}' "$NEW_NAME" 2>/dev/null || true)"
[[ "$status" == "running" ]]

log "waiting for /v1/models on port ${PORT} (max ~9 min)"
ready=0
for i in $(seq 1 108); do
  if out="$(curl -fsS --max-time 5 "http://127.0.0.1:${PORT}/v1/models" 2>/dev/null)"; then
    echo "$out" | python3 -c 'import json,sys; d=json.load(sys.stdin); ids=[m.get("id") for m in d.get("data",[])]; print(ids); assert "GLM-5.2-v13-lmcache" in ids' && ready=1 && break
  fi
  if ! docker ps --format '{{.Names}}' | grep -qx "$NEW_NAME"; then
    log "v12 container disappeared/exited"
    docker logs --tail 250 "$NEW_NAME" || true
    exit 1
  fi
  if (( i % 12 == 0 )); then
    log "still starting... ${i}/108"
    docker logs --tail 20 "$NEW_NAME" || true
  fi
  sleep 5
done
if [[ "$ready" != "1" ]]; then
  log "v12 did not become API-ready in time; tail logs:"
  docker logs --tail 300 "$NEW_NAME" || true
  exit 1
fi

log "smoke test chat completion"
chat_http="$(curl -sS --max-time 120 -o /tmp/glm52_v12_smoke.json -w '%{http_code}' \
  -H 'Content-Type: application/json' \
  -d '{"model":"GLM-5.2-v13-lmcache","messages":[{"role":"user","content":"Reply with OK only."}],"max_tokens":4,"temperature":0}' \
  "http://127.0.0.1:${PORT}/v1/chat/completions")"
log "chat HTTP ${chat_http}"
if [[ "$chat_http" != "200" ]]; then
  cat /tmp/glm52_v12_smoke.json || true
  exit 1
fi
python3 - <<'PY'
import json
p='/tmp/glm52_v12_smoke.json'
d=json.load(open(p))
print('smoke response:', d['choices'][0]['message'].get('content'))
PY

log "GPU snapshot:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits || true

trap - ERR
log "SUCCESS: v12 is live on port ${PORT} as ${MODEL_NAME}; v13 is stopped"
log "swap log saved at $LOG"
