#!/usr/bin/env bash
# EXL3 TP=2 PRODUCTION-CANDIDATE launcher (Zeus cuda-exl3 on the official vLLM GLM-5.3 image).
# One rank per node. Start the worker FIRST and the head LAST:
#   ./start-tp2full.sh 1   on worker-1
#   ./start-tp2full.sh 0   on the head node (rank 0, serves the API)
# Tear BOTH ranks down before relaunching either of them.
#
# This is ~/exl3-zeus/start-tp2.sh (the bring-up launcher) plus the four pieces
# of the TP=3 production launcher that are not about rank count:
#   1. the F1 settle gate  -- wait for the kernel to reclaim the previous
#      container's pages before vLLM snapshots MemAvailable, or the rank awards
#      itself KV memory it does not have (docs/07 sec.1.1).
#   2. the fastload sidecar block (FASTLOAD_MODE=dump|load, identity mount).
#   3. PROF_ARG -- this vLLM takes the torch profiler as --profiler-config, not
#      VLLM_TORCH_PROFILER_DIR; without the flag /start_profile answers 404.
#   4. NODE_RANK / TP_SIZE / ENABLE_EP / TP3_DIR into the container. The
#      bring-up launcher passed none of them, which is why its prelude printed
#      "rank=? tp=?" -- and the fastload sidecar cannot work without NODE_RANK.
#
# TP3_DIR is deliberately set to the SAME directory as TP2_DIR: preflight-fastload.py
# and harem_fastload.py both read TP3_DIR to find the patch set they must hash.
# Pointing it at the tp2full tree is what makes the sidecar identity cover the
# two-node patch set instead of a three-node one that is not mounted.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$HOME/exl3-zeus/.env.tp2-full}"
test -f "$ENV_FILE" || { echo "ENV_FILE not found: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
NODE_RANK="${1:-${NODE_RANK:?set NODE_RANK or pass rank}}"
IMAGE="${IMAGE:-exl3-zeus:754421f}"
NAME="${NAME:-exl3-tp2}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-tr3-4bpw}"
MODEL_PATH="${MODEL_PATH:-/models/glm-5.3-flash-tr3-4bpw}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-/var/tmp/dflash2-draft-tp2}"
DRAFT_PATH="${DRAFT_PATH:-/models/dflash2-draft}"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-/var/tmp/exl3-zeus-cache}"
PRELUDE_DIR="${PRELUDE_DIR:-$ROOT}"
PORT="${PORT:-8001}"
NNODES="${NNODES:-2}"
TP_SIZE="${TP_SIZE:-2}"
MASTER_ADDR="${MASTER_ADDR:?}"
MASTER_PORT="${MASTER_PORT:-29531}"
HOST_IP="${HOST_IP:?}"
GLOO_IFACE="${GLOO_IFACE:-enP7s7}"
TRANSPORT="${TRANSPORT:-mesh}"
NCCL_MESH_PLUGIN_DIR="${NCCL_MESH_PLUGIN_DIR:-$HOME/nccl-mesh}"
SPEC_METHOD="${SPEC_METHOD:-none}"      # none | mtp | dflash
SPEC_TOKENS="${SPEC_TOKENS:-1}"
[ "$TP_SIZE" = "2" ] || { echo "start-tp2full.sh is the two-rank launcher (TP_SIZE=$TP_SIZE)" >&2; exit 2; }
test -f "$MODEL_HOST_PATH/config.json" || { echo "no model at $MODEL_HOST_PATH" >&2; exit 2; }
mkdir -p "$CACHE_HOST_PATH/triton" "$CACHE_HOST_PATH/tilelang" "$CACHE_HOST_PATH/flashinfer" "$CACHE_HOST_PATH/tune"
# MASTER_ADDR must be the MANAGEMENT address of rank 0. A fabric (RoCE) address
# hangs the rendezvous silently instead of failing. Set FABRIC_PREFIX to the first
# octets of your fabric subnet and this refuses one outright.
FABRIC_PREFIX="${FABRIC_PREFIX:-}"
if [ -n "$FABRIC_PREFIX" ]; then
  case "$MASTER_ADDR" in "$FABRIC_PREFIX"*) echo "MASTER_ADDR is a fabric address; use the management IP" >&2; exit 2 ;; esac
fi
HEADLESS=""; [ "$NODE_RANK" != "0" ] && HEADLESS="--headless"
SPEC_ARG=(); DRAFT_MOUNT=()
case "$SPEC_METHOD" in
  none) ;;
  mtp) SPEC_ARG=(--speculative-config "{\"method\":\"${MTP_METHOD:-glm5_next_mtp}\",\"num_speculative_tokens\":${SPEC_TOKENS}}") ;;
  dflash) test -f "$DRAFT_HOST_PATH/config.json" || { echo "no drafter at $DRAFT_HOST_PATH" >&2; exit 2; }
          DRAFT_MOUNT=(-v "$DRAFT_HOST_PATH:$DRAFT_PATH:ro")
          SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_PATH}\",\"num_speculative_tokens\":${SPEC_TOKENS},\"kv_cache_dtype\":\"auto\"}") ;;
  *) echo "SPEC_METHOD must be none|mtp|dflash" >&2; exit 2 ;;
esac
# clear_thinking:true — prior turns' <think> blocks are NOT re-rendered into later prompts. The upstream template
# defaults to retaining them since its 27 Aug 2026 revision (docs/14 §9.12). enable_thinking is not read by the
# template and is no longer sent. reasoning_effort is appended when REASONING_EFFORT is set.
THINKING_ARG=(--default-chat-template-kwargs "{\"clear_thinking\":true${REASONING_EFFORT:+,\"reasoning_effort\":\"${REASONING_EFFORT}\"}}")
PROF_ARG=()
if [ -n "${PROFILER_DIR:-}" ]; then
  PROF_ARG=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}")
fi
# Expert parallelism is OPTIONAL at two ranks: 2048 / 2 = 1024 = 8 x 128, so the
# routed-expert trellis slices cleanly. preflight-tp3.py owns the arithmetic and
# will refuse --ep 0 at tp=3 while accepting it here.
EP_ARG=(); [ "${ENABLE_EP:-0}" = "1" ] && EP_ARG=(--enable-expert-parallel)
EAGER_ARG=(); [ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_ARG=(--enforce-eager)
KV_MEM_ARG=(); [ -n "${KV_CACHE_MEMORY:-}" ] && KV_MEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")
CT_ARG=(); [ -n "${CHAT_TEMPLATE_HOST:-}" ] && { CT_ARG=(--chat-template /models/chat_template.jinja); CT_MOUNT=(-v "$CHAT_TEMPLATE_HOST:/models/chat_template.jinja:ro"); } || CT_MOUNT=()
MM_ARG=(); if [ "${LANGUAGE_MODEL_ONLY:-1}" = "1" ]; then MM_ARG=(--language-model-only); else MM_ARG=(--skip-mm-profiling --limit-mm-per-prompt '{"image": 4, "video": 1}'); fi
EXTRA_ENV_ARG=(); for _kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARG+=(-e "$_kv"); done

# --- HAREM fastload sidecar (tp2full/harem_fastload.py) ----------------------
# dump : normal load from the checkpoint, then write this rank's post-load
#        tensors to $FASTLOAD_DIR-r$NODE_RANK (rw mount)
# load : restore from that directory instead of re-slicing the checkpoint (ro)
# unset: upstream behaviour, nothing is mounted.
# At tp=2 the sidecar is about half the checkpoint per rank (EP off, so every
# rank owns half of every tensor) against a third at tp=3 with EP -- check
# `df -h /var/tmp` before the first dump.
FASTLOAD_MOUNT=()
if [ -n "${FASTLOAD_MODE:-}" ]; then
  [ -n "${FASTLOAD_DIR:-}" ] || { echo "FASTLOAD_MODE set but FASTLOAD_DIR empty" >&2; exit 2; }
  _FL="${FASTLOAD_DIR}-r${NODE_RANK}"
  case "$FASTLOAD_MODE" in
    dump) mkdir -p "$_FL"; FASTLOAD_MOUNT=(-v "$_FL:$_FL") ;;
    load) test -f "$_FL/MANIFEST.json" || { echo "no fastload sidecar at $_FL (run FASTLOAD_MODE=dump first)" >&2; exit 2; }
          FASTLOAD_MOUNT=(-v "$_FL:$_FL:ro") ;;
    *) echo "FASTLOAD_MODE must be dump|load (got $FASTLOAD_MODE)" >&2; exit 2 ;;
  esac
  EXTRA_ENV_ARG+=(-e "HAREM_FASTLOAD_MODE=$FASTLOAD_MODE" -e "HAREM_FASTLOAD_DIR=$FASTLOAD_DIR"
                  -e "HAREM_FASTLOAD_VERIFY=${FASTLOAD_VERIFY:-64}")
  [ -n "${FASTLOAD_POSTHASH:-}" ] && EXTRA_ENV_ARG+=(-e "HAREM_FASTLOAD_POSTHASH=$FASTLOAD_POSTHASH")
  [ -n "${FASTLOAD_POSTHASH_N:-}" ] && EXTRA_ENV_ARG+=(-e "HAREM_FASTLOAD_POSTHASH_N=$FASTLOAD_POSTHASH_N")
  [ -n "${FASTLOAD_SHARD_BYTES:-}" ] && EXTRA_ENV_ARG+=(-e "HAREM_FASTLOAD_SHARD_BYTES=$FASTLOAD_SHARD_BYTES")
  [ -n "${FASTLOAD_READ:-}" ] && EXTRA_ENV_ARG+=(-e "HAREM_FASTLOAD_READ=$FASTLOAD_READ")
  echo "fastload: mode=$FASTLOAD_MODE dir=$_FL verify=${FASTLOAD_VERIFY:-64}"
fi
# The sidecar identity records which image and which patch tree produced it.
EXTRA_ENV_ARG+=(-e "HAREM_IMAGE_TAG=$IMAGE"
                -e "HAREM_FASTLOAD_MODEL_PATH=$MODEL_PATH"
                -e "HAREM_FASTLOAD_DRAFT_PATH=$DRAFT_PATH")
NCCL_ENV=( -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IGNORE_CPU_AFFINITY=1
  -e NCCL_DEBUG="${NCCL_DEBUG:-WARN}" -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" -e TP_SOCKET_IFNAME="$GLOO_IFACE" -e MN_IF_NAME="$GLOO_IFACE" )
OVERLAY_MOUNT=()
if [ -n "${OVERLAY_DIR:-}" ]; then
  VP=/usr/local/lib/python3.12/dist-packages/vllm
  for f in sparse_attn_indexer_kpool.py sparse_attn_indexer.py; do
    [ -f "$OVERLAY_DIR/$f" ] && OVERLAY_MOUNT+=(-v "$OVERLAY_DIR/$f:$VP/model_executor/layers/$f:ro") && echo "overlay: $f"
  done
fi
PLUGIN_MOUNT=()
case "$TRANSPORT" in
  mesh) test -f "$NCCL_MESH_PLUGIN_DIR/libnccl-net-mesh.so" || { echo "no mesh plugin in $NCCL_MESH_PLUGIN_DIR" >&2; exit 2; }
        PLUGIN_MOUNT=(-v "$NCCL_MESH_PLUGIN_DIR:/opt/nccl-mesh:ro")
        NCCL_ENV+=( -e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="=${GLOO_IFACE}" -e NCCL_NET_PLUGIN=mesh -e NCCL_ALGO=Ring -e NCCL_MESH_DEBUG="${NCCL_MESH_DEBUG:-1}" -e LD_LIBRARY_PATH=/opt/nccl-mesh ) ;;
  socket) NCCL_ENV+=(-e NCCL_NET=Socket -e NCCL_IB_DISABLE=1 -e NCCL_SOCKET_IFNAME="$GLOO_IFACE") ;;
  *) echo "TRANSPORT must be mesh|socket" >&2; exit 2 ;;
esac
test -f "$PRELUDE_DIR/tp2-prelude.sh" || { echo "no tp2-prelude.sh in $PRELUDE_DIR" >&2; exit 2; }
test -f "$PRELUDE_DIR/tp3-prelude.sh" || { echo "no tp3-prelude.sh hard link in $PRELUDE_DIR (the fastload identity hashes the prelude under that name)" >&2; exit 2; }
PRELUDE_MOUNT=(-v "$PRELUDE_DIR:/opt/harem-tp2:ro" -v "$PRELUDE_DIR/tp2-prelude.sh:/start.sh:ro")
echo "prelude: $PRELUDE_DIR  fullscope=${HAREM_EXL3_FULLSCOPE:-unset}"
echo "IMAGE=$IMAGE rank=$NODE_RANK nnodes=$NNODES tp=$TP_SIZE ep=${ENABLE_EP:-0} spec=$SPEC_METHOD/$SPEC_TOKENS kv=${KV_CACHE_DTYPE:-fp8} mem=${GPU_MEMORY_UTILIZATION:-0.85} transport=$TRANSPORT"
DOCKER_ARGS=(run --gpus all -d --log-opt max-size=20m --log-opt max-file=3 --name "$NAME" --restart no
  --network host --ipc host --shm-size 32g --cpuset-cpus "${CPUSET:-5-9,15-19}" --ulimit memlock=-1:-1 --cap-add IPC_LOCK
  --device /dev/infiniband:/dev/infiniband
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" "${DRAFT_MOUNT[@]}" "${CT_MOUNT[@]}"
  -v "$CACHE_HOST_PATH:/cache" -v "$CACHE_HOST_PATH/triton:/root/.triton" -v "$CACHE_HOST_PATH/tilelang:/root/.tilelang" -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer"
  "${PLUGIN_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${PRELUDE_MOUNT[@]}" "${FASTLOAD_MOUNT[@]}"
  -e VLLM_HOST_IP="$HOST_IP" -e VLLM_CACHE_ROOT=/cache -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=12.1 -e FLASHINFER_CUDA_ARCH_LIST=12.1 -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e CUDA_EXL3_MLA_TUNE_VERBOSE="${CUDA_EXL3_MLA_TUNE_VERBOSE:-1}" -e CUDA_EXL3_DEBUG_NAMES="${CUDA_EXL3_DEBUG_NAMES:-0}"
  ${NCCL_PROTO:+-e NCCL_PROTO="$NCCL_PROTO"} "${EXTRA_ENV_ARG[@]}" "${NCCL_ENV[@]}"
  -e NODE_RANK="$NODE_RANK" -e TP_SIZE="$TP_SIZE" -e ENABLE_EP="${ENABLE_EP:-0}"
  -e TP2_DIR=/opt/harem-tp2 -e TP3_DIR=/opt/harem-tp2 -e TP2_STRICT="${TP2_STRICT:-1}"
  --entrypoint bash "$IMAGE" /start.sh "$MODEL_PATH"
  --served-model-name "${SERVED_MODEL_NAME:-glm-5.3-flash-exl3}" --host 0.0.0.0 --port "$PORT" --trust-remote-code
  --quantization "${QUANTIZATION:-exl3}" --attention-backend "${ATTENTION_BACKEND:-CUSTOM}"
  --tensor-parallel-size "$TP_SIZE" --pipeline-parallel-size 1
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION:-0.85}" --max-model-len "${MAX_MODEL_LEN:-1000000}"
  --max-num-seqs "${MAX_NUM_SEQS:-8}" --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}"
  --kv-cache-dtype "${KV_CACHE_DTYPE:-fp8}" --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16
  --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser "${REASONING_PARSER:-deepseek_r1}"
  --distributed-executor-backend mp --nnodes "$NNODES" --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
  "${KV_MEM_ARG[@]}" "${SPEC_ARG[@]}" "${EP_ARG[@]}" "${EAGER_ARG[@]}" "${CT_ARG[@]}" "${MM_ARG[@]}" "${THINKING_ARG[@]}" "${PROF_ARG[@]}" )
[ -n "$HEADLESS" ] && DOCKER_ARGS+=("$HEADLESS")
[ -n "${EXTRA_ARGS:-}" ] && DOCKER_ARGS+=(${EXTRA_ARGS})
if [ "${DRY_RUN:-0}" = "1" ]; then printf 'docker'; printf ' %q' "${DOCKER_ARGS[@]}"; printf '\n'; exit 0; fi
docker rm -f "$NAME" 2>/dev/null || true
# HAREM F1 settle gate: let the kernel reclaim the previous container before vLLM
# snapshots MemAvailable (the KV pool is computed from that snapshot -- docs/07 sec.1.1).
for _i in $(seq 1 60); do _ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [ "$_ma" -ge "${SETTLE_MIN_GIB:-112}" ] && break; sleep 3; done; echo "settle: MemAvailable=${_ma} GiB after ${_i} polls"
docker "${DOCKER_ARGS[@]}"
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"; sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; docker logs $NAME" >&2; exit 1; }
