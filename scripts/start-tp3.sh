#!/usr/bin/env bash
# EXL3 TP=3 + expert-parallel launcher (Zeus cuda-exl3 on the official vLLM GLM-5.3 image).
# One rank per node. Start the workers FIRST and the head LAST:
#   ./start-tp3.sh 2   on worker-2
#   ./start-tp3.sh 1   on worker-1
#   ./start-tp3.sh 0   on the head node (rank 0, serves the API)
# Tear ALL ranks down before relaunching any of them.
# The fabric and NCCL handling is shared with our NVFP4 recipe's launcher, unchanged.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
ENV_FILE="${ENV_FILE:-$ROOT/.env.tp3}"
test -f "$ENV_FILE" || { echo "ENV_FILE not found: $ENV_FILE" >&2; exit 2; }
source "$ENV_FILE"
NODE_RANK="${1:-${NODE_RANK:?set NODE_RANK or pass rank}}"
IMAGE="${IMAGE:-exl3-zeus:serve}"
NAME="${NAME:-exl3-tp3}"
MODEL_HOST_PATH="${MODEL_HOST_PATH:-/var/tmp/glm-5.3-flash-tr3-4bpw-tp3}"   # sidecar (padded config + links)
MODEL_LINK_TARGET="${MODEL_LINK_TARGET:-/var/tmp/glm-5.3-flash-tr3-4bpw}"   # link targets must be mounted at the SAME path inside
TP3_DIR="${TP3_DIR:-$HOME/exl3-zeus/tp3}"
# IDENTITY MOUNTS. pad-tp3.py's sidecars are directories of RELATIVE symlinks
# ("../glm-5.3-flash-tr3-4bpw/model-00001-of-00120.safetensors") into their link
# target, so a sidecar only resolves while it keeps its position relative to that
# target. Mounting it anywhere else -- /models/... was the original default --
# makes "../glm-5.3-flash-tr3-4bpw" point at the sidecar itself, every weight link
# dangles, and the failure surfaces as "no safetensors found", not as a mount
# error. Mounting host path -> same container path is the one arrangement that is
# correct by construction. check_relative_sidecar() below enforces it.
MODEL_PATH="${MODEL_PATH:-$MODEL_HOST_PATH}"
DRAFT_HOST_PATH="${DRAFT_HOST_PATH:-/var/tmp/dflash2-draft-tp3}"   # 36/9 sidecar
DRAFT_LINK_TARGET="${DRAFT_LINK_TARGET:-/var/tmp/dflash2-draft-tp2}"
DRAFT_PATH="${DRAFT_PATH:-$DRAFT_HOST_PATH}"
CACHE_HOST_PATH="${CACHE_HOST_PATH:-/var/tmp/exl3-zeus-cache}"
PORT="${PORT:-8001}"
NNODES="${NNODES:-3}"
TP_SIZE="${TP_SIZE:-3}"
MASTER_ADDR="${MASTER_ADDR:?}"
MASTER_PORT="${MASTER_PORT:-29531}"
HOST_IP="${HOST_IP:?}"
GLOO_IFACE="${GLOO_IFACE:-enP7s7}"
TRANSPORT="${TRANSPORT:-mesh}"
NCCL_MESH_PLUGIN_DIR="${NCCL_MESH_PLUGIN_DIR:-$HOME/nccl-mesh}"   # read-only mount of the production plugin
SPEC_METHOD="${SPEC_METHOD:-none}"      # none | mtp | dflash
SPEC_TOKENS="${SPEC_TOKENS:-1}"
test -f "$MODEL_HOST_PATH/config.json" || { echo "no model at $MODEL_HOST_PATH" >&2; exit 2; }
# Fail closed on a sidecar that would be mounted away from its link target.
check_relative_sidecar() {   # $1 host dir, $2 container path, $3 label
  local host="$1" ctr="$2" label="$3" n
  n="$(find "$host" -maxdepth 1 -type l -printf '%l\n' 2>/dev/null | grep -c '^\.\.' || true)"
  [ "${n:-0}" -eq 0 ] && return 0
  [ "$host" = "$ctr" ] && return 0
  echo "$label: $host holds $n relative symlink(s) into its link target, so it must" >&2
  echo "  be mounted at the SAME path inside the container -- but the container path" >&2
  echo "  is $ctr. Every weight link would dangle. Unset ${label}_PATH (it now" >&2
  echo "  defaults to the host path) or set it to $host." >&2
  exit 2
}
check_relative_sidecar "$MODEL_HOST_PATH" "$MODEL_PATH" MODEL
# The link target must itself be mounted, at its own path, or the links dangle too.
test -d "$MODEL_LINK_TARGET" || { echo "no link target at $MODEL_LINK_TARGET" >&2; exit 2; }
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
          check_relative_sidecar "$DRAFT_HOST_PATH" "$DRAFT_PATH" DRAFT
          test -d "$DRAFT_LINK_TARGET" || { echo "no draft link target at $DRAFT_LINK_TARGET" >&2; exit 2; }
          DRAFT_MOUNT=(-v "$DRAFT_HOST_PATH:$DRAFT_PATH:ro" -v "$DRAFT_LINK_TARGET:$DRAFT_LINK_TARGET:ro")
          SPEC_ARG=(--speculative-config "{\"method\":\"dflash\",\"model\":\"${DRAFT_PATH}\",\"num_speculative_tokens\":${SPEC_TOKENS},\"kv_cache_dtype\":\"auto\"}") ;;
  *) echo "SPEC_METHOD must be none|mtp|dflash" >&2; exit 2 ;;
esac
# clear_thinking decides whether every PRIOR turn's <think> block is re-rendered
# into the prompt. z.ai's template defaults it to false -- RETAIN -- from
# 04c4e9e9 (27 Aug) onward, 690b7052 included, and the retention fires without a
# client asking for it: agent clients echo the previous assistant turn back
# verbatim and the template extracts <think> out of `content` itself, so the
# block returns even when the request never set reasoning_content. It compounds
# every turn. CLEAR_THINKING=0 restores the template default.
# Protects only requests that send NO chat_template_kwargs of their own: a
# per-request block REPLACES these defaults rather than merging, so a gateway
# must send the key on every variant. docs/14.
if [ "${CLEAR_THINKING:-1}" = "1" ]; then _CT=true; else _CT=false; fi
THINKING_ARG=(); [ -n "${REASONING_EFFORT:-}" ] && THINKING_ARG=(--default-chat-template-kwargs "{\"enable_thinking\":true,\"clear_thinking\":${_CT},\"reasoning_effort\":\"${REASONING_EFFORT}\"}")
[ "${ENABLE_EP:-1}" = "1" ] || { echo "TP=3 requires ENABLE_EP=1 (EXL3 trellis cannot be sliced by 3)" >&2; exit 2; }
EP_ARG=(--enable-expert-parallel)
EAGER_ARG=(); [ "${ENFORCE_EAGER:-0}" = "1" ] && EAGER_ARG=(--enforce-eager)
KV_MEM_ARG=(); [ -n "${KV_CACHE_MEMORY:-}" ] && KV_MEM_ARG=(--kv-cache-memory "$KV_CACHE_MEMORY")
CT_ARG=(); [ -n "${CHAT_TEMPLATE_HOST:-}" ] && { CT_ARG=(--chat-template /models/chat_template.jinja); CT_MOUNT=(-v "$CHAT_TEMPLATE_HOST:/models/chat_template.jinja:ro"); } || CT_MOUNT=()
MM_ARG=(); if [ "${LANGUAGE_MODEL_ONLY:-1}" = "1" ]; then MM_ARG=(--language-model-only); else MM_ARG=(--skip-mm-profiling --limit-mm-per-prompt '{"image": 4, "video": 1}'); fi
EXTRA_ENV_ARG=(); for _kv in ${EXTRA_ENV:-}; do EXTRA_ENV_ARG+=(-e "$_kv"); done

# Torch profiler, off unless PROFILER_DIR is set. This vLLM takes it as
# --profiler-config and NOT as VLLM_TORCH_PROFILER_DIR (vllm/config/profiler.py):
# with the environment variable alone the /start_profile route is never attached
# and the endpoint answers 404, which is exactly how a step-time breakdown gets
# postponed for a week. Set it before you need it -- an unset PROFILER_DIR costs
# nothing at run time. docs/09 section 4.1.
PROF_ARG=()
if [ -n "${PROFILER_DIR:-}" ]; then
  PROF_ARG=(--profiler-config "{\"profiler\":\"torch\",\"torch_profiler_dir\":\"${PROFILER_DIR}\",\"torch_profiler_with_stack\":false,\"ignore_frontend\":true}")
fi

# --- HAREM fastload sidecar (tp3/harem_fastload.py) -------------------------
# FASTLOAD_MODE=dump  : normal load from the checkpoint, then write this rank's
#                       post-load tensors to $FASTLOAD_DIR-r$NODE_RANK (rw mount)
# FASTLOAD_MODE=load  : restore from that directory instead of re-slicing the
#                       full checkpoint (ro mount; refuses if it is not there)
# unset               : upstream behaviour, nothing is mounted.
# The directory is identity-mounted, like the model sidecars, so a path printed
# in the container is the same path on the host.
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
# The sidecar identity records which image produced it.
EXTRA_ENV_ARG+=(-e "HAREM_IMAGE_TAG=$IMAGE" -e "TP3_DIR=/opt/harem-tp3"
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
echo "IMAGE=$IMAGE rank=$NODE_RANK nnodes=$NNODES tp=$TP_SIZE ep=${ENABLE_EP:-0} spec=$SPEC_METHOD/$SPEC_TOKENS kv=${KV_CACHE_DTYPE:-fp8} mem=${GPU_MEMORY_UTILIZATION:-0.85} transport=$TRANSPORT"
DOCKER_ARGS=(run --gpus all -d --log-opt max-size=20m --log-opt max-file=3 --name "$NAME" --restart no
  --network host --ipc host --shm-size 32g --cpuset-cpus "${CPUSET:-5-9,15-19}" --ulimit memlock=-1:-1 --cap-add IPC_LOCK
  --device /dev/infiniband:/dev/infiniband
  -v "$MODEL_HOST_PATH:$MODEL_PATH:ro" -v "$MODEL_LINK_TARGET:$MODEL_LINK_TARGET:ro" -v "$TP3_DIR:/opt/harem-tp3:ro" -v "$TP3_DIR/tp3-prelude.sh:/start.sh:ro" "${DRAFT_MOUNT[@]}" "${CT_MOUNT[@]}"
  -v "$CACHE_HOST_PATH:/cache" -v "$CACHE_HOST_PATH/triton:/root/.triton" -v "$CACHE_HOST_PATH/tilelang:/root/.tilelang" -v "$CACHE_HOST_PATH/flashinfer:/root/.cache/flashinfer"
  "${PLUGIN_MOUNT[@]}" "${OVERLAY_MOUNT[@]}" "${FASTLOAD_MOUNT[@]}"
  -e VLLM_HOST_IP="$HOST_IP" -e VLLM_CACHE_ROOT=/cache -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
  -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-1800}"
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=12.1 -e FLASHINFER_CUDA_ARCH_LIST=12.1 -e FLASHINFER_DISABLE_VERSION_CHECK=1
  -e CUDA_EXL3_MLA_TUNE_VERBOSE="${CUDA_EXL3_MLA_TUNE_VERBOSE:-1}" -e CUDA_EXL3_DEBUG_NAMES="${CUDA_EXL3_DEBUG_NAMES:-0}"
  ${NCCL_PROTO:+-e NCCL_PROTO="$NCCL_PROTO"} "${EXTRA_ENV_ARG[@]}" "${NCCL_ENV[@]}"
  -e NODE_RANK="$NODE_RANK" -e TP_SIZE="$TP_SIZE" -e ENABLE_EP=1 -e TP3_STRICT="${TP3_STRICT:-1}"
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

# SETTLE GATE. vLLM sizes the KV pool from a *delta*: MemAvailable just after NCCL
# init minus MemAvailable after the memory profile. On this integrated-GPU part
# "free GPU memory" IS /proc/meminfo MemAvailable, so whatever the kernel has not
# yet reclaimed from the container we just killed (~90 GiB) is charged to the pool
# -- and in the wrong direction: the dirtier the node starts, the BIGGER its pool
# reads. The last node started is the one given least time to reclaim, which is how
# a 9 GiB spread in per-rank startup free memory, and ~6 % of boot-to-boot noise in
# the pool number, appeared out of nothing. Waiting for the host to settle costs
# up to 180 s of boot (measured: well under 20 s) and buys zero tokens; what it
# buys is a pool number that means something. docs/07 section 1.1, docs/08 5.1.
sync
SETTLE_MIN_GIB="${SETTLE_MIN_GIB:-112}"
settle_avail=0
for settle_i in $(seq 1 60); do
  settle_avail=$(awk '/^MemAvailable:/{printf "%d", $2/1048576}' /proc/meminfo)
  [ "$settle_avail" -ge "$SETTLE_MIN_GIB" ] && break
  sleep 3
done
echo "mem settle: MemAvailable=${settle_avail} GiB (target ${SETTLE_MIN_GIB}) after $(( (settle_i - 1) * 3 ))s"

docker "${DOCKER_ARGS[@]}"
echo "launched $NAME rank=$NODE_RANK host=$HOST_IP"; sleep 2
docker ps --format '{{.Names}} {{.Status}}' | grep "$NAME" || { echo "$NAME exited; docker logs $NAME" >&2; exit 1; }
