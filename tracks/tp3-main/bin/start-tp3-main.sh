#!/usr/bin/env bash
# Launcher for the TP=3 main-stack engine: starts THIS node's rank. Called by
# harem-exl3.service as ExecStart (see the drop-in or the full unit next to this file).
# Same shape as the old tracks/tp3 launcher: oneshot, `docker run -d`, check that the
# container is up, exit (the unit stays active with RemainAfterExit).
#
# Everything comes from ENV_FILE (env.tp3-main.example, derived per node). The docker
# arguments below are the arguments of the production engine; the only things that move
# between environments are in the env file.
#
# Things this launcher does on purpose, each of which cost us something once:
#   * --ulimit core=0     a core dump of a crashed rank filled the disk (about 13 GiB each)
#   * --limit-mm-per-prompt is passed ONCE ({"image":16,"video":4}); a second copy of the
#                         flag silently wins over the first
#   * the previous container's log is saved before the container is removed (an earlier
#     launcher ran `docker rm -f` first and deleted the evidence of the crash)
#   * the memory settle gate: drop the page cache, then wait for MemAvailable
#   * HAREM_GLM47_FAILCLOSED=1 is set explicitly (the patch default is already 1; this is
#     documentation that survives a change of that default)
#
# DRY_RUN=1 prints the docker command and exits; it needs no GPU and starts nothing
# (it does call `docker image inspect IMG`, to read the image's LD_LIBRARY_PATH).
set -uo pipefail
export LC_ALL=C          # in a tr_TR locale [a-z] does not cover the letter i
: "${ENV_FILE:?ENV_FILE is not set}"
set -a; . "$ENV_FILE"; set +a
for v in NODE_RANK HOST_IP INSTALL_DIR MASTER_ADDR MASTER_PORT GLOO_IFACE CPUSET MESH_DIR MESH_TIMEOUT PORT \
         SERVED_MODEL_NAME IMG IMG_ID CKPT_DIR TARGET_SIDECAR DRAFT_DIR DRAFT_SIDECAR FASTLOAD_DIR FASTLOAD_MODE \
         CHAT_TEMPLATE CACHE_DIR GMU MAX_NUM_SEQS MNBT SETTLE_MIN_GIB; do
  [[ -n "${!v:-}" ]] || { echo "REFUSED: $v is empty or missing in $ENV_FILE" >&2; exit 2; }
done
case "$FASTLOAD_MODE" in
  load) FL_SUFFIX=":ro" ;;
  dump) FL_SUFFIX="" ;;
  *) echo "REFUSED: FASTLOAD_MODE must be load or dump (got '$FASTLOAD_MODE')" >&2; exit 2 ;;
esac
NAME=exl3-tp3
LOGD=$INSTALL_DIR/log; mkdir -p "$LOGD/stackdump"
log() { echo "[$(date '+%F %T')] [start-tp3-main r$NODE_RANK] $*" | tee -a "$LOGD/launch.log"; }

# --- draft configuration -------------------------------------------------------------
SPEC="{\"method\":\"dflash\",\"model\":\"$DRAFT_SIDECAR\",\"num_speculative_tokens\":7,\"kv_cache_dtype\":\"fp8\"}"
if [[ -n "${SPEC_EXTRA_JSON:-}" ]]; then
  B64=$(python3 "$INSTALL_DIR/tools/spec_extra.py" kodla "$SPEC_EXTRA_JSON") || { log "REFUSED: SPEC_EXTRA_JSON could not be encoded"; exit 2; }
  python3 "$INSTALL_DIR/tools/spec_extra.py" denetle "$B64" --nst 7 > /dev/null || { log "REFUSED: SPEC_EXTRA_JSON failed the check"; exit 2; }
  SPEC=$(python3 "$INSTALL_DIR/tools/spec_extra.py" birlestir "$SPEC" "$B64" 2>> "$LOGD/launch.log") || { log "REFUSED: SPEC could not be merged"; exit 2; }
fi

IMG_LDLP=$(docker image inspect "$IMG" --format '{{range .Config.Env}}{{println .}}{{end}}' | sed -n 's/^LD_LIBRARY_PATH=//p')
mkdir -p "$CACHE_DIR/triton" "$CACHE_DIR/tilelang" "$CACHE_DIR/flashinfer" "$CACHE_DIR/tune"
A=(run -d --gpus all --log-opt max-size=50m --log-opt max-file=5 --name "$NAME" --restart no
   --network host --ipc host --shm-size 32g --cpuset-cpus "$CPUSET" --ulimit memlock=-1:-1 --ulimit core=0 --cap-add IPC_LOCK
   --device /dev/infiniband:/dev/infiniband
   -v "$TARGET_SIDECAR:$TARGET_SIDECAR:ro" -v "$CKPT_DIR:$CKPT_DIR:ro" -v "$DRAFT_SIDECAR:$DRAFT_SIDECAR:ro" -v "$DRAFT_DIR:$DRAFT_DIR:ro"
   -v "$INSTALL_DIR/patches:/opt/harem-tp3:ro" -v "$INSTALL_DIR/patches/harem-prelude.sh:/start.sh:ro"
   -v "$CHAT_TEMPLATE:/models/chat_template.jinja:ro" -v "$INSTALL_DIR/stack-dump:/opt/harem-yigin:ro" -v "$LOGD/stackdump:/harem-kanit"
   -v "$CACHE_DIR:/cache" -v "$CACHE_DIR/triton:/root/.triton" -v "$CACHE_DIR/tilelang:/root/.tilelang" -v "$CACHE_DIR/flashinfer:/root/.cache/flashinfer"
   -v "$MESH_DIR:/opt/nccl-mesh:ro" -v "$FASTLOAD_DIR-r$NODE_RANK:$FASTLOAD_DIR-r$NODE_RANK$FL_SUFFIX"
   -e VLLM_HOST_IP="$HOST_IP" -e VLLM_CACHE_ROOT=/cache -e HF_HOME=/cache/huggingface -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1
   -e VLLM_ENGINE_READY_TIMEOUT_S=3600 -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT_S:-1800}"
   -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -e TORCH_CUDA_ARCH_LIST=12.1 -e FLASHINFER_CUDA_ARCH_LIST=12.1 -e FLASHINFER_DISABLE_VERSION_CHECK=1
   -e CUDA_EXL3_MLA_TUNE_VERBOSE=1 -e CUDA_EXL3_DEBUG_NAMES=1
   -e PYTHONPATH=/opt/harem-yigin -e HAREM_YIGIN_SN=0 -e HAREM_YIGIN_TEKRAR=0 -e HAREM_YIGIN_DIZIN=/harem-kanit
   -e HAREM_PREFIX_HIT=1 -e HAREM_SW_BLOCK_SIZE=256 -e HAREM_MAMBA_GRID=1 -e HAREM_VISION=1 -e HAREM_VISION_VIDEO_LIMIT=4
   -e HAREM_GLM47_FAILCLOSED=1
   -e VLLM_ALLREDUCE_USE_FLASHINFER=0 -e CUDA_EXL3_TUNE_CACHE=/cache/tune
   -e "HAREM_FASTLOAD_MODEL_PATH=$TARGET_SIDECAR" -e "HAREM_FASTLOAD_DRAFT_PATH=$DRAFT_SIDECAR" -e "HAREM_IMAGE_TAG=$IMG" -e TP3_DIR=/opt/harem-tp3
   -e HAREM_FASTLOAD_MODE="$FASTLOAD_MODE" -e "HAREM_FASTLOAD_DIR=$FASTLOAD_DIR" -e HAREM_FASTLOAD_VERIFY=32 -e HAREM_FASTLOAD_SHARD_BYTES=2147483648
   -e NCCL_CUMEM_ENABLE=0 -e NCCL_NVLS_ENABLE=0 -e NCCL_CROSS_NIC=0 -e NCCL_IB_MERGE_NICS=0 -e NCCL_IGNORE_CPU_AFFINITY=1
   -e NCCL_DEBUG=WARN -e TORCH_NCCL_ASYNC_ERROR_HANDLING=1 -e GLOO_SOCKET_IFNAME="$GLOO_IFACE" -e TP_SOCKET_IFNAME="$GLOO_IFACE" -e MN_IF_NAME="$GLOO_IFACE"
   -e NCCL_NET=Mesh -e NCCL_IB_DISABLE=1 -e "NCCL_SOCKET_IFNAME==$GLOO_IFACE" -e NCCL_NET_PLUGIN=mesh -e NCCL_ALGO=Ring -e NCCL_MESH_DEBUG=1
   -e "LD_LIBRARY_PATH=/opt/nccl-mesh${IMG_LDLP:+:$IMG_LDLP}"
   -e NCCL_MAX_NCHANNELS=8 -e NCCL_MESH_LINKS_PER_PEER=0 -e NCCL_MESH_MIN_RNR_TIMER=1 -e NCCL_MESH_PTR_CUDA=1 -e NCCL_MESH_FLUSH=1
   -e NCCL_MESH_TIMEOUT_SEC="$MESH_TIMEOUT"
   -e NODE_RANK="$NODE_RANK" -e TP_SIZE=3 -e ENABLE_EP=1 -e TP3_STRICT=1
   --entrypoint bash "$IMG" /start.sh "$TARGET_SIDECAR"
   --served-model-name "$SERVED_MODEL_NAME" --host 0.0.0.0 --port "$PORT" --trust-remote-code
   --quantization exl3 --attention-backend CUSTOM --tensor-parallel-size 3 --pipeline-parallel-size 1
   --gpu-memory-utilization "$GMU" --max-model-len 1000000 --max-num-seqs "$MAX_NUM_SEQS" --max-num-batched-tokens "$MNBT"
   --kv-cache-dtype fp8 --enable-prefix-caching --enable-chunked-prefill --dtype bfloat16
   --tool-call-parser glm47 --enable-auto-tool-choice --reasoning-parser deepseek_r1
   --distributed-executor-backend mp --nnodes 3 --node-rank "$NODE_RANK" --master-addr "$MASTER_ADDR" --master-port "$MASTER_PORT"
   --enable-expert-parallel --kda-prefill-backend flashkda
   --speculative-config "$SPEC"
   --chat-template /models/chat_template.jinja --skip-mm-profiling
   --default-chat-template-kwargs '{"clear_thinking":true,"reasoning_effort":"low"}'
   --block-size 256 --enable-ep-weight-filter --safetensors-load-strategy eager --no-enable-flashinfer-autotune
   --hf-overrides '{"index_topk":2048}' --mm-encoder-tp-mode data
   --mm-processor-kwargs '{"max_pixels":12544000,"max_image_tokens":8000}' --limit-mm-per-prompt '{"image":16,"video":4}'
   --mm-processor-cache-gb 0)
[[ "$NODE_RANK" != "0" ]] && A+=(--headless)
[[ "${PROMPT_DETAILS:-0}" == "1" ]] && A+=(--enable-prompt-tokens-details)

if [[ "${DRY_RUN:-0}" == "1" ]]; then printf '%q ' docker "${A[@]}"; echo; exit 0; fi

# --- dump mode: the sidecar directory is created here and is never overwritten ---------
if [[ "$FASTLOAD_MODE" == "dump" ]]; then
  if [[ -e "$FASTLOAD_DIR-r$NODE_RANK/MANIFEST.json" ]]; then
    log "REFUSED: $FASTLOAD_DIR-r$NODE_RANK already holds a sidecar (a dump boot never overwrites one; use a new FASTLOAD_DIR)"; exit 1
  fi
  mkdir -p "$FASTLOAD_DIR-r$NODE_RANK"
fi

# --- previous container: save its log, then remove it ---------------------------------------
if docker inspect "$NAME" > /dev/null 2>&1; then
  docker logs "$NAME" > "$LOGD/previous-container-$(date +%Y%m%d-%H%M%S).log" 2>&1
  docker rm -f "$NAME" > /dev/null 2>&1
fi
# --- keep the newest 12 of the old container / stop logs (a container log can reach 250 MB) --
for _p in previous-container stop; do ls -1t "$LOGD"/$_p-*.log 2>/dev/null | tail -n +13 | xargs -r rm -f; done
# --- settle gate: drop the page cache, then wait for MemAvailable to settle ---------------------
sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches > /dev/null
for _i in $(seq 1 100); do
  ma=$(awk '/MemAvailable/{print int($2/1048576)}' /proc/meminfo); [[ $ma -ge $SETTLE_MIN_GIB ]] && break; sleep 3
done
[[ $ma -ge $SETTLE_MIN_GIB ]] || { log "REFUSED: MemAvailable did not settle ($ma GiB < $SETTLE_MIN_GIB)"; exit 1; }
printf '%q ' docker "${A[@]}" > "$LOGD/docker-command-last.txt"; echo >> "$LOGD/docker-command-last.txt"
log "start: img=$IMG gmu=$GMU seqs=$MAX_NUM_SEQS mode=$FASTLOAD_MODE spec=$SPEC MemAvailable=${ma}GiB pswpout=$(awk '/^pswpout/{print $2}' /proc/vmstat)"
docker "${A[@]}" > "$LOGD/container-id.txt" 2>&1 || { log "REFUSED: docker run: $(cat "$LOGD/container-id.txt")"; exit 1; }
sleep 2
docker ps --format '{{.Names}}' | grep -qx "$NAME" || { log "REFUSED: $NAME stopped at once (docker logs $NAME)"; exit 1; }
log "container up: $NAME $(cut -c1-12 "$LOGD/container-id.txt")"
