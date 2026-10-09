#!/usr/bin/env bash
# Preflight for the TP=3 main-stack engine, run by harem-exl3.service as ExecStartPre on
# every node. Fail-closed: exit 1 stops the unit before anything starts, and the reason is
# in the journal. It starts and changes NOTHING (the page-cache drop is in the launcher).
#
# What it checks, in order:
#   1. docker answers; all four ConnectX-7 ports are PORT_ACTIVE; both fabric neighbours
#      answer a ping (warnings only: MTU 9000, 200 Gb/s link speed, a jumbo-frame ping)
#   2. no sibling engine unit is active, no container of ours is running, the GPU is empty
#   3. identities: the image Id, the install directory's MANIFEST.sha256 (if you made one),
#      the target sidecar, the draft's harem_tp_pad tag, the mesh plugin sha, the chat
#      template sha, the fast-load sidecar (only in FASTLOAD_MODE=load), free disk space
#   4. the JIT-cache gate (head only): the three nodes must hold identical JIT cache listings
#   5. a note if memory looks fragmented (never blocks)
#
# Why 1: at boot the engine starts long before the fabric is ready. Without it the NCCL
# rendezvous hangs with no useful error and the unit sits in "activating" until
# TimeoutStartSec. Failing here is loud and names the missing link.
#
# Why 4: JIT-compiled kernels (triton, tilelang, flashinfer, deep_gemm) are cached per node.
# A rank that has to compile what its neighbours already have arrives tens of seconds late
# at the first collective, and the mesh plugin's operation timeout turns that into a boot that
# hangs without a message. Equal caches mean equal timing. See the README, "The JIT cache gate".
#
# FABRIC_PEERS: the two fabric addresses THIS node must reach. The three nodes are wired as
# a ring of three cables carrying six logical links, so each node has two neighbours and a
# different pair of addresses (docs/00-hardware-and-os.md section 4.5). The values below are
# the example range of that page, NOT yours: set FABRIC_PEERS in the environment file or edit
# the case.
#
#   head      172.31.0.2 172.31.2.2
#   worker-1  172.31.0.1 172.31.4.2
#   worker-2  172.31.2.1 172.31.4.1
set -uo pipefail
export LC_ALL=C
: "${ENV_FILE:?ENV_FILE is not set}"
[[ -f "$ENV_FILE" ]] || { echo "PREFLIGHT-REFUSED: no env file: $ENV_FILE"; exit 1; }
set -a; . "$ENV_FILE"; set +a
t0=$(date +%s)
red() { echo "PREFLIGHT-REFUSED: $*"; exit 1; }
warn() { echo "PREFLIGHT-WARNING: $*"; }

NODE="${NODE_NAME:-$(hostname)}"
case "$NODE" in
  head)     PEERS="172.31.0.2 172.31.2.2"; WANT_RANK=0 ;;
  worker-1) PEERS="172.31.0.1 172.31.4.2"; WANT_RANK=1 ;;
  worker-2) PEERS="172.31.2.1 172.31.4.1"; WANT_RANK=2 ;;
  *)        PEERS=""; WANT_RANK="" ;;
esac
[[ -z "${FABRIC_PEERS:-}" ]] || PEERS="$FABRIC_PEERS"
[[ -n "$PEERS" ]] || red "unknown node name '$NODE': set NODE_NAME to head, worker-1 or worker-2, or set FABRIC_PEERS"
[[ -z "$WANT_RANK" || "$NODE_RANK" == "$WANT_RANK" ]] || red "$NODE must be rank $WANT_RANK (the env file says $NODE_RANK) -- was the env file copied from another node?"

# 1) hardware and fabric
until docker info > /dev/null 2>&1; do sleep 5; (( $(date +%s) - t0 > 300 )) && red "docker is not ready"; done
until [[ "$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)" == "4" ]]; do
  sleep 5; (( $(date +%s) - t0 > 600 )) && red "ConnectX-7 not 4/4: $(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)/4"
done
for p in $PEERS; do until ping -c1 -W2 "$p" > /dev/null 2>&1; do sleep 5; (( $(date +%s) - t0 > 600 )) && red "fabric neighbour $p is unreachable"; done; done
for i in enp1s0f0np0 enp1s0f1np1 enP2p1s0f0np0 enP2p1s0f1np1; do
  m=$(cat /sys/class/net/$i/mtu 2>/dev/null); s=$(cat /sys/class/net/$i/speed 2>/dev/null)
  [[ "$m" == 9000 && "$s" == 200000 ]] || warn "fabric port $i mtu=$m speed=$s (expected 9000/200000)"
done
for p in $PEERS; do ping -c1 -W2 -M do -s 8972 "$p" > /dev/null 2>&1 || warn "jumbo-frame (8972) ping to $p failed"; done

# 2) nothing else is using this node
for s in harem-motor; do [[ "$(systemctl is-active $s 2>/dev/null)" == "active" ]] && red "$s (the sibling engine's unit) is active"; done
k=$(docker ps --format '{{.Names}}' | grep -E '^exl3-tp3$' | tr '\n' ' ')
[[ -z "$k" ]] || red "an engine container is already running: $k"
g=$(nvidia-smi --query-compute-apps=pid,process_name --format=csv,noheader 2>/dev/null)
[[ -z "$g" ]] || red "a process is using the GPU: $g"

# 3) identities
[[ "$(docker image inspect "$IMG" --format '{{.Id}}' 2>/dev/null)" == "$IMG_ID" ]] || red "image Id of $IMG is not $IMG_ID"
if [[ -f "$INSTALL_DIR/MANIFEST.sha256" ]]; then
  ( cd "$INSTALL_DIR" && sha256sum -c --quiet MANIFEST.sha256 ) || red "the install directory does not match its MANIFEST.sha256 ($INSTALL_DIR): patches, tools or scripts changed"
else
  warn "no $INSTALL_DIR/MANIFEST.sha256 (README, step 'Install the files'); changes to the patches will only be caught by the fast-load identity, minutes into the boot"
fi
[[ -f $TARGET_SIDECAR/config.json && -f $TARGET_SIDECAR/model.safetensors.index.json ]] || red "target sidecar is missing: $TARGET_SIDECAR"
python3 -c "import json,sys; t=json.load(open('$DRAFT_SIDECAR/config.json')).get('harem_tp_pad') or {}; sys.exit(0 if t.get('tp')==3 else 1)" \
  || red "the draft sidecar has no TP3 harem_tp_pad tag (or is missing): $DRAFT_SIDECAR"
if [[ -n "${MESH_SO_SHA:-}" ]]; then
  [[ "$(sha256sum "$MESH_DIR/libnccl-net-mesh.so" 2>/dev/null | cut -c1-16)" == "$MESH_SO_SHA" ]] || red "sha of the mesh plugin ($MESH_DIR) is not $MESH_SO_SHA"
else
  [[ -f "$MESH_DIR/libnccl-net-mesh.so" ]] || red "the mesh plugin is missing: $MESH_DIR/libnccl-net-mesh.so"
fi
[[ -f "$CHAT_TEMPLATE" ]] || red "the chat template is missing: $CHAT_TEMPLATE"
if [[ -n "${CHAT_TEMPLATE_SHA:-}" ]]; then
  [[ "$(sha256sum "$CHAT_TEMPLATE" | cut -c1-16)" == "$CHAT_TEMPLATE_SHA" ]] || red "sha of the chat template ($CHAT_TEMPLATE) is not $CHAT_TEMPLATE_SHA"
fi
# The fast-load sidecar is required only in load mode. In dump mode the launcher creates it
# (the same narrowing the old tracks/tp3 preflight made, after 18 minutes of downtime).
if [[ "$FASTLOAD_MODE" == "load" ]]; then
  [[ -f "$FASTLOAD_DIR-r$NODE_RANK/MANIFEST.json" ]] || red "the fast-load sidecar is missing: $FASTLOAD_DIR-r$NODE_RANK (a new image needs a dump boot first: FASTLOAD_MODE=dump)"
fi
df=$(df -BG --output=avail /var/tmp | tail -1 | tr -dc 0-9); (( df >= 20 )) || red "free space in /var/tmp is ${df} GB < 20"

# 4) JIT-cache gate (head only; the workers have no ssh key to each other). The peers may still
#    be booting, so it retries for 10 minutes.
if [[ "$NODE_RANK" == "0" && "${JIT_GATE:-1}" == "1" ]]; then
  # A missing cache directory is an EMPTY listing (a fresh node), not an error: all three being empty is equal.
  man() { local h=$1; local c="export LC_ALL=C; { cd $CACHE_DIR 2>/dev/null && for d in triton tilelang flashinfer deep_gemm/cache; do [ -d \$d ] && find \$d -type f -printf '%p %s\n'; done; } | sort | sha256sum | cut -c1-16"
          if [[ "$h" == "-" ]]; then bash -c "$c"; else ssh -n -o BatchMode=yes -o ConnectTimeout=6 "$h" "$c"; fi; }
  read -r P1 P2 _ <<< "${JIT_PEERS:-}"
  [[ -n "${P1:-}" && -n "${P2:-}" ]] || red "JIT_PEERS must name the two other nodes (or set JIT_GATE=0)"
  for _j in $(seq 1 60); do
    a=$(man -); b=$(man "$P1" 2>/dev/null); c=$(man "$P2" 2>/dev/null)
    [[ -n "$b" && -n "$c" ]] && break; sleep 10
  done
  [[ -n "$a" && "$a" == "$b" && "$a" == "$c" ]] || red "JIT caches are not identical: head=$a $P1=${b:-?} $P2=${c:-?} (README, 'The JIT cache gate')"
  echo "JIT-GATE: passed ($a)"
fi

# 5) fragmentation note: a fresh boot has many free order-13 pages; after many engine restarts
#    on the same boot there are few, and memory bandwidth was measured 8-11 % lower.
o13=$(awk '$4=="Normal"{print $(5+13)}' /proc/buddyinfo | paste -sd+ | bc 2>/dev/null)
up=$(cut -d. -f1 /proc/uptime)
if [[ -n "$o13" ]] && (( o13 < ${FRAG_MIN_BLOCKS:-2000} )); then
  warn "memory looks fragmented: ${o13} free order-13 blocks < ${FRAG_MIN_BLOCKS:-2000} (uptime ${up}s); bandwidth may be 8-11 % lower, a whole-cluster reboot restores it"
fi
echo "preflight ok: $(( $(date +%s) - t0 )) s, rank=$NODE_RANK img=$IMG gmu=$GMU mode=$FASTLOAD_MODE sidecar=$FASTLOAD_DIR-r$NODE_RANK order13=${o13:-?} uptime=${up}s"
