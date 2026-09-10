#!/bin/bash
# Engine preflight, run by harem-exl3.service as ExecStartPre on every node.
# Waits (at most 10 minutes) until this node can actually serve, then checks
# that the three things a boot needs are on disk before the container starts.
#
#   1. docker is up and answering
#   2. all four ConnectX-7 ports report PORT_ACTIVE (ibv_devinfo)
#   3. this node can ping both of its fabric neighbours
#   4. sync + drop_caches
#   5. the env file exists
#   6. the image named in it is present locally
#   7. this rank's fast-load sidecar has its MANIFEST.json -- ONLY in load mode
#
# Why 2 and 3 exist: at boot the engine starts long before the fabric is ready.
# Without them the NCCL rendezvous hangs with no useful error and the unit sits
# in "activating" until TimeoutStartSec expires. Failing here instead is loud
# and names the missing link.
#
# Why 5 to 7 exist: each of them has failed us as a silent boot. A missing env
# file starts the engine on defaults; a missing image pulls or fails deep inside
# docker; a missing or half-written sidecar turns a 251 s boot into a 620 s one
# or into the fast-load refusal in docs/14 section 3.1. Checking them costs
# milliseconds and turns three late failures into one early message.
#
# FABRIC_PEERS: the two fabric addresses THIS node must be able to reach. The
# three nodes are wired as a ring of three cables carrying six logical links, so
# each node has two neighbours and a different pair of addresses (docs/00
# section 4.5). Set it per node in the environment, or fill in the case below.
# The addresses shown follow the example range in docs/00 section 4.5 -- they
# are NOT ours and you must substitute your own.
#
#   head      FABRIC_PEERS="172.31.0.2 172.31.2.2"
#   worker-1  FABRIC_PEERS="172.31.0.1 172.31.4.2"
#   worker-2  FABRIC_PEERS="172.31.2.1 172.31.4.1"
#
# Note what this does NOT check: it pings one address per neighbour, which is
# one link of each pair. PORT_ACTIVE on the other two is link state, not proof
# that anything crosses them -- read port_xmit_data as a delta after the first
# benchmark instead (docs/06 section 6).
#
# drop_caches needs root: either run the unit as root, or give the service user
# a NOPASSWD sudoers line for /usr/bin/tee /proc/sys/vm/drop_caches. If it is
# not permitted the script does not fail -- it just skips the drop.

PEERS="${FABRIC_PEERS:-}"
if [ -z "$PEERS" ]; then
  case "${NODE_NAME:-$(hostname)}" in
    head)     PEERS="172.31.0.2 172.31.2.2" ;;
    worker-1) PEERS="172.31.0.1 172.31.4.2" ;;
    worker-2) PEERS="172.31.2.1 172.31.4.1" ;;
    *)        PEERS="" ;;
  esac
fi

t0=$(date +%s)
until docker info >/dev/null 2>&1; do
  sleep 5
  [ $(( $(date +%s)-t0 )) -gt 300 ] && { echo "docker not ready"; exit 1; }
done

until [ "$(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)" = "4" ]; do
  sleep 5
  [ $(( $(date +%s)-t0 )) -gt 600 ] && {
    echo "ConnectX-7 not 4/4: $(ibv_devinfo 2>/dev/null | grep -c PORT_ACTIVE)/4"; exit 1; }
done

for p in $PEERS; do
  until ping -c1 -W2 "$p" >/dev/null 2>&1; do
    sleep 5
    [ $(( $(date +%s)-t0 )) -gt 600 ] && { echo "fabric peer $p unreachable"; exit 1; }
  done
done

sync; echo 3 | sudo -n tee /proc/sys/vm/drop_caches >/dev/null 2>&1
echo "preflight ok: $(( $(date +%s)-t0 )) s, ConnectX-7 4/4, peers: $PEERS"

# The engine's own preconditions: env file, image, this rank's sidecar.
ENVF=${ENV_FILE:-$HOME/exl3-zeus/.env.tp3}
test -f "$ENVF" || { echo "no env file: $ENVF"; exit 1; }

IMG=$(grep -E "^IMAGE=" "$ENVF" | cut -d= -f2)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image not present: $IMG"; exit 1; }

# Check 7 is MODE-AWARE, and it cost 18 minutes of downtime to make it so
# (docs/14 section 10.6). A missing sidecar is fatal only in FASTLOAD_MODE=load.
# In dump mode the launcher creates the directory itself, so demanding the
# MANIFEST there means a NEW FASTLOAD_DIR can never be booted: the mode whose job
# is to create the directory requires the directory to already exist. With the
# mode empty, fast loading is not in play at all. The gate is narrowed, not
# removed -- in load mode a missing or half-written sidecar is still a boot this
# script should stop here rather than four minutes later.
FD=$(grep -E "^FASTLOAD_DIR=" "$ENVF" | cut -d= -f2)
R=$(grep -E "^NODE_RANK=" "$ENVF" | cut -d= -f2)
FM=$(grep -E "^FASTLOAD_MODE=" "$ENVF" | cut -d= -f2)
[ -z "$FD" ] || [ "$FM" != load ] || test -f "$FD-r$R/MANIFEST.json" || { echo "fast-load sidecar missing: $FD-r$R"; exit 1; }

echo "exl3 preflight ok: env=$ENVF image=$IMG sidecar=$FD-r$R mode=${FM:-none}"
