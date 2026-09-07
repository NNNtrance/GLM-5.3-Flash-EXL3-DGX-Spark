#!/bin/bash
# Engine preflight for the TWO-NODE recipe, run by harem-exl3-tp2.service as
# ExecStartPre on both nodes. It is motor-onkosul-exl3.sh with exactly two
# differences, and they are the ones docs/15 section 2.6 names:
#
#   * FABRIC_PEERS is ONE address per node, not two. Two nodes are one peer
#     pair joined by one cable pair; the third node's links are not part of
#     this cluster and waiting for them would hang the boot.
#   * the default ENV_FILE is .env.tp2d (candidate D).
#
# Everything else is unchanged, deliberately:
#   1. docker is up and answering
#   2. all four ConnectX-7 ports report PORT_ACTIVE (ibv_devinfo) -- STILL 4/4.
#      That counts ports on this node, not peers, and a node whose other two
#      ports are down has a hardware problem worth failing on even when this
#      cluster does not use them.
#   3. this node can ping its fabric neighbour
#   4. sync + drop_caches
#   5. the env file exists
#   6. the image named in it is present locally
#   7. this rank's fast-load sidecar has its MANIFEST.json
#
# The addresses below follow the example range in docs/00 section 4.5 -- they
# are NOT ours and you must substitute your own. Set FABRIC_PEERS in the
# environment instead if you prefer not to edit the file.
#
#   head      FABRIC_PEERS="172.31.0.2"
#   worker-1  FABRIC_PEERS="172.31.0.1"
#
# And read the reboot rule with it: at two ranks you reboot BOTH nodes together
# or neither. This preflight passes on a single node whose peer is gone, then
# starts a rank into a cluster that will never form (docs/00 section 3.4).

PEERS="${FABRIC_PEERS:-}"
if [ -z "$PEERS" ]; then
  case "${NODE_NAME:-$(hostname)}" in
    head)     PEERS="172.31.0.2" ;;
    worker-1) PEERS="172.31.0.1" ;;
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
echo "preflight ok: $(( $(date +%s)-t0 )) s, ConnectX-7 4/4, peer: $PEERS"

ENVF=${ENV_FILE:-$HOME/exl3-zeus/.env.tp2d}
test -f "$ENVF" || { echo "no env file: $ENVF"; exit 1; }

IMG=$(grep -E "^IMAGE=" "$ENVF" | cut -d= -f2)
docker image inspect "$IMG" >/dev/null 2>&1 || { echo "image not present: $IMG"; exit 1; }

FD=$(grep -E "^FASTLOAD_DIR=" "$ENVF" | cut -d= -f2)
R=$(grep -E "^NODE_RANK=" "$ENVF" | cut -d= -f2)
[ -z "$FD" ] || test -f "$FD-r$R/MANIFEST.json" || { echo "fast-load sidecar missing: $FD-r$R"; exit 1; }

echo "exl3 tp2 preflight ok: env=$ENVF image=$IMG sidecar=$FD-r$R"
