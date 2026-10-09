#!/usr/bin/env bash
# jit-cache-sync.sh -- make the JIT-compiled kernel caches of the three nodes identical, which is
# what the preflight's JIT gate (bin/preflight-tp3-main.sh, head only) demands before it lets the
# engine start. Run it from any machine that has key-based ssh to the three nodes.
#
#   jit-cache-sync.sh check   print each node's cache size and say whether the three are identical
#   jit-cache-sync.sh sync    build the UNION of the three caches in a staging directory here, write that
#                             union back to every node (--delete), then check
#
# Why a union and not "copy the head's cache": each node compiles the kernels its own rank needs,
# and the three sets differ; the union has everything, and with the union on every node no rank has to
# compile anything at the first request, so no rank arrives late at the first collective
# (the failure the mesh timeout turns into a silent hang -- README, "The JIT cache gate").
#
# Only the JIT directories are synchronised: triton, tilelang, flashinfer, deep_gemm/cache. The tuner
# measurements (tune/, flashinfer_autotune_cache/, modelinfos/) are per node by design and stay as they are.
# The engine must be STOPPED while you sync. `sync` needs `sudo -n rsync` and `sudo -n find` on the nodes
# (the cache files are written by a root process in the container).
#
# Environment:
#   NODES      ssh names of the three nodes, head first      (default: "head worker-1 worker-2")
#   CACHE_DIR  the cache directory, the SAME absolute path on every node (CACHE_DIR of the env file)
#   STAGE      a scratch directory on THIS machine           (default: $HOME/jit-stage)
set -uo pipefail
MODE="${1:?usage: jit-cache-sync.sh check|sync}"
read -r -a NODES <<< "${NODES:-head worker-1 worker-2}"
CACHE_DIR="${CACHE_DIR:-/var/tmp/exl3-main/cache}"
STAGE="${STAGE:-$HOME/jit-stage}"
DIRS="triton tilelang flashinfer deep_gemm/cache"
LISTS=$(mktemp -d); trap 'rm -rf "$LISTS"' EXIT

listing() {   # $1 node -> "relative-path size", sorted
  ssh -n -o BatchMode=yes "$1" "cd $CACHE_DIR 2>/dev/null && for d in $DIRS; do [ -d \$d ] && sudo -n find \$d -type f -printf '%p %s\n'; done | sort"
}

check() {
  local ok=0 h
  for h in "${NODES[@]}"; do listing "$h" > "$LISTS/$h.txt"; done
  for h in "${NODES[@]:1}"; do
    if ! cmp -s "$LISTS/${NODES[0]}.txt" "$LISTS/$h.txt"; then
      ok=1; echo "JIT-CACHES DIFFER: ${NODES[0]} and $h: $(diff "$LISTS/${NODES[0]}.txt" "$LISTS/$h.txt" | grep -c '^[<>]') lines"
    fi
  done
  for h in "${NODES[@]}"; do echo "  $h: $(wc -l < "$LISTS/$h.txt") files, $(awk '{s+=$2} END{print s+0}' "$LISTS/$h.txt") bytes"; done
  [[ $ok -eq 0 ]] && echo "JIT-CACHES IDENTICAL ($(wc -l < "$LISTS/${NODES[0]}.txt") files)" || echo "JIT-CACHES NOT IDENTICAL"
  return $ok
}

sync_all() {
  local h d
  rm -rf "$STAGE"; mkdir -p "$STAGE"
  # the first node (head) is the base; the others only add the files it lacks (--ignore-existing)
  for h in "${NODES[@]}"; do
    for d in $DIRS; do
      mkdir -p "$STAGE/$d"
      rsync -a --ignore-existing --rsync-path="sudo -n rsync" "$h:$CACHE_DIR/$d/" "$STAGE/$d/" || { echo "SYNC-REFUSED: cannot read $h:$d"; return 2; }
    done
  done
  for h in "${NODES[@]}"; do
    for d in $DIRS; do
      ssh -n -o BatchMode=yes "$h" "sudo -n mkdir -p $CACHE_DIR/$d"
      rsync -a --delete --rsync-path="sudo -n rsync" "$STAGE/$d/" "$h:$CACHE_DIR/$d/" || { echo "SYNC-REFUSED: cannot write $h:$d"; return 2; }
    done
    echo "synchronised: $h"
  done
  check
}

case "$MODE" in check) check ;; sync) sync_all ;; *) echo "usage: jit-cache-sync.sh check|sync"; exit 2 ;; esac
