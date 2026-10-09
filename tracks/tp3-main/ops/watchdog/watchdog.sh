#!/bin/bash
# Engine watchdog for the TP=3 main stack. Runs on rank 0 (head), once a minute, from a systemd
# timer. It reboots the WHOLE cluster, in the safe order, when the engine is down or hung.
#
# Why a whole-cluster reboot and not a restart of one rank: a node that reboots on its own can
# leave ConnectX-7 link pairs on the OTHER nodes down, and the engine then dies on the next
# collective (docs/00-hardware-and-os.md sections 3.1 and 3.4: both ends of a link must come up
# together). The rule on this cluster is "reboot all three, or none". So the action here is three
# reboots at once, the workers first and the head last.
#
# What it does, each round:
#   * "health 200" is NOT enough: a small completion request (max_tokens 8) that does not return
#     in 60 s counts as a hang (the mesh timeout failure mode: /health answers, requests never do).
#   * 3 CONSECUTIVE failures -> action. The action is the triple reboot: first the peers (ssh,
#     sudo -n systemctl reboot), then this node.
#   * rate limit: at most 1 reboot in 30 minutes and 3 in 6 hours. If exceeded it STOPS and
#     writes an ALARM file (nothing acts until the file is deleted by a person).
#   * pause file: while PAUSE exists it does nothing (use it before maintenance).
#   * --dry-run: writes a "WOULD DO" log and triggers nothing (no ssh reboot, no reboot, no
#     evidence collection).
# Safety additions, each written down on purpose:
#   - busy-queue exemption: if the probe did not return BUT the token counters in /metrics
#     advanced while it waited, the engine is "busy" (with all seats full the probe may simply
#     have queued). Frozen counters are a real hang. It is switched off with WATCHDOG_EXEMPTION=0
#     and used at most WATCHDOG_BUSY_MAX (10) times in a row, so a hang cannot be masked forever.
#   - when the unit is 'inactive' (stopped on purpose, or not started yet) it does not touch
#     anything; when it is 'failed' that counts as a failure.
#   - it does nothing in the first 15 minutes of uptime, nor while the container is younger than 25 minutes.
#   - if a peer cannot be reached over ssh it does NOT reboot (a one-node reboot kills the other
#     ports) and writes an ALARM instead.
#   - before acting it collects evidence: `docker logs --tail 300 exl3-tp3` from every node, into
#     $STATE/evidence-<time>/.
# The exit status is always 0 (so the timer does not show "failed"); the decision is in the log:
# $STATE/events.log and the journal (tag harem-watchdog).
set -u
DRY=0; { [ "${1:-}" = "--dry-run" ] || [ "${1:-}" = "--kuru" ]; } && DRY=1      # --kuru is the old name of --dry-run
PAUSE=${WATCHDOG_PAUSE:-/var/tmp/harem-watchdog-pause}
STATE=${WATCHDOG_STATE:-/var/tmp/harem-watchdog}
BASE=${WATCHDOG_URL:-http://127.0.0.1:8001}
MODEL=${WATCHDOG_MODEL:-glm-5.3-flash}
PEERS=${WATCHDOG_PEERS:-worker-1 worker-2}
PROBE_TIMEOUT=${WATCHDOG_PROBE_TIMEOUT:-60}
FAIL_NEEDED=${WATCHDOG_FAIL_NEEDED:-3}
BOOT_GRACE_S=${WATCHDOG_BOOT_GRACE_S:-900}
CONT_GRACE_S=${WATCHDOG_CONT_GRACE_S:-1500}
W1=${WATCHDOG_W1:-1800}; MAX1=${WATCHDOG_MAX1:-1}
W2=${WATCHDOG_W2:-21600}; MAX2=${WATCHDOG_MAX2:-3}
BUSY_MAX=${WATCHDOG_BUSY_MAX:-10}
EXEMPTION=${WATCHDOG_EXEMPTION:-1}
SSH=${WATCHDOG_SSH:-ssh}
SELF_REBOOT=${WATCHDOG_SELF_REBOOT:-sudo -n systemctl reboot}
DOCKER_LOGS=${WATCHDOG_DOCKER_LOGS:-sudo -n docker logs --tail 300 exl3-tp3}
NOW=${WATCHDOG_NOW:-$(date +%s)}
mkdir -p "$STATE" 2>/dev/null || { echo "watchdog: cannot create $STATE" >&2; exit 0; }
SFX=""; [ "$DRY" = 1 ] && SFX=".dry"
F_FAILS="$STATE/fails$SFX"; F_BUSY="$STATE/busy$SFX"; F_REST="$STATE/reboots$SFX"; F_ALARM="$STATE/ALARM$SFX"
ET=""; [ "$DRY" = 1 ] && ET="[DRY] "
log() { logger -t harem-watchdog -- "$ET$*" 2>/dev/null; echo "$(date -d "@$NOW" '+%F %T') $ET$*" >> "$STATE/events.log"; [ "$DRY" = 1 ] && echo "$ET$*"; }
round() { echo "$(date -d "@$NOW" '+%F %T') $ET$*" > "$STATE/last-round.txt"; }     # heartbeat: every round writes its last state

exec 9> "$STATE/lock"; flock -n 9 || exit 0

[ -e "$PAUSE" ] && { round "paused ($PAUSE)"; [ "$DRY" = 1 ] && echo "${ET}pause file exists: I do nothing"; exit 0; }
if [ -e "$F_ALARM" ]; then
  round "ALARM file exists, no action"
  # at most one log line an hour
  if [ ! -e "$STATE/alarm-log$SFX" ] || [ $(( NOW - $(stat -c %Y "$STATE/alarm-log$SFX" 2>/dev/null || echo 0) )) -gt 3600 ]; then log "ALARM open ($F_ALARM): a person is needed, I do nothing"; touch "$STATE/alarm-log$SFX"; fi
  exit 0
fi

# --- boot / stop immunity ---
UP=${WATCHDOG_UPTIME_S:-$(cut -d. -f1 /proc/uptime)}
if [ "$UP" -lt "$BOOT_GRACE_S" ]; then round "boot grace (uptime ${UP}s < $BOOT_GRACE_S)"; exit 0; fi
UNIT=${WATCHDOG_UNIT_STATE:-$(systemctl is-active harem-exl3 2>/dev/null)}
case "$UNIT" in
  activating|reloading) round "the unit is starting, I do not touch it"; exit 0 ;;
  inactive) round "unit inactive (stopped on purpose or not started yet): no engine to watch"; echo 0 > "$F_FAILS"; echo 0 > "$F_BUSY"; exit 0 ;;
esac
if [ -n "${WATCHDOG_CONT_AGE_S:-}" ]; then
  [ "$WATCHDOG_CONT_AGE_S" != "none" ] && [ "$WATCHDOG_CONT_AGE_S" -lt "$CONT_GRACE_S" ] && { round "container is young (${WATCHDOG_CONT_AGE_S}s)"; exit 0; }
else
  started=$(sudo -n docker inspect -f '{{.State.StartedAt}}' exl3-tp3 2>/dev/null)
  if [ -n "$started" ]; then
    age=$(( NOW - $(date -d "$started" +%s 2>/dev/null || echo 0) ))
    if [ "$age" -lt "$CONT_GRACE_S" ] && sudo -n docker ps --format '{{.Names}}' 2>/dev/null | grep -qx exl3-tp3; then round "container is young (${age}s)"; exit 0; fi
  fi
fi

# --- 1) health + progress probe ---
counter() { curl -s -m 5 "$BASE/metrics" 2>/dev/null | awk '/^vllm:(generation_tokens_total|prompt_tokens_total)[{ ]/ {s+=$NF; n++} END {if (n>0) printf "%.0f", s}'; }
FAIL_REASON=""; STATUS=ok
code=$(curl -s -m 5 -o /dev/null -w '%{http_code}' "$BASE/health" 2>/dev/null)
if [ "$code" != "200" ]; then
  FAIL_REASON="health http=${code:-000}"; STATUS=fail
else
  c0=$(counter)
  body="$STATE/probe-body$SFX.json"
  pcode=$(curl -s -m "$PROBE_TIMEOUT" -o "$body" -w '%{http_code}' -H 'Content-Type: application/json' \
          -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"1+1=\"}],\"max_tokens\":8,\"temperature\":0,\"user\":\"watchdog\"}" \
          "$BASE/v1/chat/completions" 2>/dev/null)
  if [ "$pcode" = "200" ] && grep -q '"choices"' "$body" 2>/dev/null; then
    STATUS=ok; echo 0 > "$F_BUSY"
  else
    c1=$(counter)
    if [ "$EXEMPTION" = 1 ] && [ -n "$c0" ] && [ -n "$c1" ] && [ "$c1" -gt "$c0" ]; then
      y=$(( $(cat "$F_BUSY" 2>/dev/null || echo 0) + 1 )); echo "$y" > "$F_BUSY"
      if [ "$y" -lt "$BUSY_MAX" ]; then STATUS=busy; log "probe did not return within ${PROBE_TIMEOUT}s (http=${pcode:-000}) BUT the token counter advanced ($c0 -> $c1): engine busy, NOT counted as a hang ($y/$BUSY_MAX)"
      else FAIL_REASON="probe does not return, $y rounds in a row on the 'busy' exemption (the counter advances but no request ever returns)"; STATUS=fail; fi
    else
      FAIL_REASON="probe did not return or was bad (http=${pcode:-000}, time limit ${PROBE_TIMEOUT}s, token counter ${c0:-?} -> ${c1:-?})"; STATUS=fail
    fi
  fi
fi
if [ "$STATUS" = ok ] || [ "$STATUS" = busy ]; then
  echo 0 > "$F_FAILS"; round "$STATUS"; [ "$DRY" = 1 ] && echo "${ET}round result: $STATUS"; exit 0
fi

# --- 2) failure counter ---
fails=$(( $(cat "$F_FAILS" 2>/dev/null || echo 0) + 1 )); echo "$fails" > "$F_FAILS"
if [ "$fails" -lt "$FAIL_NEEDED" ]; then log "NO HEALTH/PROGRESS ($fails/$FAIL_NEEDED): $FAIL_REASON -- waiting"; round "fail $fails/$FAIL_NEEDED"; exit 0; fi

# --- 3) rate limit ---
touch "$F_REST"
n1=$(awk -v n="$NOW" -v w="$W1" '$1 > n-w' "$F_REST" | wc -l); n2=$(awk -v n="$NOW" -v w="$W2" '$1 > n-w' "$F_REST" | wc -l)
if [ "$n1" -ge "$MAX1" ] || [ "$n2" -ge "$MAX2" ]; then
  { echo "ALARM $(date -d "@$NOW" '+%F %T')"; echo "reason: action was needed ($FAIL_REASON) but the rate limit was exceeded: $n1 reboots in the last $((W1/60)) min (limit $MAX1), $n2 in the last $((W2/3600)) h (limit $MAX2)"
    echo "what was done: NOTHING. The watchdog stopped."; echo "a person is needed: find out why it hangs/falls (container logs, $STATE/evidence-*), then delete this file: $F_ALARM"
    echo "last reboot stamps:"; tail -5 "$F_REST" | while read -r t; do echo "  $(date -d "@$t" '+%F %T')"; done; } > "$F_ALARM"
  logger -p user.crit -t harem-watchdog -- "${ET}ALARM: rate limit exceeded, watchdog stopped ($F_ALARM)" 2>/dev/null
  log "ALARM: action was needed ($FAIL_REASON) but the rate limit was exceeded (n1=$n1/$MAX1 n2=$n2/$MAX2) -> I STOP, alarm file written"
  echo 0 > "$F_FAILS"; round "ALARM rate limit"; exit 0
fi

# --- 4) are the peers reachable (if not, NO one-node reboot) ---
for p in $PEERS; do
  if ! $SSH -o BatchMode=yes -o ConnectTimeout=6 "$p" true 2>/dev/null; then
    { echo "ALARM $(date -d "@$NOW" '+%F %T')"; echo "reason: $p is not reachable over ssh; a triple reboot cannot be coordinated (a one-node reboot kills the other ports)"; echo "trigger: $FAIL_REASON"; echo "a person is needed; then delete this file: $F_ALARM"; } > "$F_ALARM"
    logger -p user.crit -t harem-watchdog -- "${ET}ALARM: peer $p not reachable, no reboot" 2>/dev/null
    log "ALARM: peer $p not reachable; NO reboot, alarm written"; echo 0 > "$F_FAILS"; round "ALARM peer"; exit 0
  fi
done

# --- 5) action ---
log "ENGINE DOWN/HUNG: $fails consecutive rounds ($FAIL_REASON) -> TRIPLE REBOOT (first $PEERS, then this node)"
if [ "$DRY" = 1 ]; then
  log "WOULD DO: collect evidence (docker logs --tail 300 exl3-tp3, all three nodes) -> $STATE/evidence-<time>/"
  for p in $PEERS; do log "WOULD DO: $SSH -o BatchMode=yes $p 'sudo -n systemctl reboot'"; done
  log "WOULD DO: wait 5 s; then $SELF_REBOOT"
  echo "$NOW" >> "$F_REST"; echo 0 > "$F_FAILS"; round "DRY: would reboot"; exit 0
fi
echo "$NOW" >> "$F_REST"; echo 0 > "$F_FAILS"       # the stamp is written BEFORE the reboot (this node becomes the script when it restarts)
date -d "@$NOW" '+%F %T TRIPLE REBOOT (watchdog): '"$FAIL_REASON" > "$STATE/last-event.txt"
KD="$STATE/evidence-$(date -d "@$NOW" +%Y%m%d-%H%M%S)"; mkdir -p "$KD"
curl -s -m 5 "$BASE/metrics" > "$KD/metrics.txt" 2>/dev/null
timeout 20 $DOCKER_LOGS > "$KD/head.log" 2>&1 &
for p in $PEERS; do timeout 20 $SSH -o BatchMode=yes -o ConnectTimeout=6 "$p" "$DOCKER_LOGS" > "$KD/$p.log" 2>&1 & done
wait
for p in $PEERS; do timeout 20 $SSH -o BatchMode=yes -o ConnectTimeout=6 "$p" 'sudo -n systemctl reboot' >> "$STATE/events.log" 2>&1 & done
wait      # the ssh commands are sent (at most 20 s); then a short wait, THEN this node
sleep 3
log "reboot given to the peers; now this node: $SELF_REBOOT"
$SELF_REBOOT >> "$STATE/events.log" 2>&1
exit 0
