#!/usr/bin/env bash
# run-tests.sh -- test suite for ops/watchdog/watchdog.sh against a MOCK engine. It touches no real
# machine, no real engine, no real ssh and no real reboot:
#   * a mock engine on 127.0.0.1:18081 (mock_engine.py), a mock ssh and a mock reboot that only write to a log,
#   * a scratch state directory, and a clock that is advanced with WATCHDOG_NOW (so the rate limits can be tested).
# Usage: bash run-tests.sh      (exit status 0 = every check passed)
set -uo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
B=${WATCHDOG_SCRIPT:-$(cd "$HERE/.." && pwd)/watchdog.sh}
W=${WATCHDOG_TEST_DIR:-$(mktemp -d)}; rm -rf "$W"; mkdir -p "$W"
PORT=${WATCHDOG_TEST_PORT:-18081}; CONF=$W/engine.json; CALLS=$W/calls.log; : > "$CALLS"
PASS=0; FAIL=0
t() { if [[ "$2" == "0" ]]; then echo "PASS  $1"; PASS=$((PASS+1)); else echo "FAIL  $1   ${3:-}"; FAIL=$((FAIL+1)); fi; }
engine() { echo "$1" > "$CONF"; }
# --- mock commands ---
cat > "$W/mock-ssh" <<EOS
#!/bin/bash
# usage: mock-ssh [-o ..]... <peer> <command...>   -> records the call
args=("\$@"); while [[ "\${args[0]}" == -o ]]; do args=("\${args[@]:2}"); done
peer=\${args[0]}; cmd="\${args[*]:1}"
echo "\$(date +%s.%N) SSH \$peer :: \$cmd" >> "$CALLS"
[[ "\$cmd" == true && -e "$W/unreachable-\$peer" ]] && exit 255
exit 0
EOS
cat > "$W/mock-reboot" <<EOS
#!/bin/bash
echo "\$(date +%s.%N) SELF-REBOOT" >> "$CALLS"
EOS
chmod +x "$W/mock-ssh" "$W/mock-reboot"
python3 "$HERE/mock_engine.py" $PORT "$CONF" & ENGINE=$!
trap 'kill $ENGINE 2>/dev/null' EXIT
sleep 1
engine '{"health":200,"chat_delay":0}'
curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:$PORT/health | grep -q 200 || { echo "the mock engine did not start"; exit 2; }

export WATCHDOG_URL=http://127.0.0.1:$PORT WATCHDOG_PROBE_TIMEOUT=2 WATCHDOG_UPTIME_S=100000 WATCHDOG_UNIT_STATE=active WATCHDOG_CONT_AGE_S=none \
       WATCHDOG_PAUSE=$W/pause WATCHDOG_SSH=$W/mock-ssh WATCHDOG_SELF_REBOOT=$W/mock-reboot WATCHDOG_DOCKER_LOGS="echo mock-docker-log"
fresh() { export WATCHDOG_STATE=$W/state-$1; rm -rf "$WATCHDOG_STATE"; unset WATCHDOG_NOW; rm -f "$W/pause" "$W"/unreachable-*; : > "$CALLS"; engine '{"health":200,"chat_delay":0}'; }
round() { bash "$B" "$@" > "$W/last-output.txt" 2>&1; }
count() { grep -c "$1" "$CALLS"; }

echo "=== S1 healthy engine ==="
fresh s1; round; [[ "$(cat $WATCHDOG_STATE/fails)" == 0 ]] && grep -q "ok" $WATCHDOG_STATE/last-round.txt; t "S1a healthy: fails=0, last round ok" $?
[[ "$(count 'systemctl reboot')" == 0 ]]; t "S1b no reboot call" $?

echo "=== S2 silent hang (health 200, probe does not return, counters frozen) -- DRY RUN ==="
fresh s2; engine '{"health":200,"chat_delay":6,"metrics_move":false}'
round --dry-run; round --dry-run; grep -q "waiting" $WATCHDOG_STATE/events.log; t "S2a rounds 1-2: 'waiting' (no action)" $?
round --dry-run; grep -q "WOULD DO: .*worker-1.*systemctl reboot" $WATCHDOG_STATE/events.log && grep -q "WOULD DO: .*worker-2.*systemctl reboot" $WATCHDOG_STATE/events.log && grep -q "WOULD DO: wait 5 s" $WATCHDOG_STATE/events.log; t "S2b round 3: 'WOULD DO' log (worker-1, worker-2, self)" $?
[[ "$(count 'systemctl reboot')" == 0 && "$(count SELF-REBOOT)" == 0 ]]; t "S2c dry run: NO real reboot call" $? "$(cat $CALLS)"
[[ -s $WATCHDOG_STATE/reboots.dry && ! -e $WATCHDOG_STATE/reboots ]]; t "S2d the dry-run stamp is in reboots.dry, the real reboots file is untouched" $?
grep -q "token counter" $WATCHDOG_STATE/events.log; t "S2e the log gives the reason (probe/counter)" $?

echo "=== S3 the same hang in REAL mode (mock ssh/reboot): order and stamp ==="
fresh s3; engine '{"health":200,"chat_delay":6}'
round; round; : > "$CALLS"; round
[[ "$(count 'SSH worker-1 :: sudo -n systemctl reboot')" == 1 && "$(count 'SSH worker-2 :: sudo -n systemctl reboot')" == 1 && "$(count SELF-REBOOT)" == 1 ]]; t "S3a worker-1, worker-2 and SELF were each rebooted once" $? "$(cat $CALLS)"
ORDER=$(grep -E 'systemctl reboot|SELF-REBOOT' "$CALLS" | awk '{print $3,$4}' | tr '\n' ' ')
[[ "$(grep -nE 'SSH (worker-1|worker-2) :: sudo -n systemctl reboot' "$CALLS" | tail -1 | cut -d: -f1)" -lt "$(grep -n SELF-REBOOT "$CALLS" | cut -d: -f1)" ]]; t "S3b ORDER: the peers first, THEN this node" $? "$ORDER"
[[ -s $WATCHDOG_STATE/reboots && -s $WATCHDOG_STATE/last-event.txt && -d $(ls -d $WATCHDOG_STATE/evidence-* | head -1) ]]; t "S3c stamp + last-event + evidence directory written" $?
[[ "$(cat $WATCHDOG_STATE/fails)" == 0 ]]; t "S3d fails reset after the action" $?
n=$(count 'SSH .* :: true'); [[ "$n" -ge 2 ]]; t "S3e before acting the peers were probed with 'true'" $?

echo "=== S4 rate limit: 1 in 30 min / 3 in 6 h ==="
BASE=$(date +%s); export WATCHDOG_NOW=$BASE
echo "$BASE" > $WATCHDOG_STATE/reboots; rm -f $WATCHDOG_STATE/ALARM
export WATCHDOG_NOW=$((BASE+600)); : > "$CALLS"; round; round; round
[[ -e $WATCHDOG_STATE/ALARM && "$(count 'systemctl reboot')" == 0 && "$(count SELF-REBOOT)" == 0 ]]; t "S4a second action after 10 min: ALARM written, NO reboot" $? "$(cat $CALLS)"
grep -q "rate limit" $WATCHDOG_STATE/ALARM; t "S4b the alarm file gives the reason" $?
export WATCHDOG_NOW=$((BASE+700)); round; round; round
[[ "$(count SELF-REBOOT)" == 0 ]]; t "S4c while the ALARM exists later rounds do NOT act" $?
rm -f $WATCHDOG_STATE/ALARM; export WATCHDOG_NOW=$((BASE+2000)); round; round; round
[[ "$(count SELF-REBOOT)" == 1 ]]; t "S4d after 33 min (alarm deleted) the second reboot is FREE" $? "$(count SELF-REBOOT)"
export WATCHDOG_NOW=$((BASE+4000)); : > "$CALLS"; round; round; round
[[ "$(count SELF-REBOOT)" == 1 ]]; t "S4e 67 min: the third reboot is free" $?
export WATCHDOG_NOW=$((BASE+6200)); : > "$CALLS"; round; round; round
[[ -e $WATCHDOG_STATE/ALARM && "$(count SELF-REBOOT)" == 0 ]] && grep -q "6 h" $WATCHDOG_STATE/ALARM; t "S4f a fourth action inside 6 h: ALARM ('6 h'), NO reboot" $? "$(head -3 $WATCHDOG_STATE/ALARM)"
export WATCHDOG_NOW=$((BASE+6200+21600+10)); rm -f $WATCHDOG_STATE/ALARM; : > "$CALLS"; round; round; round
[[ "$(count SELF-REBOOT)" == 1 ]]; t "S4g after 6 h the window has emptied: free again" $?

echo "=== S5 busy-queue exemption ==="
fresh s5; engine '{"health":200,"chat_delay":6,"metrics_move":true}'
round; round; round; round
[[ "$(cat $WATCHDOG_STATE/fails)" == 0 && "$(count 'systemctl reboot')" == 0 ]] && grep -q "engine busy" $WATCHDOG_STATE/events.log; t "S5a probe does not return BUT the counter advances: 'busy', NOT counted as a hang" $?
export WATCHDOG_BUSY_MAX=3; fresh s5b; engine '{"health":200,"chat_delay":6,"metrics_move":true}'
for i in 1 2 3 4 5 6; do round; done
[[ "$(count SELF-REBOOT)" -ge 1 || -e $WATCHDOG_STATE/ALARM ]] && grep -q "exemption" $WATCHDOG_STATE/events.log; t "S5b the exemption used BUSY_MAX times in a row counts as a hang (no endless masking)" $?
unset WATCHDOG_BUSY_MAX
fresh s5c; engine '{"health":200,"chat_delay":6,"metrics_move":true}'; export WATCHDOG_EXEMPTION=0
round; round; round; [[ "$(count SELF-REBOOT)" == 1 ]]; t "S5c WATCHDOG_EXEMPTION=0: a hang counts even if the counter advances (the pure rule)" $?
unset WATCHDOG_EXEMPTION

echo "=== S6 health 503 ==="
fresh s6; engine '{"health":503}'; round; round; round
[[ "$(count SELF-REBOOT)" == 1 ]] && grep -q "health http=503" $WATCHDOG_STATE/events.log; t "S6 /health 503 x3 -> action" $?

echo "=== S7 pause file ==="
fresh s7; engine '{"health":503}'; touch "$W/pause"; round; round; round; round
[[ "$(count SELF-REBOOT)" == 0 && ! -e $WATCHDOG_STATE/fails ]]; t "S7 while the pause file exists NOTHING is done" $?

echo "=== S8 immunity: inactive, boot, young container ==="
fresh s8; engine '{"health":503}'; WATCHDOG_UNIT_STATE=inactive round; WATCHDOG_UNIT_STATE=inactive round; WATCHDOG_UNIT_STATE=inactive round; WATCHDOG_UNIT_STATE=inactive round
[[ "$(count SELF-REBOOT)" == 0 ]]; t "S8a unit inactive: it does not touch anything" $?
WATCHDOG_UNIT_STATE=activating round; WATCHDOG_UPTIME_S=300 round; WATCHDOG_UPTIME_S=300 round; WATCHDOG_UPTIME_S=300 round; WATCHDOG_UPTIME_S=300 round
[[ "$(count SELF-REBOOT)" == 0 ]]; t "S8b first 15 min (uptime 300 s): it does not touch anything" $?
for i in 1 2 3 4; do WATCHDOG_CONT_AGE_S=400 round; done
[[ "$(count SELF-REBOOT)" == 0 ]]; t "S8c container 400 s old: it does not touch anything" $?
WATCHDOG_UNIT_STATE=failed round; WATCHDOG_UNIT_STATE=failed round; WATCHDOG_UNIT_STATE=failed round
[[ "$(count SELF-REBOOT)" == 1 ]]; t "S8d unit 'failed': counts as a failure" $?

echo "=== S9 a peer is unreachable ==="
fresh s9; engine '{"health":503}'; touch "$W/unreachable-worker-1"; round; round; round
[[ -e $WATCHDOG_STATE/ALARM && "$(count 'systemctl reboot')" == 0 && "$(count SELF-REBOOT)" == 0 ]] && grep -q "worker-1 is not reachable over ssh" $WATCHDOG_STATE/ALARM; t "S9 unreachable peer: NO reboot, ALARM written" $?

echo "=== S10 probe 4xx and recovery ==="
fresh s10; engine '{"health":200,"chat_code":400}'; round; round; round
[[ "$(count SELF-REBOOT)" == 1 ]]; t "S10a probe HTTP 400 counts as a hang (a wrong model name is caught by the dry run before enabling)" $?
fresh s10b; engine '{"health":200,"chat_delay":6}'; round; round; engine '{"health":200,"chat_delay":0}'; round
[[ "$(cat $WATCHDOG_STATE/fails)" == 0 && "$(count SELF-REBOOT)" == 0 ]]; t "S10b after 2 failures a recovery resets the counter" $?
fresh s10c; engine '{"health":200,"chat_delay":6}'; round; engine '{"health":200,"chat_delay":0}'; round; engine '{"health":200,"chat_delay":6}'; round; round
[[ "$(count SELF-REBOOT)" == 0 ]]; t "S10c failures that are not CONSECUTIVE trigger nothing" $?

echo "=== S11 two simultaneous runs (flock) ==="
fresh s11; engine '{"health":200,"chat_delay":4}'
( bash "$B" > /dev/null 2>&1 & bash "$B" > /dev/null 2>&1 & wait ); sleep 1
[[ "$(cat $WATCHDOG_STATE/fails 2>/dev/null || echo 0)" -le 1 ]]; t "S11 two simultaneous runs did not collide (counter <= 1)" $?

echo
echo "TOTAL: PASS $PASS / FAIL $FAIL"
[[ $FAIL -eq 0 ]]
