# ops/watchdog -- optional engine watchdog for the tp3-main track

Optional. The recipe runs without it; the autostart unit brings the engine up at boot and nothing
restarts it afterwards (the container runs with `--restart no`, on purpose: a rank that quietly
retries on its own is the "fluent and wrong" failure class this stack is built to refuse). This
watchdog is the answer to the other half: **an engine that dies or hangs while nobody is looking**.

It runs on the **head** node once a minute, probes the API, and after three failures in a row reboots
**all three nodes** in the safe order. Why all three, and why in that order:
[docs/00-hardware-and-os.md](../../../../docs/00-hardware-and-os.md) sections 3.1 and 3.4 (a node that
reboots alone can leave link pairs on the others down; both ends of a link must come up together).

| File | What it is |
|---|---|
| [`watchdog.sh`](watchdog.sh) | the script; read its header, it is the specification |
| [`harem-watchdog.service`](harem-watchdog.service) / [`.timer`](harem-watchdog.timer) | the one-shot unit and the one-minute timer |

## What it does, in six lines

1. **A healthy `/health` is not enough.** It sends a completion of eight tokens and gives it 60 s.
   The failure it was written for is the silent one: `/health` answers 200 and requests never return.
2. **Busy is not hung.** If the probe did not return but the token counters in `/metrics` moved while
   it waited, the engine is busy (five seats full, the probe queued). At most ten rounds in a row
   count that way; frozen counters are a hang at once. `WATCHDOG_EXEMPTION=0` removes the exemption.
3. **Three consecutive failures** are needed before it acts. One recovered round resets the count.
4. **The action** is a triple reboot: evidence first (`docker logs --tail 300` from all three nodes,
   into `$STATE/evidence-<time>/`), then `sudo -n systemctl reboot` on both workers over ssh, three
   seconds later on the head.
5. **A rate limit.** One reboot in 30 minutes, three in six hours. Beyond that it stops, writes
   `$STATE/ALARM` and the journal gets a `user.crit` line; it does nothing more until a person deletes
   the file. A reboot loop on a cluster that is genuinely broken is worse than an engine that stays down.
6. **It will not do half the job.** If a worker is unreachable over ssh it reboots nothing and writes
   the alarm: one reboot would leave the cluster in the state the whole-cluster rule exists to avoid.
   It also does nothing in the first 15 minutes after boot, while the container is younger than 25
   minutes, or while the unit is `inactive` (stopped on purpose). A `failed` unit counts as a failure.

## Install (head only)

The head's user needs: key-based ssh to both workers, and passwordless sudo for `systemctl reboot` on
all three nodes and for `docker logs`/`docker inspect`/`docker ps` on each (or set
`WATCHDOG_DOCKER_LOGS` to a command that works for you).

```
mkdir -p ~/exl3-main/watchdog && cp ops/watchdog/watchdog.sh ~/exl3-main/watchdog/
```

```
sed "s#@USER@#$USER#; s#@HOME@#$HOME#g" ops/watchdog/harem-watchdog.service | sudo tee /etc/systemd/system/harem-watchdog.service
```

```
sudo cp ops/watchdog/harem-watchdog.timer /etc/systemd/system/ && sudo systemctl daemon-reload
```

**Before you enable it, run it in dry mode against the live engine:**

```
bash ~/exl3-main/watchdog/watchdog.sh --dry-run
```

It prints what it would have done and touches nothing (its state files get a `.dry` suffix). Then:

```
sudo systemctl enable --now harem-watchdog.timer
```

The probe uses the model name `WATCHDOG_MODEL` (default `glm-5.3-flash`, the `SERVED_MODEL_NAME` of the
env example). A **wrong model name makes every probe fail** with HTTP 404, and after three rounds the
watchdog would reboot a healthy cluster: that is why the dry run comes first. Other knobs are the
`WATCHDOG_*` variables at the top of the script (peers, probe timeout, grace periods, rate limits).

## Maintenance, and getting out of an alarm

| You want to | Do |
|---|---|
| work on the cluster without it acting | `touch /var/tmp/harem-watchdog-pause`; remove the file afterwards |
| see what it decided | `tail $STATE/events.log` (default `$STATE` is `/var/tmp/harem-watchdog`), or `journalctl -t harem-watchdog` |
| see its heartbeat | `cat $STATE/last-round.txt` (written every round) |
| re-arm it after an alarm | find out why the engine keeps falling (the `evidence-*` directories), then `rm $STATE/ALARM` |

**Nothing reads the alarm file for you.** If you want to be told, point your own monitoring at
`$STATE/ALARM` or at the `user.crit` journal line.

## What was tested, and what was not

**A mock suite**, run against a mock engine, a mock `ssh` and a mock reboot command: 33 checks in eleven
scenarios (healthy; a silent hang in dry mode; the same hang in real mode with the order of the reboots; the
two rate limits and the alarm; the busy exemption and its limits; health 503; the pause file; boot,
`inactive` and young-container immunity; an unreachable peer; probe errors and recovery; two simultaneous
runs). It is in [`test/`](test/) and touches no real machine; run `bash test/run-tests.sh` (about a minute and
a half). Three deliberate breakages of this script were each caught by it `[measured-here]`: the rate limit
switched off fails 5 checks, the order of the reboots reversed fails 1, the dry-run guard removed fails 2.
To repeat one, break a copy of `watchdog.sh` and run `WATCHDOG_SCRIPT=<the copy> bash test/run-tests.sh`.

**One real drill.** The engine container of one worker was killed with `docker kill`; the watchdog counted
three failed rounds, collected evidence, rebooted both workers and then the head, and the units brought the
engine back by themselves, `/health` 200 **8.2 minutes after the kill**, no alarm
[`results/main-stack/boot-and-watchdog.md`](../../../../results/main-stack/boot-and-watchdog.md) section 4
`[measured-here]`.

**Not tested:** the silent-hang path (`/health` 200, the probe times out) against a live hang, the rate-limit
and alarm path, and the unreachable-peer refusal were exercised only in the mock suite `[not tested]` on the
real cluster. The watchdog lives on the head: if the head itself dies, nothing watches `[not tested]`.
