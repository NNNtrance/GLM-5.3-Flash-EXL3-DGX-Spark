#!/usr/bin/env bash
# build.sh -- build the tp3-main engine image on ONE aarch64 DGX Spark node.
#
#   official vLLM nightly 21d93d0d8 (arm64, pinned by digest)
#   + the cuda-exl3 fork at 448f1d6, compiled for sm_121a
#   + the 14 anchored patches (patches/), applied by the prelude in its dry-run mode
#   = harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6
#
# It uses no GPU. It needs the nightly image already pulled (README step 1), `git`, and
# about 40 GB of RAM for the container (BUILD_MEM) for the kernel compile; `MAX_JOBS` 8 took
# a few minutes on a GB10. Run it once and copy the image to the other two nodes with
# `docker save | docker load`: builds of the same source are not bit-identical (docs/14
# section 7.10), the preflight compares the image Id, and all three nodes must carry the SAME
# image.
#
# Fail-closed: patches that do not match SHA256SUMS, a nightly that is not the pinned one, a
# tag that already exists, a fork checkout that is not the pinned commit, a patch that does
# not apply, a kernel binary built for anything but sm_121a, or a failed DFlash2 gate all
# stop the build with a message; nothing is tagged until every check has passed.
#
# Environment (all optional):
#   DOCKER      the docker command, e.g. "sudo docker"        (default: docker)
#   TAG         the image tag                                  (default below; the tag enters the
#                                                               fast-load identity, so do not
#                                                               change it between nodes)
#   WORK        a scratch directory for the fork checkout and the build record
#   FORK_URL    where to clone the fork from                   (default: the public repository)
#   BUILD_MEM, BUILD_CPUS, MAX_JOBS     container limits for the compile
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
DOCKER=${DOCKER:-docker}
BASE_DIGEST=sha256:6f0d5e677145fb4003891a994f11fed993d022d141c71844157d3e84e55b1e0a
BASE=vllm/vllm-openai@$BASE_DIGEST
VLLM_COMMIT=21d93d0d8c0e9627900020382bfce4730e61cab7
FORK_URL=${FORK_URL:-https://github.com/NNNtrance/cuda-exl3}
FORK_COMMIT=448f1d6b8964b96e7d4e299494538bfc240149ef
TAG=${TAG:-harem/vllm-exl3-tasima:a4i-21d93d0d8-448f1d6}
WORK=${WORK:-$HOME/exl3-main-build}
BUILD_MEM=${BUILD_MEM:-40g}
BUILD_CPUS=${BUILD_CPUS:-16}
MAX_JOBS=${MAX_JOBS:-8}
OUT=$WORK/build-$(date +%Y%m%d-%H%M%S)
mkdir -p "$OUT"
log() { echo "[$(date +%T)] $*" | tee -a "$OUT/build.log"; }
red() { log "REFUSED: $*"; echo "FAILED: $*" > "$OUT/RESULT"; exit 1; }

# --- 0. the host ---------------------------------------------------------------------------
[[ "$(uname -m)" == "aarch64" ]] || red "this build is for aarch64 (a DGX Spark); this host is $(uname -m)"
$DOCKER info > /dev/null 2>&1 || red "'$DOCKER info' failed: is docker running, and may this user use it?"
command -v git > /dev/null || red "git is not installed"
( cd "$HERE/patches" && sha256sum -c --quiet SHA256SUMS ) || red "patches/ does not match patches/SHA256SUMS"

# --- 1. the base image: pinned by digest, not by tag ------------------------------------------
id=$($DOCKER image inspect "$BASE" --format '{{.Id}} {{range .RepoDigests}}{{.}} {{end}}' 2>/dev/null) \
  || red "the base image is not present: $DOCKER pull $BASE"
[[ " $id " == *"$BASE_DIGEST"* ]] || red "the image at $BASE does not carry the digest $BASE_DIGEST (got: $id)"
$DOCKER image inspect "$TAG" > /dev/null 2>&1 && red "$TAG already exists (a build never overwrites a tag; remove it deliberately first)"
for c in exl3-main-build; do
  $DOCKER ps -a --format '{{.Names}}' | grep -qx "$c" && red "a container named $c already exists"
done
log "start: base=$BASE tag=$TAG out=$OUT"

# --- 2. the fork, at the pinned commit --------------------------------------------------------
mkdir -p "$WORK"
if [[ ! -d "$WORK/cuda-exl3/.git" ]]; then
  git clone --quiet "$FORK_URL" "$WORK/cuda-exl3" || red "git clone $FORK_URL failed"
fi
git -C "$WORK/cuda-exl3" cat-file -e "$FORK_COMMIT^{commit}" 2> /dev/null || git -C "$WORK/cuda-exl3" fetch --quiet origin "$FORK_COMMIT" \
  || red "commit $FORK_COMMIT is not in $WORK/cuda-exl3 and could not be fetched"
[[ "$(git -C "$WORK/cuda-exl3" rev-parse "$FORK_COMMIT^{commit}")" == "$FORK_COMMIT" ]] || red "the fork does not resolve $FORK_COMMIT to itself"
# Export exactly the commit's tree (no .git, no untracked files).
SRC=$OUT/cuda-exl3-src
mkdir -p "$SRC" && git -C "$WORK/cuda-exl3" archive "$FORK_COMMIT" | tar -x -C "$SRC" || red "git archive failed"
echo "$FORK_COMMIT" > "$SRC/COMMIT"
log "fork: $FORK_URL @ $FORK_COMMIT exported to $SRC"

# --- 3. the inside of the build container -----------------------------------------------------
cat > "$OUT/inner.sh" <<'EOF'
#!/usr/bin/env bash
set -uo pipefail
export PYTHONDONTWRITEBYTECODE=1
VP=/usr/local/lib/python3.12/dist-packages/vllm
PATCHES=/opt/track/patches
[[ "${VLLM_BUILD_COMMIT:-}" == "21d93d0d8c0e9627900020382bfce4730e61cab7" ]] || { echo "INNER-FAIL: VLLM_BUILD_COMMIT"; exit 80; }
python3 - <<'PY' || exit 81
import platform, torch, vllm
assert vllm.__version__.endswith("+g21d93d0d8"), vllm.__version__
assert torch.__version__ == "2.13.0+cu130", torch.__version__
assert platform.machine() == "aarch64", platform.machine()
print("environment ok:", vllm.__version__, torch.__version__, platform.machine())
PY
python3 -c "import cuda_exl3" 2>/dev/null && { echo "INNER-FAIL: the base image already has cuda_exl3"; exit 82; }
# The image's vLLM files must be exactly the files the patches were written against (their pinned base sha256).
python3 - <<'PY' || exit 83
import hashlib, sys
sys.path.insert(0, "/opt/track/patches")
import harem_yama as Y
from pathlib import Path
t = Y.kardes_yamalar(Path("/opt/track/patches"))
pin = {f: s for y in t.values() for f, s in y.taban.items()}
for f, s in sorted(pin.items()):
    got = hashlib.sha256(open(f"/usr/local/lib/python3.12/dist-packages/vllm/{f}", "rb").read()).hexdigest()
    assert got == s, f"{f} in the image differs from the pinned base"
print(f"image == upstream 21d93d0d8: {len(pin)} files match their pinned base sha256")
PY
# --- the plugin --------------------------------------------------------------------------------
rm -rf /tmp/plugin && cp -a /src/cuda-exl3 /tmp/plugin
export TORCH_CUDA_ARCH_LIST=12.1a MAX_JOBS="${MAX_JOBS:-8}"
( time pip install --no-build-isolation --no-deps -v /tmp/plugin ) > /out/pip-plugin.log 2>&1 || { echo "INNER-FAIL: plugin build"; tail -30 /out/pip-plugin.log; exit 84; }
python3 - <<'PY' || exit 84
import hashlib, os, subprocess, cuda_exl3
src = "/tmp/plugin/src/cuda_exl3"; dst = os.path.dirname(cuda_exl3.__file__)
diff = [f for f in sorted(os.listdir(src)) if f.endswith(".py") and
        hashlib.sha256(open(os.path.join(src, f), "rb").read()).digest()
        != hashlib.sha256(open(os.path.join(dst, f), "rb").read()).digest()]
assert not diff, diff
so = [f for f in os.listdir(dst) if f.startswith("_C") and f.endswith(".so")]
assert len(so) == 1, so
elf = subprocess.run(["cuobjdump", "--list-elf", os.path.join(dst, so[0])], capture_output=True, text=True).stdout
archs = sorted({ln.split(".")[-2] for ln in elf.split() if ln.endswith(".cubin")})
assert archs == ["sm_121a"], archs
print("plugin installed:", dst, so[0], archs)
PY
python3 - <<'PY' || exit 84
import os, cuda_exl3
d = os.path.dirname(cuda_exl3.__file__)
m = open(os.path.join(d, "moe.py")).read()
assert "routed experts clamp" in m and "_routed_swiglu_limit" in m, "the installed moe.py has no SwiGLU clamp"
assert "HAREM-MOE-KAPI" in m, "the installed moe.py has no boot gate (router bias / clamp)"
so = [f for f in os.listdir(d) if f.startswith("_C") and f.endswith(".so")][0]
b = open(os.path.join(d, so), "rb").read()
assert b"CUDA_EXL3_GEMM_TUNE_VERBOSE" in b and b"swiglu_limit" in b, "the .so lacks the GEMM tuner knobs / swiglu_limit"
print("plugin traces ok: SwiGLU clamp, [HAREM-MOE-KAPI] gate, GEMM tuner knobs")
PY
# --- the patches --------------------------------------------------------------------------------
rm -rf /opt/harem && mkdir -p /opt/harem/yamalar
( cd "$PATCHES" && awk '{print $2}' SHA256SUMS ) | while read -r f; do cp -p "$PATCHES/$f" /opt/harem/yamalar/; done
HAREM_PRELUDE_KURU=1 TP3_DIR=/opt/harem/yamalar bash /opt/harem/yamalar/harem-prelude.sh > /out/prelude-1.log 2>&1 || { echo "INNER-FAIL: prelude"; cat /out/prelude-1.log; exit 85; }
HAREM_PRELUDE_KURU=1 TP3_DIR=/opt/harem/yamalar bash /opt/harem/yamalar/harem-prelude.sh > /out/prelude-2.log 2>&1 || { echo "INNER-FAIL: prelude, second run"; exit 85; }
# "yazıldı" ("written") is what a patch script prints for every file it writes.
n_second=$(grep -c "yazıldı" /out/prelude-2.log); [[ "$n_second" == "0" ]] || { echo "INNER-FAIL: the second run wrote $n_second files"; exit 85; }
n_first=$(grep -c "yazıldı" /out/prelude-1.log); echo "prelude: first run wrote $n_first patch results, second run 0"
for y in /opt/harem/yamalar/yama-*.py; do python3 $y --root $VP --check > /tmp/c.txt 2>&1; grep -q "uygulanmis" /tmp/c.txt || { echo "INNER-FAIL: $y is not applied"; cat /tmp/c.txt; exit 85; }; done
n_y=$(ls /opt/harem/yamalar/yama-*.py | wc -l); [[ "$n_y" == "14" ]] || { echo "INNER-FAIL: $n_y patch scripts (expected 14)"; exit 85; }
echo "patches: $n_y/14 --check applied"
grep -q "harem_taslak_kapi.py" /opt/harem/yamalar/harem-prelude.sh || { echo "INNER-FAIL: the prelude does not call the draft-tag gate"; exit 88; }
grep -q "yazildi_defteri" $VP/model_executor/model_loader/harem_fastload.py || { echo "INNER-FAIL: fastload in the image has no written-parts ledger"; exit 88; }
echo "draft-tag gate: called by the prelude; fastload ledger: in the image"
cp /opt/track/tools/dflash2_build_gate.py /opt/harem/dflash2-gate.py
( cd /tmp && python3 /opt/harem/dflash2-gate.py ) > /out/gate.log 2>&1 || { echo "INNER-FAIL: DFlash2 gate"; tail -20 /out/gate.log; exit 87; }
grep -q "DFLASH2 BUILD GATE (21d93d0d8 + 14 yama): OK" /out/gate.log || { echo "INNER-FAIL: the gate did not print its OK line"; exit 87; }
echo "gate: $(tail -1 /out/gate.log)"
# --- the identity record, kept inside the image and next to the build log ---------------------------
python3 - <<'PY' || exit 86
import hashlib, json, os, platform, time, cuda_exl3, torch, vllm
def sha(p): return hashlib.sha256(open(p, "rb").read()).hexdigest()
d = os.path.dirname(cuda_exl3.__file__)
so = [f for f in os.listdir(d) if f.startswith("_C") and f.endswith(".so")][0]
y = "/opt/harem/yamalar"
k = {
  "base_image": os.environ.get("HAREM_BASE_IMAGE"),
  "vllm": vllm.__version__, "vllm_commit": os.environ.get("VLLM_BUILD_COMMIT"),
  "torch": torch.__version__, "machine": platform.machine(),
  "cuda_exl3_commit": open("/src/cuda-exl3/COMMIT").read().strip(),
  "cuda_exl3_arch": "12.1a", "cuda_exl3_so": so, "cuda_exl3_so_sha256": sha(os.path.join(d, so)),
  "cuda_exl3_py_sha256": {f: sha(os.path.join(d, f)) for f in sorted(os.listdir(d)) if f.endswith(".py")},
  "patches_sha256": {f: sha(os.path.join(y, f)) for f in sorted(os.listdir(y)) if os.path.isfile(os.path.join(y, f))},
  "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
}
json.dump(k, open("/opt/harem/IDENTITY.json", "w"), indent=1)
print("identity record written; .so", k["cuda_exl3_so_sha256"][:16])
PY
cp /opt/harem/IDENTITY.json /out/IDENTITY.json
rm -rf /tmp/plugin /root/.cache/pip
echo "INNER-DONE"
EOF
chmod +x "$OUT/inner.sh"

log "container 1/1: plugin build + patches + gates (this takes a while)"
$DOCKER run --name exl3-main-build --memory="$BUILD_MEM" --memory-swap="$BUILD_MEM" --cpus="$BUILD_CPUS" \
  -e HOME=/root -e MAX_JOBS="$MAX_JOBS" -e HAREM_BASE_IMAGE="$BASE" --entrypoint bash \
  -v "$HERE:/opt/track:ro" -v "$SRC:/src/cuda-exl3:ro" -v "$OUT:/out" \
  "$BASE" /out/inner.sh > "$OUT/inner.log" 2>&1
rc=$?
tail -12 "$OUT/inner.log" | tee -a "$OUT/build.log"
if [[ $rc -ne 0 ]] || ! grep -q "INNER-DONE" "$OUT/inner.log"; then
  $DOCKER rm exl3-main-build > /dev/null 2>&1
  red "the build container exited with $rc (see $OUT/inner.log)"
fi

# --- 4. commit: the engine entry point is `vllm serve`, as in the base image ----------------------
KS=$(sha256sum "$OUT/IDENTITY.json" | cut -c1-64)
$DOCKER commit \
  --change 'ENTRYPOINT ["vllm","serve"]' --change 'CMD []' --change 'WORKDIR /vllm-workspace' \
  --change "LABEL harem.stack=tp3-main harem.base=$BASE_DIGEST harem.cuda-exl3.commit=$FORK_COMMIT harem.cuda-exl3.arch=12.1a harem.identity.sha256=$KS" \
  exl3-main-build "$TAG" > "$OUT/commit.txt" 2>&1 || red "docker commit failed"
$DOCKER rm exl3-main-build > /dev/null
IMAGE_ID=$($DOCKER image inspect "$TAG" --format '{{.Id}}')
echo "$IMAGE_ID" > "$WORK/IMAGE_ID"
echo "PASSED" > "$OUT/RESULT"
log "image: $TAG"
log "image Id: $IMAGE_ID   (put this in IMG_ID of the env file; written to $WORK/IMAGE_ID)"
log "next: copy to the other nodes:  $DOCKER save $TAG | ssh <node> docker load   -- then check that 'docker image inspect $TAG --format {{.Id}}' is the same everywhere"
