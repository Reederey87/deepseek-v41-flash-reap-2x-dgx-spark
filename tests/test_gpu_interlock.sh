#!/usr/bin/env bash
# GPU-exclusivity interlock tests.
#
# The interlock runs `docker inspect` on the head and, over SSH, on the worker.
# OpenSSH does not forward argv: it joins the arguments with spaces and the
# remote login shell re-parses the result. A Go template that contains a space
# (`{{json .HostConfig.DeviceRequests}}`) therefore splits in transit and
# `docker inspect -f` fails. An earlier revision of this kit passed argv
# straight through a mock that used `exec "$@"`, which hid the defect; the ssh
# stub below joins and re-parses like the real thing.
#
# Run: bash tests/test_gpu_interlock.sh
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_SH="$HERE/../start.sh"

# `start.sh` is the host-side launcher and is not copied into the image, so
# skip cleanly there rather than reporting a failure.
if [ ! -f "$START_SH" ]; then
  echo "start.sh not present (container layout); gpu interlock tests skipped"
  exit 0
fi

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$TMP/bin"

# docker stub: rejects a malformed template exactly as real docker does.
cat > "$TMP/bin/docker" <<'EOS'
#!/usr/bin/env bash
set -u
[ "${MOCK_DOCKER_DOWN:-0}" = "1" ] && exit 1
[ "${1:-}" = "ps" ] && { echo "${MOCK_IDS:-}"; exit 0; }
if [ "${1:-}" = "inspect" ]; then
  [ "${2:-}" != "-f" ] && { echo "docker: inspect without -f" >&2; exit 1; }
  fmt="$3"; id="${4:-}"
  case "$fmt" in
    '{{json .HostConfig.DeviceRequests}}'|'{{.Name}}') ;;
    *) echo "docker: template parsing error: $fmt" >&2; exit 1 ;;
  esac
  [ -n "$id" ] || { echo "docker: requires a container" >&2; exit 1; }
  [ "${MOCK_INSPECT_FAIL_ID:-}" = "$id" ] && exit 1
  case "$fmt" in
    *DeviceRequests*) eval "echo \"\${MOCK_DR_${id}:-null}\"" ;;
    *Name*)           eval "echo \"\${MOCK_NAME_${id}:-/$id}\"" ;;
  esac
  exit 0
fi
exit 0
EOS
chmod +x "$TMP/bin/docker"

# ssh stub: join the command args with spaces, then hand the result to a
# remote shell. This is what OpenSSH does and what the original bug missed.
cat > "$TMP/bin/ssh" <<'EOS'
#!/usr/bin/env bash
set -u
while [ $# -gt 0 ]; do
  case "$1" in
    -o) shift 2 ;;
    *)  break ;;
  esac
done
host="$1"; shift
export MOCK_IDS="${MOCK_WORKER_IDS:-}"
remote_cmd="$*"
exec sh -c "$remote_cmd"
EOS
chmod +x "$TMP/bin/ssh"

{
  echo "PATH=\"$TMP/bin:\$PATH\""
  echo 'die() { echo "REFUSED: $*" >&2; exit 9; }'
  echo 'HEAD_NAME=dsv41-reap-head; WORKER_NAME=dsv41-reap-worker'
  echo 'WORKER_SSH=nvidia@spark2; REFUSE_IF_GPU_BUSY=1'
  awk '/^docker_probe\(\)/,/^}/' "$START_SH"
  awk '/^gpu_container_names\(\)/,/^}/' "$START_SH"
  awk '/^refuse_busy_gpu\(\)/,/^}/' "$START_SH"
  echo 'refuse_busy_gpu && echo ALLOWED'
} > "$TMP/run.sh"

fails=0
expect() {
  local want="$1" label="$2"
  shift 2
  local out
  out="$(env "$@" bash "$TMP/run.sh" 2>&1)"
  local got=REFUSED
  case "$out" in *ALLOWED*) got=ALLOWED ;; esac
  if [ "$got" = "$want" ]; then
    printf 'ok   %-44s %s\n' "$label" "$got"
  else
    printf 'FAIL %-44s want %s got %s\n' "$label" "$want" "$got"
    printf '     output: %s\n' "$out"
    fails=$((fails + 1))
  fi
}

# Worker probe must survive the SSH reparse.
expect ALLOWED "cpu-only container on worker" \
  MOCK_IDS= MOCK_WORKER_IDS=w2 MOCK_DR_w2=null MOCK_NAME_w2=/cpu-thing
expect ALLOWED "own candidate on worker" \
  MOCK_IDS= MOCK_WORKER_IDS=w1 MOCK_DR_w1='[{"Driver":"nvidia"}]' MOCK_NAME_w1=/dsv41-reap-worker
expect REFUSED "foreign GPU container on worker" \
  MOCK_IDS= MOCK_WORKER_IDS=g1 MOCK_DR_g1='[{"Driver":"nvidia"}]' MOCK_NAME_g1=/glm53-exl3-worker
expect REFUSED "worker inspect failure" \
  MOCK_IDS= MOCK_WORKER_IDS=w1 MOCK_DR_w1='[{"Driver":"nvidia"}]' MOCK_NAME_w1=/dsv41-reap-worker \
  MOCK_INSPECT_FAIL_ID=w1

# Head probe (no SSH).
expect ALLOWED "own candidate on head" \
  MOCK_IDS=c1 MOCK_DR_c1='[{"Driver":"nvidia"}]' MOCK_NAME_c1=/dsv41-reap-head
expect ALLOWED "cpu-only container on head" \
  MOCK_IDS=c2 MOCK_DR_c2=null MOCK_NAME_c2=/cpu-thing
expect REFUSED "GLM container on head" \
  MOCK_IDS=g1 MOCK_DR_g1='[{"Driver":"nvidia"}]' MOCK_NAME_g1=/glm53-exl3-head
expect REFUSED "unknown GPU container on head" \
  MOCK_IDS=u1 MOCK_DR_u1='[{"Driver":"nvidia"}]' MOCK_NAME_u1=/other-gpu
expect REFUSED "head docker unreachable" MOCK_IDS= MOCK_DOCKER_DOWN=1

# Negative guard: the unquoted argv form must still split, proving the stub
# really reparses (so a passing suite above is meaningful, not vacuous).
probe_out="$(env PATH="$TMP/bin:$PATH" MOCK_IDS= MOCK_WORKER_IDS=w1 \
  MOCK_DR_w1=null bash -c 'ssh -o BatchMode=yes nvidia@spark2 docker inspect -f {{json .HostConfig.DeviceRequests}} w1' 2>&1)"
probe_rc=$?
if [ "$probe_rc" -ne 0 ] && printf '%s' "$probe_out" | grep -q 'template parsing error: {{json'; then
  printf 'ok   %-44s %s\n' "unsplit argv form splits (stub reparses)" "exit=$probe_rc"
else
  printf 'FAIL %-44s exit=%s out=%s\n' "unsplit argv form must split" "$probe_rc" "$probe_out"
  fails=$((fails + 1))
fi

# And the serialized form must survive the same reparse.
probe_out="$(env PATH="$TMP/bin:$PATH" MOCK_IDS= MOCK_WORKER_IDS=w1 \
  MOCK_DR_w1=null bash -c 'ssh -o BatchMode=yes nvidia@spark2 "$(printf "%q " docker inspect -f "{{json .HostConfig.DeviceRequests}}" w1)"' 2>&1)"
if [ "$probe_out" = "null" ]; then
  printf 'ok   %-44s %s\n' "serialized form arrives intact" "$probe_out"
else
  printf 'FAIL %-44s out=%s\n' "serialized form arrives intact" "$probe_out"
  fails=$((fails + 1))
fi

echo
if [ "$fails" -eq 0 ]; then
  echo "gpu interlock tests OK"
else
  echo "gpu interlock tests FAILED ($fails)"
  exit 1
fi
