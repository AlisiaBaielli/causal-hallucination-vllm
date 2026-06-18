#!/usr/bin/env bash
# Submit the full thesis reproduction grid as SLURM jobs:
#   - hallucination: {llava,qwen3,internvl} x {chair,pope,amber,mme}   (12 jobs)
#   - capability:    {llava,qwen3} x {mmvp,mmbench}                    (4 jobs)
#   - quality:       {llava,qwen3,internvl} METEOR+CLIPScore          (3 jobs, after chair)
#
#   bash scripts/reproduce/submit_all.sh                # submit everything
#   MODELS="llava" BENCHES="chair pope" bash .../submit_all.sh   # subset
#   SKIP_EXISTING=1 bash .../submit_all.sh              # resume
#   DRY=1 bash .../submit_all.sh                        # print, do not submit
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

MODELS="${MODELS:-llava qwen3 internvl}"
BENCHES="${BENCHES:-chair pope amber mme}"
CAP_BENCHES="${CAP_BENCHES:-mmvp mmbench}"
SKIP_EXISTING="${SKIP_EXISTING:-0}"
PART="${PART:-gpu_a100}"
JOB="scripts/reproduce/_job.sh"
DRY="${DRY:-0}"

declare -A TIME=( [chair]=10:00:00 [pope]=12:00:00 [amber]=10:00:00 [mme]=12:00:00
                  [mmvp]=04:00:00 [mmbench]=24:00:00 [quality]=02:00:00 )

submit() {  # name kind model bench [extra sbatch args...]
  local name="$1" kind="$2" model="$3" bench="$4"; shift 4
  local t="${TIME[${bench:-quality}]:-10:00:00}"
  local args=(--partition="${PART}" --job-name="${name}" --time="${t}"
              --export="ALL,RKIND=${kind},RMODEL=${model},RBENCH=${bench},SKIP_EXISTING=${SKIP_EXISTING}"
              "$@" "${JOB}")
  if [[ "${DRY}" == "1" ]]; then echo "sbatch ${args[*]}"; echo "DRYRUN_${name}"; return 0; fi
  sbatch "${args[@]}"
}

declare -A CHAIR_JOBID
for model in ${MODELS}; do
  for bench in ${BENCHES}; do
    out="$(submit "repro_${model}_${bench}" bench "${model}" "${bench}")"
    echo "${out}"
    jid="$(echo "${out}" | grep -oE '[0-9]+' | tail -1)"
    [[ "${bench}" == "chair" ]] && CHAIR_JOBID[${model}]="${jid}"
  done
done

# capability (LLaVA + Qwen3 only)
for model in ${MODELS}; do
  [[ "${model}" == "internvl" ]] && continue
  for bench in ${CAP_BENCHES}; do
    submit "repro_${model}_${bench}" bench "${model}" "${bench}"
  done
done

# quality after each model's chair job
for model in ${MODELS}; do
  dep=()
  jid="${CHAIR_JOBID[${model}]:-}"
  [[ -n "${jid}" ]] && dep=(--dependency="afterok:${jid}")
  submit "repro_${model}_quality" quality "${model}" quality "${dep[@]}"
done

echo "Submitted. Monitor: squeue -u \$USER"
