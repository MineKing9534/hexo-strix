#!/usr/bin/env bash
set -euo pipefail

candidate="runs/klent/axis-gine/s4-from-d6-qhead-tau16-alpha001-noclip-from10/checkpoints/checkpoint_000020.pt"
artifact_dir="artifacts/klent-head-to-head"

run_match() {
    local slug="$1"
    local opponent="$2"
    local seed="$3"
    local state_file="${artifact_dir}/cpu-forcing-off-unclipped20-vs-${slug}-20260806.json"
    local log_file="${artifact_dir}/cpu-forcing-off-unclipped20-vs-${slug}-20260806.log"

    if [[ -f "$state_file" ]] && [[ "$(jq -r '.decision // "continue"' "$state_file")" != "continue" ]]; then
        echo "[skip] ${slug}: terminal state already exists"
        return
    fi

    echo "[start] forcing-off unclipped generation 20 vs ${slug}"
    nice -n 10 script -q -f -e -c \
        ".venv/bin/hexo-a0 head-to-head --checkpoint-a ${candidate} --checkpoint-b ${opponent} --win-length 6 --radius 8 --max-moves 1000 --mcts-sims 64 --mcts-m-actions 16 --device cpu --sprt-s0 0.50 --sprt-s1 0.55 --sprt-alpha 0.05 --sprt-beta 0.05 --window-size 0 --max-games 1000 --seed ${seed} --state-file ${state_file} --opening-plies 8 --opening-generator alternate --opening-temperature 0.5 --disable-forcing-solver" \
        "$log_file"
    echo "[done] ${slug}"
}

run_match \
    "strix-d6-qhead-215547" \
    "runs/gine-mini/4l-128p32v-lean-d6-qhead/checkpoints/checkpoint_00215547.pt" \
    20260861

run_match \
    "alpha001-clipped20" \
    "runs/klent/axis-gine/s4-from-d6-qhead-tau16-alpha001-from10/checkpoints/checkpoint_000020.pt" \
    20260862

run_match \
    "alpha003-tau16-20" \
    "runs/klent/axis-gine/s4-from-d6-qhead-tau16-from10/checkpoints/checkpoint_000020.pt" \
    20260863

run_match \
    "strix-rel2-full-237000" \
    "runs/gine-mini/4l-128p32v-jkcat-rel2/checkpoints/checkpoint_00237000.pt" \
    20260864

echo "[complete] all forcing-off unclipped generation-20 CPU matches finished"
