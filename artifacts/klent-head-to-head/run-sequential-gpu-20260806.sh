#!/usr/bin/env bash
set -euo pipefail

script -q -f -e -c \
    '.venv/bin/hexo-a0 head-to-head --checkpoint-a runs/klent/axis-gine/s4-from-d6-qhead-tau16-alpha001-from10/checkpoints/final.pt --checkpoint-b runs/klent/axis-gine/s4-from-d6-qhead-tau16-from10/checkpoints/final.pt --win-length 6 --radius 8 --max-moves 1000 --mcts-sims 64 --mcts-m-actions 16 --device cuda --sprt-s0 0.50 --sprt-s1 0.55 --sprt-alpha 0.05 --sprt-beta 0.05 --window-size 0 --max-games 1000 --seed 20260806 --state-file artifacts/klent-head-to-head/gpu-alpha001-vs-alpha003-tau16-20260806.json --opening-plies 8 --opening-generator alternate --opening-temperature 0.5' \
    artifacts/klent-head-to-head/gpu-alpha001-vs-alpha003-tau16-20260806.log

script -q -f -e -c \
    '.venv/bin/hexo-a0 head-to-head --checkpoint-a runs/klent/axis-gine/s4-from-d6-qhead-tau16-alpha001-from10/checkpoints/final.pt --checkpoint-b runs/klent/axis-gine/s4-from-d6-qhead-lr2e4-warmup5/checkpoints/final.pt --win-length 6 --radius 8 --max-moves 1000 --mcts-sims 64 --mcts-m-actions 16 --device cuda --sprt-s0 0.50 --sprt-s1 0.55 --sprt-alpha 0.05 --sprt-beta 0.05 --window-size 0 --max-games 1000 --seed 20260806 --state-file artifacts/klent-head-to-head/gpu-alpha001-vs-alpha003-tau8-20260806.json --opening-plies 8 --opening-generator alternate --opening-temperature 0.5' \
    artifacts/klent-head-to-head/gpu-alpha001-vs-alpha003-tau8-20260806.log

script -q -f -e -c \
    '.venv/bin/hexo-a0 head-to-head --checkpoint-a runs/klent/axis-gine/s4-from-d6-qhead-tau16-alpha001-from10/checkpoints/final.pt --checkpoint-b runs/gine-mini/4l-128p32v-lean-d6-qhead/checkpoints/checkpoint_00215547.pt --win-length 6 --radius 8 --max-moves 1000 --mcts-sims 64 --mcts-m-actions 16 --device cuda --sprt-s0 0.50 --sprt-s1 0.55 --sprt-alpha 0.05 --sprt-beta 0.05 --window-size 0 --max-games 1000 --seed 20260806 --state-file artifacts/klent-head-to-head/gpu-alpha001-vs-strix-d6-qhead-215547-20260806.json --opening-plies 8 --opening-generator alternate --opening-temperature 0.5' \
    artifacts/klent-head-to-head/gpu-alpha001-vs-strix-d6-qhead-215547-20260806.log
