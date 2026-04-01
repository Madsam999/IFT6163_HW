#!/bin/bash
# Run sim_eval.py on all 3 HW3 checkpoints
# Run from: /project/60004/samuel/IFT6163_HW


EVAL="python hw3/sim_eval.py"

echo "=== Dense PPO seed 0 ==="
$EVAL \
    +checkpoint=./hw3/miniGRP_Model/miniGRP.pth \
    simEval=[libero_fast] \
    testing=true \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.eval_episodes=20

echo "=== Dense PPO seed 2 ==="
$EVAL \
    +checkpoint=/outputs/hw3_dense_ppo_seed2/dense_ppo_final.pth \
    simEval=[libero_fast] \
    testing=true \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.eval_episodes=20

echo "=== Transformer GRPO-GT ==="
$EVAL \
    +checkpoint=/outputs/hw3_transformer_grpo/transformer_rl_final.pth \
    simEval=[libero_fast] \
    testing=true \
    sim.task_set=libero_spatial \
    sim.eval_tasks=[9] \
    sim.eval_episodes=20
