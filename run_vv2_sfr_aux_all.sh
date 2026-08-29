#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./run_vv2_sfr_aux_all.sh
#   ./run_vv2_sfr_aux_all.sh output/my_sfr_aux_run
#
# This script keeps the run_vv2_corrmap.sh scene layout and enables the
# SFR auxiliary RL state. Most scene groups omit frequency-loss parameters;
# Deep Blending keeps the per-scene frequency setting used by the standalone runs.

OUT_ROOT="${1:-output/vv2_sfr_aux_$(date +%Y%m%d_%H%M%S)}"
PYTHON_CMD=(conda run --no-capture-output -n LeGS python)

APSR_MIP="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_MIP_PSNR="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.65 --apsr_edge_weight 0.08 --apsr_structure_weight 0.10 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_TNT_TRAIN_NEW="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_TNT="--lambda_apsr 0.38 --apsr_hard_mse_weight 0.6 --apsr_edge_weight 0.08 --apsr_structure_weight 0.10 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_TNT_DENSITY_WEAK="--lambda_apsr 0.0 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"
APSR_DB="--lambda_apsr 0.32 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.03 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"
APSR_DB_FREQ="--lambda_apsr 0.32 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.03 --apsr_color_weight 0.01 --apsr_pyramid_levels 2 --frequency_weight 0.05 --frequency_kernel_size 3 --frequency_band_weight 0.75 --frequency_high_ratio 0.35"

DENSITY_DEFAULT="--use_apsr_density_control 1 --lambda_apsr_density 0.20 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"
DENSITY_TNT_TRAIN_NEW="--use_apsr_density_control 1 --lambda_apsr_density 0.20 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_prune_scale 1.0"
DENSITY_DEFAULT_PRUNE095="--use_apsr_density_control 1 --lambda_apsr_density 0.20 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 0.95 --apsr_density_log_interval 100"
DENSITY_CONSERVATIVE="--use_apsr_density_control 1 --lambda_apsr_density 0.10 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.25 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"
DENSITY_CONSERVATIVE_PRUNE095="--use_apsr_density_control 1 --lambda_apsr_density 0.10 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.25 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 0.95 --apsr_density_log_interval 100"
DENSITY_TRUCK="--use_apsr_density_control 1 --lambda_apsr_density 0.30 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.75 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"

#SFR_MIP="--sfr_aux_enable 1 --sfr_aux_use_state 1 --sfr_aux_use_residual_adapter 1 --sfr_aux_adapter_scale 1.0 --sfr_aux_start_iter 1500 --sfr_aux_ramp_iters 3000 --sfr_aux_kernel_size 5 --sfr_aux_tau 0.5 --sfr_aux_map_clamp 6.0 --sfr_aux_state_weight 0.25 --sfr_aux_use_contribution_gate 1 --sfr_aux_corr_gate_min 0.20 --sfr_aux_log_interval 100"
SFR_MIP_ROOM="--sfr_aux_enable 1 --sfr_aux_use_state 1 --sfr_aux_use_residual_adapter 1 --sfr_aux_adapter_scale 0.75 --sfr_aux_start_iter 2500 --sfr_aux_ramp_iters 5000 --sfr_aux_kernel_size 5 --sfr_aux_tau 0.30 --sfr_aux_map_clamp 4.0 --sfr_aux_state_weight 0.15 --sfr_aux_use_contribution_gate 1 --sfr_aux_corr_gate_min 0.25 --sfr_aux_log_interval 100"
SFR_TNT_TRAIN_NEW="--sfr_aux_enable 1 --sfr_aux_use_state 1 --sfr_aux_use_residual_adapter 1 --sfr_aux_start_iter 1500 --sfr_aux_ramp_iters 3000 --sfr_aux_kernel_size 5 --sfr_aux_tau 0.5 --sfr_aux_state_weight 0.25 --sfr_aux_use_contribution_gate 1 --sfr_aux_corr_gate_min 0.20 --sfr_aux_log_interval 100"
SFR_TNT="--sfr_aux_enable 1 --sfr_aux_use_state 1 --sfr_aux_use_residual_adapter 1 --sfr_aux_adapter_scale 1.0 --sfr_aux_start_iter 1500 --sfr_aux_ramp_iters 3000 --sfr_aux_kernel_size 5 --sfr_aux_tau 0.5 --sfr_aux_map_clamp 6.0 --sfr_aux_state_weight 0.25 --sfr_aux_use_contribution_gate 1 --sfr_aux_corr_gate_min 0.20 --sfr_aux_log_interval 100"
SFR_DB="--sfr_aux_enable 1 --sfr_aux_use_state 1 --sfr_aux_use_residual_adapter 1 --sfr_aux_adapter_scale 1.0 --sfr_aux_start_iter 1500 --sfr_aux_ramp_iters 3000 --sfr_aux_kernel_size 3 --sfr_aux_tau 0.5 --sfr_aux_map_clamp 6.0 --sfr_aux_state_weight 0.25 --sfr_aux_use_contribution_gate 1 --sfr_aux_corr_gate_min 0.20 --sfr_aux_log_interval 100"

run_scene() {
    local name="$1"
    local model="$2"
    shift 2
    echo "[TRAIN] $name -> $model"
    OAR_JOB_ID="$name" "${PYTHON_CMD[@]}" train.py "$@" -m "$model"
}

render_and_metric() {
    local model="$1"
    local mult="${2:-}"
    echo "[RENDER] $model"
    if [[ -n "$mult" ]]; then
        "${PYTHON_CMD[@]}" render.py -m "$model" --skip_train --mult "$mult"
    else
        "${PYTHON_CMD[@]}" render.py -m "$model" --skip_train
    fi
    echo "[METRIC] $model"
    "${PYTHON_CMD[@]}" metrics.py -m "$model"
}

mkdir -p "$OUT_ROOT"
echo "Output root: $OUT_ROOT"

# Mip-NeRF360
# run_scene bicycle "$OUT_ROOT/bicycle" -s /root/360_v2/bicycle -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP
# render_and_metric "$OUT_ROOT/bicycle"

# run_scene flowers "$OUT_ROOT/flowers" -s /root/360_v2/flowers -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP
# render_and_metric "$OUT_ROOT/flowers"

# run_scene garden "$OUT_ROOT/garden" -s /root/360_v2/garden -i images_4 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP_ROOM
# render_and_metric "$OUT_ROOT/garden"

# run_scene room "$OUT_ROOT/room" -s /root/360_v2/room -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP_ROOM
# render_and_metric "$OUT_ROOT/room"

# run_scene bonsai "$OUT_ROOT/bonsai" -s /root/360_v2/bonsai -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_MIP $DENSITY_DEFAULT_PRUNE095 $SFR_MIP_ROOM
# render_and_metric "$OUT_ROOT/bonsai"

run_scene counter "$OUT_ROOT/counter" -s /root/360_v2/counter -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_CONSERVATIVE_PRUNE095 $SFR_MIP_ROOM
render_and_metric "$OUT_ROOT/counter"

# run_scene kitchen "$OUT_ROOT/kitchen" -s /root/360_v2/kitchen -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP
# render_and_metric "$OUT_ROOT/kitchen"

# run_scene stump "$OUT_ROOT/stump" -s /root/360_v2/stump -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP_PSNR $DENSITY_CONSERVATIVE $SFR_MIP
# render_and_metric "$OUT_ROOT/stump"

# run_scene treehill "$OUT_ROOT/treehill" -s /root/360_v2/treehill -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0005 $APSR_MIP $DENSITY_DEFAULT $SFR_MIP
# render_and_metric "$OUT_ROOT/treehill"

# Tanks&Temples
# run_scene train "$OUT_ROOT/train" -s /root/tandt_db/tandt/train --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_TNT_TRAIN_NEW $DENSITY_TNT_TRAIN_NEW $SFR_TNT_TRAIN_NEW
# render_and_metric "$OUT_ROOT/train" 0.7

# run_scene truck "$OUT_ROOT/truck" -s /root/tandt_db/tandt/truck --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_TNT_DENSITY_WEAK $DENSITY_TRUCK $SFR_TNT
# render_and_metric "$OUT_ROOT/truck" 0.7

# Deep Blending
# run_scene drjohnson "$OUT_ROOT/drjohnson" -s /root/tandt_db/db/drjohnson --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DB_FREQ $DENSITY_DEFAULT $SFR_DB
# render_and_metric "$OUT_ROOT/drjohnson" 0.7

# run_scene playroom "$OUT_ROOT/playroom" -s /root/tandt_db/db/playroom --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DB_FREQ $DENSITY_DEFAULT $SFR_DB
# render_and_metric "$OUT_ROOT/playroom" 0.7

echo "Finished. Metrics are under $OUT_ROOT/<scene>/results.json"
