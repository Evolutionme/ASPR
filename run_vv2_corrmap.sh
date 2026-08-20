#!/usr/bin/env bash
set -euo pipefail

OUT_ROOT="${1:-output/vv2_corrmap}"
PYTHON_CMD=(conda run -n LeGS python)

APSR_MIP="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_MIP_PSNR="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.65 --apsr_edge_weight 0.08 --apsr_structure_weight 0.10 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_TNT="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"
APSR_DB="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"
APSR_DENSITY_WEAK="--lambda_apsr 0.0 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"

DENSITY_DEFAULT="--use_apsr_density_control 1 --lambda_apsr_density 0.20 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"
DENSITY_DEFAULT_PRUNE095="--use_apsr_density_control 1 --lambda_apsr_density 0.20 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 0.95 --apsr_density_log_interval 100"
DENSITY_CONSERVATIVE="--use_apsr_density_control 1 --lambda_apsr_density 0.10 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.25 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"
DENSITY_CONSERVATIVE_PRUNE095="--use_apsr_density_control 1 --lambda_apsr_density 0.10 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.25 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 0.95 --apsr_density_log_interval 100"
DENSITY_TRUCK="--use_apsr_density_control 1 --lambda_apsr_density 0.30 --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000 --apsr_density_map_clamp 6.0 --apsr_density_use_state 1 --apsr_density_state_weight 0.75 --apsr_density_use_contribution_gate 1 --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0 --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1 --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05 --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100"

run_scene() {
    local name="$1"
    local model="$2"
    shift 2
    OAR_JOB_ID="$name" "${PYTHON_CMD[@]}" train.py "$@" -m "$model"
}

render_and_metric() {
    local model="$1"
    local mult="${2:-}"
    if [[ -n "$mult" ]]; then
        "${PYTHON_CMD[@]}" render.py -m "$model" --skip_train --mult "$mult"
    else
        "${PYTHON_CMD[@]}" render.py -m "$model" --skip_train
    fi
    "${PYTHON_CMD[@]}" metrics.py -m "$model"
}

# run_scene bicycle "$OUT_ROOT/bicycle" -s /root/360_v2/bicycle -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/bicycle"

# run_scene flowers "$OUT_ROOT/flowers" -s /root/360_v2/flowers -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/flowers"

# run_scene garden "$OUT_ROOT/garden" -s /root/360_v2/garden -i images_4 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --loss_thresh 0.06 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/garden"

# run_scene room "$OUT_ROOT/room" -s /root/360_v2/room -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/room"

# run_scene bonsai "$OUT_ROOT/bonsai" -s /root/360_v2/bonsai -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_MIP $DENSITY_DEFAULT_PRUNE095
# render_and_metric "$OUT_ROOT/bonsai"

# run_scene counter "$OUT_ROOT/counter" -s /root/360_v2/counter -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_MIP $DENSITY_CONSERVATIVE_PRUNE095
# render_and_metric "$OUT_ROOT/counter"

# run_scene kitchen "$OUT_ROOT/kitchen" -s /root/360_v2/kitchen -i images_2 --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/kitchen"

# run_scene stump "$OUT_ROOT/stump" -s /root/360_v2/stump -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0004 $APSR_MIP_PSNR $DENSITY_CONSERVATIVE
# render_and_metric "$OUT_ROOT/stump"

# run_scene treehill "$OUT_ROOT/treehill" -s /root/360_v2/treehill -i images_4 --eval --densification_interval 100 --optimizer_type default --grad_abs_thresh 0.0005 $APSR_MIP $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/treehill"

run_scene train "$OUT_ROOT/train" -s /root/tandt_db/tandt/train --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_TNT $DENSITY_DEFAULT
render_and_metric "$OUT_ROOT/train" 0.7

# run_scene truck "$OUT_ROOT/truck" -s /root/tandt_db/tandt/truck --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_DENSITY_WEAK $DENSITY_TRUCK
# render_and_metric "$OUT_ROOT/truck" 0.7

# run_scene drjohnson "$OUT_ROOT/drjohnson" -s /root/tandt_db/db/drjohnson --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DB $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/drjohnson" 0.7

# run_scene playroom "$OUT_ROOT/playroom" -s /root/tandt_db/db/playroom --eval --densification_interval 100 --optimizer_type default --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DB $DENSITY_DEFAULT
# render_and_metric "$OUT_ROOT/playroom" 0.7
