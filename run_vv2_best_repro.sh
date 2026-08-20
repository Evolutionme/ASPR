#!/usr/bin/env bash
set -Eeuo pipefail

# Reproduce the best per-scene training configurations recorded in the VV2 logs.
# The default output root is timestamped, so an old experiment is never replaced.
# Usage:
#   ./run_vv2_best_repro.sh
#   SCENES=room,train ./run_vv2_best_repro.sh output/my_reproduction

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

ITERATIONS="${ITERATIONS:-30000}"
OUT_ROOT="${1:-output/vv2_best_repro_$(date +%Y%m%d_%H%M%S)}"
SCENES="${SCENES:-all}"
DRY_RUN="${DRY_RUN:-0}"
FORCE="${FORCE:-0}"
RUN_FREQUENCY_CALIBRATION="${RUN_FREQUENCY_CALIBRATION:-1}"

# TFRC is shared within each dataset. Deep Blending intentionally keeps the
# previously validated scene-specific settings below.
TFRC_MIP_SCALE="${TFRC_MIP_SCALE:-2.0}"
TFRC_TNT_SCALE="${TFRC_TNT_SCALE:-2.0}"
TFRC_DB_DRJOHNSON_SCALE="${TFRC_DB_DRJOHNSON_SCALE:-3.5}"
TFRC_DB_PLAYROOM_SCALE="${TFRC_DB_PLAYROOM_SCALE:-1.5}"

# safe_state() fixes the Python/Torch seed to 0. These variables also reduce
# run-to-run differences in CUDA and Python hashing where supported.
export PYTHONHASHSEED="${PYTHONHASHSEED:-0}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
export PYTHONDONTWRITEBYTECODE=1

PYTHON_CMD=(conda run --no-capture-output -n LeGS python)

APSR_MIP=(
    --lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08
    --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05
    --apsr_color_weight 0.02 --apsr_pyramid_levels 3
)
APSR_MIP_PSNR=(
    --lambda_apsr 0.35 --apsr_hard_mse_weight 0.65 --apsr_edge_weight 0.08
    --apsr_structure_weight 0.10 --apsr_perceptual_weight 0.05
    --apsr_color_weight 0.02 --apsr_pyramid_levels 3
)
APSR_TNT=(
    --lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08
    --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05
    --apsr_color_weight 0.02 --apsr_pyramid_levels 3
)
APSR_DB=(
    --lambda_apsr 0.35 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06
    --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025
    --apsr_color_weight 0.01 --apsr_pyramid_levels 2
)
APSR_DENSITY_WEAK=(
    --lambda_apsr 0.0 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06
    --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025
    --apsr_color_weight 0.01 --apsr_pyramid_levels 2
)

DENSITY_DEFAULT=(
    --use_apsr_density_control 1 --lambda_apsr_density 0.20
    --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000
    --apsr_density_map_clamp 6.0 --apsr_density_use_state 1
    --apsr_density_state_weight 0.5 --apsr_density_use_contribution_gate 1
    --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0
    --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1
    --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05
    --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100
)
DENSITY_CONSERVATIVE=(
    --use_apsr_density_control 1 --lambda_apsr_density 0.10
    --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000
    --apsr_density_map_clamp 6.0 --apsr_density_use_state 1
    --apsr_density_state_weight 0.25 --apsr_density_use_contribution_gate 1
    --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0
    --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1
    --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05
    --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100
)
DENSITY_TRUCK=(
    --use_apsr_density_control 1 --lambda_apsr_density 0.30
    --apsr_density_start_iter 1500 --apsr_density_warmup_iters 3000
    --apsr_density_map_clamp 6.0 --apsr_density_use_state 1
    --apsr_density_state_weight 0.75 --apsr_density_use_contribution_gate 1
    --apsr_density_gate_temperature 1.0 --apsr_density_gate_floor 0.0
    --apsr_density_use_corr_gate 1 --apsr_density_corr_gate_floor 0.1
    --apsr_density_use_reward 0 --apsr_density_reward_weight 0.05
    --apsr_density_prune_scale 1.0 --apsr_density_log_interval 100
)

mkdir -p "$OUT_ROOT/logs"

print_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
}

run_logged() {
    local log_file="$1"
    shift
    print_command "$@" | tee -a "$log_file"
    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    "$@" 2>&1 | tee -a "$log_file"
}

scene_selected() {
    [[ "$SCENES" == "all" || ",${SCENES}," == *",$1,"* ]]
}

require_dataset() {
    local scene="$1"
    local source="$2"
    if [[ ! -d "$source" ]]; then
        printf 'Dataset directory for %s does not exist: %s\n' "$scene" "$source" >&2
        exit 1
    fi
}

train_scene() {
    local name="$1"
    local source="$2"
    local images="$3"
    local mult="$4"
    shift 4
    local model="$OUT_ROOT/$name"
    local log_file="$OUT_ROOT/logs/${name}.log"
    local point_cloud="$model/point_cloud/iteration_${ITERATIONS}/point_cloud.ply"

    require_dataset "$name" "$source"
    if [[ "$FORCE" != "1" && -e "$model" ]]; then
        printf 'Refusing to reuse existing scene directory: %s\n' "$model" >&2
        printf 'Use a new OUT_ROOT or set FORCE=1 explicitly.\n' >&2
        exit 1
    fi

    mkdir -p "$model"
    {
        printf '# VV2 best-result reproduction\n'
        printf '# scene=%s source=%s iterations=%s\n' "$name" "$source" "$ITERATIONS"
        printf '# output=%s\n' "$model"
    } > "$log_file"

    local command=("${PYTHON_CMD[@]}" train.py
        -s "$source" -m "$model" -i "$images" --eval
        --iterations "$ITERATIONS" --save_iterations "$ITERATIONS"
        --checkpoint_iterations "$ITERATIONS" --densification_interval 100
        --mult "$mult")
    command+=("$@")
    run_logged "$log_file" "${command[@]}"

    if [[ "$DRY_RUN" == "1" ]]; then
        return 0
    fi
    if [[ ! -f "$point_cloud" ]]; then
        printf 'Training finished without the expected checkpoint: %s\n' "$point_cloud" >&2
        exit 1
    fi
}

render_and_evaluate() {
    local name="$1"
    local mult="$2"
    local model="$OUT_ROOT/$name"
    local log_file="$OUT_ROOT/logs/${name}.log"
    local raw_method="$model/test/ours_${ITERATIONS}_raw"

    run_logged "$log_file" "${PYTHON_CMD[@]}" render.py -m "$model" \
        --iteration "$ITERATIONS" --skip_train --mult "$mult" --render_suffix _raw
    run_logged "$log_file" "${PYTHON_CMD[@]}" metrics.py -m "$raw_method"
    if [[ "$DRY_RUN" != "1" && ! -f "$raw_method/results.json" ]]; then
        printf 'Metric evaluation did not create %s/results.json\n' "$raw_method" >&2
        exit 1
    fi

    if [[ "$RUN_FREQUENCY_CALIBRATION" == "1" ]]; then
        local calibration_mult="$mult"
        local calibration_scale="$TFRC_MIP_SCALE"
        case "$name" in
            train|truck)
                calibration_scale="$TFRC_TNT_SCALE"
                ;;
            drjohnson)
                calibration_mult="0.6"
                calibration_scale="$TFRC_DB_DRJOHNSON_SCALE"
                ;;
            playroom)
                calibration_mult="0.6"
                calibration_scale="$TFRC_DB_PLAYROOM_SCALE"
                ;;
        esac
        run_logged "$log_file" "${PYTHON_CMD[@]}" render.py -m "$model" \
            --iteration "$ITERATIONS" --skip_train --mult "$calibration_mult" \
            --render_suffix _tfrc --frequency_calibration scalar \
            --frequency_calibration_views 32 --frequency_kernel_size 3 \
            --frequency_coefficient_scale "$calibration_scale"
        local calibrated_method="$model/test/ours_${ITERATIONS}_tfrc"
        run_logged "$log_file" "${PYTHON_CMD[@]}" metrics.py -m "$calibrated_method"
        if [[ "$DRY_RUN" != "1" && ! -f "$calibrated_method/results.json" ]]; then
            printf 'Metric evaluation did not create %s/results.json\n' "$calibrated_method" >&2
            exit 1
        fi
    fi
}

run_one() {
    local name="$1"
    local source="$2"
    local images="$3"
    local mult="$4"
    shift 4
    if ! scene_selected "$name"; then
        return 0
    fi
    train_scene "$name" "$source" "$images" "$mult" "$@"
    render_and_evaluate "$name" "$mult"
}

printf 'Output root: %s\n' "$OUT_ROOT"
printf 'Scenes: %s | iterations: %s | dry-run: %s | frequency calibration: %s\n' \
    "$SCENES" "$ITERATIONS" "$DRY_RUN" "$RUN_FREQUENCY_CALIBRATION"

# run_one bicycle /root/360_v2/bicycle images_4 0.5 \
#    --optimizer_type default --grad_abs_thresh 0.0004 "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
 run_one flowers /root/360_v2/flowers images_4 0.5 \
     --optimizer_type default --grad_abs_thresh 0.0004 "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
# run_one garden /root/360_v2/garden images_4 0.5 \
#     --optimizer_type default --highfeature_lr 0.02 --loss_thresh 0.06 \
#     --grad_abs_thresh 0.0002 "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
# run_one room /root/360_v2/room images_2 0.5 \
#     --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 \
#     "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
# run_one bonsai /root/360_v2/bonsai images_2 0.5 \
#     --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 \
#     --late_mse_weight 0.25 --late_mse_start_iter 15000 --late_mse_warmup_iters 5000 \
#     "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
# run_one counter /root/360_v2/counter images_2 0.5 \
#     --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 \
#     "${APSR_MIP[@]}" "${DENSITY_CONSERVATIVE[@]}"
# run_one kitchen /root/360_v2/kitchen images_2 0.5 \
#     --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0001 \
#     "${APSR_MIP[@]}" "${DENSITY_DEFAULT[@]}"
# run_one stump /root/360_v2/stump images_4 0.5 \
#     --optimizer_type default --grad_abs_thresh 0.0004 "${APSR_MIP_PSNR[@]}" \
#     "${DENSITY_DEFAULT[@]}"
# run_one treehill /root/360_v2/treehill images_4 0.5 \
#     --optimizer_type default --grad_abs_thresh 0.0005 "${APSR_MIP[@]}" \
#     "${DENSITY_DEFAULT[@]}"
# run_one train /root/tandt_db/tandt/train images 0.7 \
#     --optimizer_type default --highfeature_lr 0.042 --grad_abs_thresh 0.0001 \
#     "${APSR_TNT[@]}" "${DENSITY_DEFAULT[@]}"
# run_one truck /root/tandt_db/tandt/truck images 0.7 \
#     --optimizer_type default --highfeature_lr 0.04 --grad_abs_thresh 0.0001 \
#     "${APSR_DENSITY_WEAK[@]}" "${DENSITY_TRUCK[@]}"
# run_one drjohnson /root/tandt_db/db/drjohnson images 0.7 \
#     --optimizer_type default --highfeature_lr 0.0025 --lowfeature_lr 0.0005 \
#     --grad_abs_thresh 0.0002 "${APSR_DB[@]}" "${DENSITY_DEFAULT[@]}"
# run_one playroom /root/tandt_db/db/playroom images 0.7 \
#     --optimizer_type default --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 \
#     "${APSR_DB[@]}" "${DENSITY_DEFAULT[@]}"

printf '\nFinished. Raw metrics are under %s/<scene>/test/ours_%s_raw/results.json\n' "$OUT_ROOT" "$ITERATIONS"
if [[ "$RUN_FREQUENCY_CALIBRATION" == "1" ]]; then
    printf 'Frequency-calibrated metrics are under %s/<scene>/test/ours_%s_tfrc/results.json\n' "$OUT_ROOT" "$ITERATIONS"
fi
