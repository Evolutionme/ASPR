#APSR_ARGS="--lambda_apsr 0.5 --apsr_hard_mse_weight 0.8 --apsr_edge_weight 0.1"
# APSR_ARGS="--lambda_apsr 0"
#APSR_ARGS="--lambda_apsr 0 --late_mse_weight 0.25 --late_mse_start_iter 15000 --late_mse_warmup_iters 5000 --main_optimizer_interval 1 --sh_optimizer_interval 1"
APSR_Mip360="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"

APSR_TanksandTemple="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.5 --apsr_edge_weight 0.08 --apsr_structure_weight 0.12 --apsr_perceptual_weight 0.05 --apsr_color_weight 0.02 --apsr_pyramid_levels 3"

APSR_DeepBlending="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"
#APSR_DeepBlending="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.2 --apsr_edge_weight 0.06 --apsr_structure_weight 0.1 --apsr_perceptual_weight 0.03 --apsr_color_weight 0.01 --apsr_pyramid_levels 3"
#APSR_ARGS="--lambda_apsr 0.35 --apsr_hard_mse_weight 0.3 --apsr_edge_weight 0.06 --apsr_structure_weight 0.08 --apsr_perceptual_weight 0.025 --apsr_color_weight 0.01 --apsr_pyramid_levels 2"
# echo "mipnerf开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=bicycle python train.py -s /root/360_v2/bicycle -m output/rl/bicycle -i images_4 --eval --densification_interval 100  --optimizer_type default  --grad_abs_thresh 0.0004 $APSR_Mip360
python render.py -m output/rl/bicycle --skip_train
python metrics.py -m output/rl/bicycle
OAR_JOB_ID=flowers python train.py -s /root/360_v2/flowers -m output/rl/flowers -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 $APSR_Mip360
python render.py -m output/rl/flowers --skip_train
python metrics.py -m output/rl/flowers
# OAR_JOB_ID=garden python train.py -s /root/360_v2/garden -m output/rl/garden -i images_4 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --loss_thresh 0.06  --grad_abs_thresh 0.0002 $APSR_Mip360   
# python render.py -m output/rl/garden --skip_train
# python metrics.py -m output/rl/garden
# OAR_JOB_ID=stump python train.py -s /root/360_v2/stump -m output/rl/stump -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0004 $APSR_Mip360
# python render.py -m output/rl/stump --skip_train
# python metrics.py -m output/rl/stump
# OAR_JOB_ID=treehill python train.py -s /root/360_v2/treehill -m output/rl/treehill -i images_4 --eval --densification_interval 100  --optimizer_type default --grad_abs_thresh 0.0005 $APSR_Mip360
# python render.py -m output/rl/treehill --skip_train
# python metrics.py -m output/rl/treehill
OAR_JOB_ID=room python train.py -s /root/360_v2/room -m output/rl/room -i images_2 --eval --densification_interval 100  --optimizer_type default --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_Mip360
python render.py -m output/rl/room --skip_train
python metrics.py -m output/rl/room
# OAR_JOB_ID=counter python train.py -s /root/360_v2/counter -m output/rl/counter -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0002 $APSR_Mip360
# python render.py -m output/rl/counter --skip_train
# python metrics.py -m output/rl/counter


# OAR_JOB_ID=kitchen python train.py -s /root/360_v2/kitchen -m output/rl/kitchen -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_Mip360
# python render.py -m output/rl/kitchen --skip_train
# python metrics.py -m output/rl/kitchen
# OAR_JOB_ID=bonsai python train.py -s /root/360_v2/bonsai -m output/rl/bonsai -i images_2 --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.02 --grad_abs_thresh 0.0001 $APSR_Mip360
# python render.py -m output/rl/bonsai --skip_train
# python metrics.py -m output/rl/bonsai
# end=$(date +%s)
# echo "mipnerf结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "mipnerf耗时: $((end - start)) s"

#二
# echo "tanksandtemples开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=truck python train.py -s /root/tandt_db/tandt/truck -m output/rl/truck --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.04 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_TanksandTemple
python render.py -m output/rl/truck --skip_train --mult 0.7
python metrics.py -m output/rl/truck
OAR_JOB_ID=train python train.py -s /root/tandt_db/tandt/train -m output/rl/train --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.042 --grad_abs_thresh 0.0001 --mult 0.7 $APSR_TanksandTemple
python render.py -m output/rl/train --skip_train --mult 0.7
python metrics.py -m output/rl/train
# end=$(date +%s)
# echo "tanksandtemples结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "tanksandtemples耗时: $((end - start)) s"

# echo "db开始时间: $(date '+%Y-%m-%d %H:%M:%S')"
# start=$(date +%s)
OAR_JOB_ID=drjohnson python train.py -s /root/tandt_db/db/drjohnson -m output/rl/drjohnson --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0025 --lowfeature_lr 0.0005 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DeepBlending
python render.py -m output/rl/drjohnson --skip_train --mult 0.7
python metrics.py -m output/rl/drjohnson
OAR_JOB_ID=playroom python train.py -s /root/tandt_db/db/playroom -m output/rl/playroom --eval --densification_interval 100  --optimizer_type default  --highfeature_lr 0.0015 --grad_abs_thresh 0.0002 --mult 0.7 $APSR_DeepBlending
python render.py -m output/rl/playroom --skip_train --mult 0.7
python metrics.py -m output/rl/playroom
# end=$(date +%s)
# echo "db结束时间: $(date '+%Y-%m-%d %H:%M:%S')"
# echo "db耗时: $((end - start)) s"


# python render.py -m output-up/rl/bicycle --skip_train
# python metrics.py -m output-up/rl/bicycle

# python render.py -m output-up/rl/flowers --skip_train
# python metrics.py -m output-up/rl/flowers

# python render.py -m output-up/rl/garden --skip_train
# python metrics.py -m output-up/rl/garden

# python render.py -m output-up/rl/stump --skip_train
# python metrics.py -m output-up/rl/stump

# python render.py -m output-up/rl/treehill --skip_train
# python metrics.py -m output-up/rl/treehill

# python render.py -m output-up/rl/room --skip_train
# python metrics.py -m output-up/rl/room

# python render.py -m output-up/rl/counter --skip_train
# python metrics.py -m output-up/rl/counter

# python render.py -m output-up/rl/kitchen --skip_train
# python metrics.py -m output-up/rl/kitchen

# python render.py -m output-up/rl/bonsai --skip_train
# python metrics.py -m output-up/rl/bonsai

# python render.py -m output/rl/truck --skip_train --mult 0.7
# python metrics.py -m output/rl/truck

# python render.py -m output/rl/train --skip_train --mult 0.7
# python metrics.py -m output/rl/train

# python render.py -m output/rl/drjohnson --skip_train --mult 0.7
# python metrics.py -m output/rl/drjohnson

# python render.py -m output/rl/playroom --skip_train --mult 0.7
# python metrics.py -m output/rl/playroom

# python report_results.py
