
export CUDA_VISIBLE_DEVICES=0

#Train model for the default fold with fold_idx=0
output_dir="./results/results_01"
cls_criteria="nipple"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize"
random_crop="True"
fold_idx=0
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_02"
cls_criteria="fat"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=0
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_03"
cls_criteria="ccpec"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=0
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_04"
cls_criteria="imf"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop"
random_crop="False"
fold_idx=0
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

#Train model for the second fold with fold_idx=1 for cross validation
output_dir="./results/results_11"
cls_criteria="nipple"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize"
random_crop="True"
fold_idx=1
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_12"
cls_criteria="fat"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=1
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_13"
cls_criteria="ccpec"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=1
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_14"
cls_criteria="imf"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop"
random_crop="False"
fold_idx=1
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

#Train model for the third fold with fold_idx=2 for cross validation
output_dir="./results/results_21"
cls_criteria="nipple"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize"
random_crop="True"
fold_idx=2
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_22"
cls_criteria="fat"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=2
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_23"
cls_criteria="ccpec"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec"
random_crop="False"
fold_idx=2
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx

output_dir="./results/results_24"
cls_criteria="imf"
dataset_path="/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop"
random_crop="False"
fold_idx=2
pretrained_path="/workspace/bpm_classification/pretrainedmodel/CMAE4.pth"
python ./train.py --warmup_steps 150 --num_steps 1000 --learning_rate 3e-2 --output_dir $output_dir --cls_criteria $cls_criteria --random_crop $random_crop --pretrained_path $pretrained_path --dataset_path $dataset_path --fold_idx $fold_idx
