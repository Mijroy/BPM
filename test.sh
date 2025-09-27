export CUDA_VISIBLE_DEVICES=0

file_path="./results/results_01/1400_bestauc_checkpoint.bin"
cls_criteria="nipple"
fold_idx=0
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_02/1820_bestauc_checkpoint.bin"
cls_criteria="fat"
fold_idx=0
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_03/910_bestauc_checkpoint.bin"
cls_criteria="ccpec"
fold_idx=0
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_04/630_bestauc_checkpoint.bin"
cls_criteria="imf"
fold_idx=0
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_11/1595_bestauc_checkpoint.bin"
cls_criteria="nipple"
fold_idx=1
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_12/1740_bestauc_checkpoint.bin"
cls_criteria="fat"
fold_idx=1
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_13/900_bestauc_checkpoint.bin"
cls_criteria="ccpec"
fold_idx=1
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_14/900_bestauc_checkpoint.bin"
cls_criteria="imf"
fold_idx=1
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_21/1540_bestauc_checkpoint.bin"
cls_criteria="nipple"
fold_idx=2
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_22/1680_bestauc_checkpoint.bin"
cls_criteria="fat"
fold_idx=2
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_23/980_bestauc_checkpoint.bin"
cls_criteria="ccpec"
fold_idx=2
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi

file_path="./results/results_24/560_bestauc_checkpoint.bin"
cls_criteria="imf"
fold_idx=2
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/mnt/cchen/bpm/phase3_ge_origin/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi