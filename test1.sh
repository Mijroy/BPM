export CUDA_VISIBLE_DEVICES=0


# file_path="/mnt/BPM/classification_criteria/checkpoint/final_review/CCPec/910_bestauc_checkpoint.bin"
# cls_criteria="ccpec"
# fold_idx=2
# if [[ $cls_criteria = "nipple" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "imf" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "ccpec" ]]; then
#     python3 ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# elif [[ $cls_criteria = "fat" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# fi

# file_path="/mnt/BPM/classification_criteria/checkpoint/final_review/NippleInProfile/1400_bestauc_checkpoint.bin"
# cls_criteria="nipple"
# fold_idx=0
# if [[ $cls_criteria = "nipple" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "imf" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "ccpec" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# elif [[ $cls_criteria = "fat" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# fi

# file_path="/mnt/BPM/classification_criteria/checkpoint/final_review/RetromammaryFatSufficient/1820_bestauc_checkpoint.bin"
# cls_criteria="fat"
# fold_idx=0
# if [[ $cls_criteria = "nipple" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "imf" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
# elif [[ $cls_criteria = "ccpec" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# elif [[ $cls_criteria = "fat" ]]; then
#     python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
# fi

file_path="/mnt/BPM/classification_criteria/checkpoint/final_review/IMFVisible/630_bestauc_checkpoint.bin"
cls_criteria="imf"
fold_idx=0
if [[ $cls_criteria = "nipple" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "True" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_proc_wocentralize" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "imf" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx
elif [[ $cls_criteria = "ccpec" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
elif [[ $cls_criteria = "fat" ]]; then
    python ./test.py --cls_criteria $cls_criteria --random_crop "False" --dataset_path "/workspace/bpm_classification/preprocessed_data/20240404/img_crop_ccpec" --stage "test" --name "test" --pretrained_path $file_path --fold_idx $fold_idx 
fi