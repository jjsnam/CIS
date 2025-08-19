#!/bin/bash

echo "Testing started"

# Define arrays
model_types=("CNN" "RCNN" "Transformer" "CNN+Transformer" "Statistical")
# model_types=("Transformer")
train_datasets=("Fused")
test_datasets=("SGDF" "OpenForensics" "200kMDID" "Fused")
# top_k_arr=("top10" "top30" "top50")
top_k_arr=("best")

for model_type in "${model_types[@]}"; do
    case "$model_type" in
        "CNN")
            echo "Testing CNN"
            cd "/root/Project/CNN Models/"
            for train_dataset in "${train_datasets[@]}"; do
                for top_k in "${top_k_arr[@]}"; do
                    for test_dataset in "${test_datasets[@]}"; do
                        echo ""
                        echo "Testing $train_dataset Trained CNN Model $top_k on $test_dataset dataset"
                        python test.py --model_path "/root/Project/weights/CNN/$train_dataset/${train_dataset}_CNN_${top_k}.pth" --test_path "/root/Project/datasets/$test_dataset/Test"
                        if [ $? -ne 0 ]; then
                            echo "FAILED: Testing $train_dataset Trained CNN Model $top_k on $test_dataset dataset" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn
                        fi
                        echo ""
                    done
                done
            done
            ;;
        "RCNN")
            echo "Testing RCNN"
            cd "/root/Project/RCNN Models/"
            for train_dataset in "${train_datasets[@]}"; do
                for top_k in "${top_k_arr[@]}"; do
                    for test_dataset in "${test_datasets[@]}"; do
                        echo ""
                        echo "Testing $train_dataset Trained RCNN Model $top_k on $test_dataset dataset"
                        python test.py --model_path "/root/Project/weights/RCNN/$train_dataset/${train_dataset}_RCNN_${top_k}.pth" --test_root "/root/Project/datasets/$test_dataset/Test" --cache_root "/root/Project/RCNN Models/cache/regions/$test_dataset/Test"
                        if [ $? -ne 0 ]; then
                            echo "FAILED: Testing $train_dataset Trained RCNN Model $top_k on $test_dataset dataset" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn
                        fi
                        echo ""
                    done
                done
            done
            ;;
        "Transformer")
            echo "Testing Transformer"
            cd "/root/Project/Transformer/"
            for train_dataset in "${train_datasets[@]}"; do
                for top_k in "${top_k_arr[@]}"; do
                    for test_dataset in "${test_datasets[@]}"; do
                        echo ""
                        echo "Testing $train_dataset Trained Transformer Model $top_k on $test_dataset dataset"
                        python test.py --model_path "/root/Project/weights/Transformer/$train_dataset/${train_dataset}_Transformer_${top_k}.pth" --test_path "/root/Project/datasets/$test_dataset/Test"
                        if [ $? -ne 0 ]; then
                            echo "FAILED: Testing $train_dataset Trained Transformer Model $top_k on $test_dataset dataset" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn
                        fi
                        echo ""
                    done
                done
            done
            ;;
        "CNN+Transformer")
            echo "Testing CNN+Transformer"
            cd "/root/Project/CNN+Transformer/"
            for train_dataset in "${train_datasets[@]}"; do
                for top_k in "${top_k_arr[@]}"; do
                    for test_dataset in "${test_datasets[@]}"; do
                        echo ""
                        echo "Testing $train_dataset Trained CNN+Transformer Model $top_k on $test_dataset dataset"
                        python test.py --model_path "/root/Project/weights/CNN+Transformer/$train_dataset/${train_dataset}_CNN+Transformer_${top_k}.pth" --test_path "/root/Project/datasets/$test_dataset/Test"
                        if [ $? -ne 0 ]; then
                            echo "FAILED: Testing $train_dataset Trained CNN+Transformer Model $top_k on $test_dataset dataset" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn
                        fi
                        echo ""
                    done
                done
            done
            ;;
        "Statistical")
            echo "Testing Statistical Models"
            cd "/root/Project/Statistical Models/"
            for train_dataset in "${train_datasets[@]}"; do
                for top_k in "${top_k_arr[@]}"; do
                    for test_dataset in "${test_datasets[@]}"; do
                        echo ""
                        echo "Testing $train_dataset Statistical Model $top_k on $test_dataset dataset"
                        python test.py --model_real "/root/Project/weights/Statistical/$train_dataset/${train_dataset}_gmm_real_${top_k}.pkl" --model_fake "/root/Project/weights/Statistical/$train_dataset/_gmm_fake_${top_k}.pkl" --test_path "/root/Project/datasets/$test_dataset/Test"
                        if [ $? -ne 0 ]; then
                            echo "FAILED: Testing $train_dataset Statistical Model $top_k on $test_dataset dataset" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn
                        fi
                        echo ""
                    done
                done
            done
            ;;
    esac
done

shutdown