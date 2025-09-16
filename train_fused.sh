#!/bin/bash

# echo "Training script started"
# echo "Sending notification email..."
# echo "Training is now started\nEmail sending works properly" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn

# # CNN
# echo "Starting training CNN model"
# echo "Changing direction"
# cd /root/Project/CNN\ Models/
# echo "Sending notification email..."
# echo "Starting training CNN model" | s-nail -s "Training Notification on step 1" jjsnam@zju.edu.cn

# echo "Starting training CNN model on Fused dataset" | s-nail -s "Training Notification on step 1.1" jjsnam@zju.edu.cn
# start_time=$(date +%s)
# start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start time: $start_fmt"

# python train.py --epochs 15 --dataset_name Fused --train_path /root/Project/datasets/Fused/Train --val_path /root/Project/datasets/Fused/Val --model_path /root/Project/weights/CNN/Fused --lr 1e-4
# status=$?
# end_time=$(date +%s)
# end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# duration=$((end_time - start_time))
# echo "End time: $end_fmt"
# echo "Duration: ${duration}s"

# if [ $status -ne 0 ]; then
#   echo -e "Training failed: CNN-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN-Fused" jjsnam@zju.edu.cn
# else
#   echo -e "Training completed: CNN-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN-Fused" jjsnam@zju.edu.cn
# fi

# # CNN+Transformer

# echo "Starting training CNN+Transformer model" | s-nail -s "Training Notification on step 2" jjsnam@zju.edu.cn
# echo "Starting training CNN+Transformer model"
# echo "Changing direction"
# cd /root/Project/CNN+Transformer/
# echo "Sending notification email..."

# echo "Starting training CNN+Transformer model on Fused dataset" | s-nail -s "Training Notification on step 2.1" jjsnam@zju.edu.cn
# start_time=$(date +%s)
# start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start time: $start_fmt"

# python train.py --epochs 15 --dataset_name Fused --train_path /root/Project/datasets/Fused/Train --val_path /root/Project/datasets/Fused/Val --model_path /root/Project/weights/CNN+Transformer/Fused --lr 1e-4
# status=$?
# end_time=$(date +%s)
# end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# duration=$((end_time - start_time))
# echo "End time: $end_fmt"
# echo "Duration: ${duration}s"

# if [ $status -ne 0 ]; then
#   echo -e "Training failed: CNN+Transformer-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN+Transformer-Fused" jjsnam@zju.edu.cn
# else
#   echo -e "Training completed: CNN+Transformer-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN+Transformer-Fused" jjsnam@zju.edu.cn
# fi

# # RCNN

# echo "Starting training RCNN model" | s-nail -s "Training Notification on step 3" jjsnam@zju.edu.cn
# echo "Starting training RCNN model"
# echo "Changing direction"
# cd /root/Project/RCNN\ Models/
# echo "Sending notification email..."

# echo "Starting training RCNN model on Fused dataset" | s-nail -s "Training Notification on step 3.1" jjsnam@zju.edu.cn
# start_time=$(date +%s)
# start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start time: $start_fmt"

# python train.py --epochs 15 --dataset_name Fused --train_path /root/Project/datasets/Fused/Train --val_path /root/Project/datasets/Fused/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/Fused/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/Fused/Val --model_path /root/Project/weights/RCNN/Fused --lr 1e-4
# status=$?
# end_time=$(date +%s)
# end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# duration=$((end_time - start_time))
# echo "End time: $end_fmt"
# echo "Duration: ${duration}s"

# if [ $status -ne 0 ]; then
#   echo -e "Training failed: RCNN-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: RCNN-Fused" jjsnam@zju.edu.cn
# else
#   echo -e "Training completed: RCNN-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: RCNN-Fused" jjsnam@zju.edu.cn
# fi

# # Statistical

# echo "Starting training Statistical model" | s-nail -s "Training Notification on step 4" jjsnam@zju.edu.cn
# echo "Starting training Statistical model"
# echo "Changing direction"
# cd /root/Project/Statistical\ Models/
# echo "Sending notification email..."

# echo "Starting training Statistical model on Fused dataset" | s-nail -s "Training Notification on step 4.1" jjsnam@zju.edu.cn
# start_time=$(date +%s)
# start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# echo "Start time: $start_fmt"

# python main.py --epochs 15 --dataset_name Fused --train_path /root/Project/datasets/Fused/Train --val_path /root/Project/datasets/Fused/Val  --model_path /root/Project/weights/Statistical/Fused
# status=$?
# end_time=$(date +%s)
# end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
# duration=$((end_time - start_time))
# echo "End time: $end_fmt"
# echo "Duration: ${duration}s"

# if [ $status -ne 0 ]; then
#   echo -e "Training failed: Statistical-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Statistical-Fused" jjsnam@zju.edu.cn
# else
#   echo -e "Training completed: Statistical-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Statistical-Fused" jjsnam@zju.edu.cn
# fi

# Transformer

echo "Starting training Transformer model" | s-nail -s "Training Notification on step 5" jjsnam@zju.edu.cn
echo "Starting training Transformer model"
echo "Changing direction"
cd /root/Project/Transformer/
echo "Sending notification email..."

echo "Starting training Transformer model on Fused dataset" | s-nail -s "Training Notification on step 5.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 15 --dataset_name Fused --train_path /root/Project/datasets/Fused/Train --val_path /root/Project/datasets/Fused/Val --model_path /root/Project/weights/Transformer/Fused --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Transformer-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Transformer-Fused" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Transformer-Fused\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Transformer-Fused" jjsnam@zju.edu.cn
fi

# SHUTDOWN

echo "Begin to shutdown"
shutdown