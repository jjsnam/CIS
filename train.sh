#!/bin/bash
# exec > >(tee -i train.log)
# exec 2>&1
# du -sh /root/Project

echo "Training script stated"
echo "Sending notification email..."
echo "Training is now started\n Email sending works properly" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn

# CNN

echo "Starting training CNN model"
echo "Changing direction"
cd /root/Project/CNN\ Models/
echo "Sending notification email..."
echo "Starting training CNN model" | s-nail -s "Training Notification on step 1" jjsnam@zju.edu.cn

echo "Starting training CNN model on SGDF dataset" | s-nail -s "Training Notification on step 1.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/CNN/SGDF --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN-SGDF" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN-SGDF" jjsnam@zju.edu.cn
fi

echo "Sending notification email..."
echo "Starting training CNN model on OpenForensics dataset" | s-nail -s "Training Notification on step 1.2" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/CNN/OpenForensics --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN-OpenForensics" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN-OpenForensics" jjsnam@zju.edu.cn
fi

echo "Sending notification email..."
echo "Starting training CNN model on 200kMDID dataset" | s-nail -s "Training Notification on step 1.3" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/CNN/200kMDID --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN-200kMDID" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN-200kMDID" jjsnam@zju.edu.cn
fi
echo "Finish training CNN model"

# CNN+Transformer

echo "Starting training CNN+Transformer model" | s-nail -s "Training Notification on step 2" jjsnam@zju.edu.cn
echo "Starting training CNN+Transformer model"
echo "Changing direction"
cd /root/Project/CNN+Transformer/
echo "Sending notification email..."

echo "Starting training CNN+Transformer model on SGDF dataset" | s-nail -s "Training Notification on step 2.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/CNN+Transformer/SGDF --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN+Transformer-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN+Transformer-SGDF" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN+Transformer-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN+Transformer-SGDF" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training CNN+Transformer model on OpenForensics dataset" | s-nail -s "Training Notification on step 2.2" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/CNN+Transformer/OpenForensics --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN+Transformer-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN+Transformer-OpenForensics" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN+Transformer-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN+Transformer-OpenForensics" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training CNN+Transformer model on 200kMDID dataset" | s-nail -s "Training Notification on step 2.3" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/CNN+Transformer/200kMDID --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: CNN+Transformer-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: CNN+Transformer-200kMDID" jjsnam@zju.edu.cn
else
  echo -e "Training completed: CNN+Transformer-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: CNN+Transformer-200kMDID" jjsnam@zju.edu.cn
fi
echo "Finish training CNN+Transformer model"

# RCNN

echo "Starting training RCNN model" | s-nail -s "Training Notification on step 3" jjsnam@zju.edu.cn
echo "Starting training RCNN model"
echo "Changing direction"
cd /root/Project/RCNN\ Models/
echo "Sending notification email..."

echo "Starting training RCNN model on SGDF dataset" | s-nail -s "Training Notification on step 3.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/SGDF/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/SGDF/Val --model_path /root/Project/weights/RCNN/SGDF --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: RCNN-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: RCNN-SGDF" jjsnam@zju.edu.cn
else
  echo -e "Training completed: RCNN-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: RCNN-SGDF" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training RCNN model on OpenForensics dataset" | s-nail -s "Training Notification on step 3.2" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/OpenForensics/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/OpenForensics/Val --model_path /root/Project/weights/RCNN/OpenForensics --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: RCNN-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: RCNN-OpenForensics" jjsnam@zju.edu.cn
else
  echo -e "Training completed: RCNN-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: RCNN-OpenForensics" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training RCNN model on 200kMDID dataset" | s-nail -s "Training Notification on step 3.3" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/200kMDID/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/200kMDID/Val --model_path /root/Project/weights/RCNN/200kMDID --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: RCNN-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: RCNN-200kMDID" jjsnam@zju.edu.cn
else
  echo -e "Training completed: RCNN-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: RCNN-200kMDID" jjsnam@zju.edu.cn
fi
echo "Finish training RCNN model"

# Statistical

echo "Starting training Statistical model" | s-nail -s "Training Notification on step 4" jjsnam@zju.edu.cn
echo "Starting training Statistical model"
echo "Changing direction"
cd /root/Project/Statistical\ Models/
echo "Sending notification email..."

echo "Starting training Statistical model on SGDF dataset" | s-nail -s "Training Notification on step 4.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val  --model_path /root/Project/weights/Statistical/SGDF
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Statistical-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Statistical-SGDF" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Statistical-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Statistical-SGDF" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training Statistical model on OpenForensics dataset" | s-nail -s "Training Notification on step 4.2" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val  --model_path /root/Project/weights/Statistical/OpenForensics
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Statistical-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Statistical-OpenForensics" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Statistical-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Statistical-OpenForensics" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training Statistical model on 200kMDID dataset" | s-nail -s "Training Notification on step 4.3" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val  --model_path /root/Project/weights/Statistical/200kMDID
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Statistical-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Statistical-200kMDID" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Statistical-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Statistical-200kMDID" jjsnam@zju.edu.cn
fi
echo "Finish training Statistical model"

# Transformer

echo "Starting training Transformer model" | s-nail -s "Training Notification on step 5" jjsnam@zju.edu.cn
echo "Starting training Transformer model"
echo "Changing direction"
cd /root/Project/Transformer/
echo "Sending notification email..."

echo "Starting training Transformer model on SGDF dataset" | s-nail -s "Training Notification on step 5.1" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/Transformer/SGDF --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Transformer-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Transformer-SGDF" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Transformer-SGDF\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Transformer-SGDF" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training Transformer model on OpenForensics dataset" | s-nail -s "Training Notification on step 5.2" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/Transformer/OpenForensics --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Transformer-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Transformer-OpenForensics" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Transformer-OpenForensics\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Transformer-OpenForensics" jjsnam@zju.edu.cn
fi
echo "Sending notification email..."

echo "Starting training Transformer model on 200kMDID dataset" | s-nail -s "Training Notification on step 5.3" jjsnam@zju.edu.cn
start_time=$(date +%s)
start_fmt=$(date "+%Y-%m-%d %H:%M:%S")
echo "Start time: $start_fmt"

python main.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/Transformer/200kMDID --lr 1e-4
status=$?
end_time=$(date +%s)
end_fmt=$(date "+%Y-%m-%d %H:%M:%S")
duration=$((end_time - start_time))
echo "End time: $end_fmt"
echo "Duration: ${duration}s"

if [ $status -ne 0 ]; then
  echo -e "Training failed: Transformer-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Failure: Transformer-200kMDID" jjsnam@zju.edu.cn
else
  echo -e "Training completed: Transformer-200kMDID\nStart: $start_fmt\nEnd: $end_fmt\nDuration: ${duration}s" | s-nail -s "Success: Transformer-200kMDID" jjsnam@zju.edu.cn
fi
echo "Finish training Transformer model"


echo "Sending notification email..."
echo "All model training completed successfully." | s-nail -s "Training Notification - All Done" jjsnam@zju.edu.cn

echo "Begin to shutdown"
shutdown

# #!/bin/bash
# exec > >(tee -i train.log)
# exec 2>&1
# du -sh /root/Project

# echo "Training script stated"
# echo "Sending notification email..."
# echo "Training is now started\n Email sending works properly" | s-nail -s "Training Notification on step 0" jjsnam@zju.edu.cn

# # CNN

# echo "Starting training CNN model"
# echo "Changing direction"
# cd /root/Project/CNN\ Models/
# echo "Sending notification email..."
# echo "Starting training CNN model" | s-nail -s "Training Notification on step 1" jjsnam@zju.edu.cn

# echo "Starting training CNN model on SGDF dataset" | s-nail -s "Training Notification on step 1.1" jjsnam@zju.edu.cn

# python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/CNN/SGDF --lr 1e-4 || echo "Training failed: CNN-SGDF" | s-nail -s "Failure: CNN-SGDF" jjsnam@zju.edu.cn

# echo "Sending notification email..."
# echo "Starting training CNN model on OpenForensics dataset" | s-nail -s "Training Notification on step 1.2" jjsnam@zju.edu.cn

# python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/CNN/OpenForensics --lr 1e-4 || echo "Training failed: CNN-OpenForensics" | s-nail -s "Failure: CNN-OpenForensics" jjsnam@zju.edu.cn

# echo "Sending notification email..."
# echo "Starting training CNN model on 200kMDID dataset" | s-nail -s "Training Notification on step 1.3" jjsnam@zju.edu.cn

# python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/CNN/200kMDID --lr 1e-4 || echo "Training failed: CNN-200kMDID" | s-nail -s "Failure: CNN-200kMDID" jjsnam@zju.edu.cn
# echo "Finish training CNN model"

# # CNN+Transformer

# echo "Starting training CNN+Transformer model" | s-nail -s "Training Notification on step 2" jjsnam@zju.edu.cn
# echo "Starting training CNN+Transformer model"
# echo "Changing direction"
# echo "Sending notification email..."

# echo "Starting training CNN+Transformer model on SGDF dataset" | s-nail -s "Training Notification on step 2.1" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/CNN+Transformer/SGDF --lr 1e-4 || echo "Training failed: CNN+Transformer-SGDF" | s-nail -s "Failure: CNN+Transformer-SGDF" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training CNN+Transformer model on OpenForensics dataset" | s-nail -s "Training Notification on step 2.2" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/CNN+Transformer/OpenForensics --lr 1e-4 || echo "Training failed: CNN+Transformer-OpenForensics" | s-nail -s "Failure: CNN+Transformer-OpenForensics" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training CNN+Transformer model on 200kMDID dataset" | s-nail -s "Training Notification on step 2.3" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/CNN+Transformer/200kMDID --lr 1e-4 || echo "Training failed: CNN+Transformer-200kMDID" | s-nail -s "Failure: CNN+Transformer-200kMDID" jjsnam@zju.edu.cn
# echo "Finish training CNN+Transformer model"

# # RCNN

# echo "Starting training RCNN model" | s-nail -s "Training Notification on step 3" jjsnam@zju.edu.cn
# echo "Starting training RCNN model"
# echo "Changing direction"
# echo "Sending notification email..."

# echo "Starting training RCNN model on SGDF dataset" | s-nail -s "Training Notification on step 3.1" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/SGDF/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/SGDF/Val --model_path /root/Project/weights/RCNN/SGDF --lr 1e-4 || echo "Training failed: RCNN-SGDF" | s-nail -s "Failure: RCNN-SGDF" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training RCNN model on OpenForensics dataset" | s-nail -s "Training Notification on step 3.2" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/OpenForensics/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/OpenForensics/Val --model_path /root/Project/weights/RCNN/OpenForensics --lr 1e-4 || echo "Training failed: RCNN-OpenForensics" | s-nail -s "Failure: RCNN-OpenForensics" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training RCNN model on 200kMDID dataset" | s-nail -s "Training Notification on step 3.3" jjsnam@zju.edu.cn
# python train.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --train_cache_path /root/Project/RCNN\ Models/cache/regions/200kMDID/Train --val_cache_path /root/Project/RCNN\ Models/cache/regions/200kMDID/Val --model_path /root/Project/weights/RCNN/200kMDID --lr 1e-4 || echo "Training failed: RCNN-200kMDID" | s-nail -s "Failure: RCNN-200kMDID" jjsnam@zju.edu.cn
# echo "Finish training RCNN model"

# # Statistical

# echo "Starting training Statistical model" | s-nail -s "Training Notification on step 4" jjsnam@zju.edu.cn
# echo "Starting training Statistical model"
# echo "Changing direction"
# echo "Sending notification email..."

# echo "Starting training Statistical model on SGDF dataset" | s-nail -s "Training Notification on step 4.1" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val  --model_path /root/Project/weights/Statistical/SGDF || echo "Training failed: Statistical-SGDF" | s-nail -s "Failure: Statistical-SGDF" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training Statistical model on OpenForensics dataset" | s-nail -s "Training Notification on step 4.2" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val  --model_path /root/Project/weights/Statistical/OpenForensics || echo "Training failed: Statistical-OpenForensics" | s-nail -s "Failure: Statistical-OpenForensics" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training Statistical model on 200kMDID dataset" | s-nail -s "Training Notification on step 4.3" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val  --model_path /root/Project/weights/Statistical/200kMDID || echo "Training failed: Statistical-200kMDID" | s-nail -s "Failure: Statistical-200kMDID" jjsnam@zju.edu.cn
# echo "Finish training Statistical model"

# # Transformer

# echo "Starting training Transformer model" | s-nail -s "Training Notification on step 5" jjsnam@zju.edu.cn
# echo "Starting training Transformer model"
# echo "Changing direction"
# echo "Sending notification email..."

# echo "Starting training Transformer model on SGDF dataset" | s-nail -s "Training Notification on step 5.1" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name SGDF --train_path /root/Project/datasets/SGDF/Train --val_path /root/Project/datasets/SGDF/Val --model_path /root/Project/weights/Transformer/SGDF --lr 1e-4 || echo "Training failed: Transformer-SGDF" | s-nail -s "Failure: Transformer-SGDF" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training Transformer model on OpenForensics dataset" | s-nail -s "Training Notification on step 5.2" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name OpenForensics --train_path /root/Project/datasets/OpenForensics/Train --val_path /root/Project/datasets/OpenForensics/Val --model_path /root/Project/weights/Transformer/OpenForensics --lr 1e-4 || echo "Training failed: Transformer-OpenForensics" | s-nail -s "Failure: Transformer-OpenForensics" jjsnam@zju.edu.cn
# echo "Sending notification email..."

# echo "Starting training Transformer model on 200kMDID dataset" | s-nail -s "Training Notification on step 5.3" jjsnam@zju.edu.cn
# python main.py --epochs 50 --dataset_name 200kMDID --train_path /root/Project/datasets/200kMDID/Train --val_path /root/Project/datasets/200kMDID/Val --model_path /root/Project/weights/Transformer/200kMDID --lr 1e-4 || echo "Training failed: Transformer-200kMDID" | s-nail -s "Failure: Transformer-200kMDID" jjsnam@zju.edu.cn
# echo "Finish training Transformer model"


# echo "Sending notification email..."
# echo "All model training completed successfully." | s-nail -s "Training Notification - All Done" jjsnam@zju.edu.cn

# echo "Begin to shutdown"
# shutdown