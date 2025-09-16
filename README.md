# A Comparative Study of Deep Learning and Traditional Models for Human Face Image Deepfake Detection

## Introduction

Hi there! This is a research to compare different models' performance on deepfake human face image detection.

## Requirements

Our experiment was performed on a V100 GPU provided by [GPU Share](https://gpushare.com/).

Key information includes:
-   Platform: Tesla V100-PCIE-16GB
-   CUDA Version: 12.1
-   CPU: Intel Core Processor (Broadwell, IBRS) 12 cores
-   Memory: 119GB

And for environments:
-   Pytorch 2.4.0
-   Python 3.11

## Dataset Sources

All of our datasets originates from [Kaggle](https://www.kaggle.com/). They are:

-   SGDF: [StyleGan-StyleGan2 Deepfake Face Images](https://www.kaggle.com/datasets/kshitizbhargava/deepfake-face-images)
-   OpenForensics: [deepfake and real images](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images)
-   200kMDID: [GRAVEX-200K](https://www.kaggle.com/datasets/muhammadbilal6305/200k-real-vs-ai-visuals-by-mbilal)

Please note that we renamed these three datasets to SGDF, OpenForensics and 200kMDID when doing our experiment.

Besides, also a Fused dataset are generated from these datasets, which will be mentioned later.

## Usage

### Clone the repository
use:

```bash
git clone --depth=1 https://github.com/jjsnam/DFD-Comparison
```

but not just `git clone` since we previously added some weights to make the repository much bigger.

### Install dependencies

```bash
pip install -r requirements.txt
```

### Dataset Processing

You shall first download the datasets from the [sources](#dataset-sources) mentioned above.

After downloading the datasets, we highly recommend that you put them under the `datasets/` directory and then rename them.

Some processing are needed to make the datasets usable:

- Run `datasets/utils/split_dataset.py` to make SGDF dataset as a formal dataset containing Train/Val/Test directory.
- Run `datasets/utils/split_by_csv.py` to make 200kMDID dataset usable from the original csv labels.
- Run `datasets/utils/fuse_dataset.py` to form the Fused dataset from the three **processed** datasets.

You may need to change some arguments to fit the correct paths in these codes.

Besides, RCNN model requires face region extraction, which shall be done before training and testing. You can change to `RCNN Models/` directory and run `extract_regions.py` (the arguments needs your change) to extract face regions including full/eye/mouth.

### Training

#### Prerequisite

Please assure that the commented arguments in `Transformer/config.py` is uncommented (Sorry for not organizing the code more precisely)

And transformer needs pretrained ViT model from Hugging face. More specifically, you need to download `vit_base_patch16_224.pth`. The pretrained weights can also be downloaded [here](#weights). After downloading, please put the weights under `Transformer/weights/` directory and change the argument `checkpoint_path` in `Transformer/config.py`.

#### Running training part

`train_basic/fused.sh` contains a compositive training commands to run in Linux environments (except those in `ML Models/`). You can try to run the scripts to reproduce the whole training process.

The commands for training each model are:

-   CNN/CNN+Transformer: 
    ```bash
    python train.py --epochs ${num of epochs} --dataset_name ${dataset name} --train_path ${train path} --val_path ${val path}--model_path ${model save path} --lr ${learning rate}
    ```

-   RCNN:
    ```bash
    python train.py --epochs ${num of epochs} --dataset_name ${dataset name} --train_path ${train path} --val_path ${val path}  --train_cache_path ${train region cache path} --val_cache_path ${val region cache path}  --model_path ${model save path} --lr ${learning rate}
    ```

-   Statistical (GMM):
    ```bash
    python main.py --epochs ${num of epochs} --dataset_name ${dataset name} --train_path ${train path} --val_path ${val path}   --model_path ${model save path}
    ```

-   Transformer:
    ```bash
    python main.py --epochs ${num of epochs} --dataset_name ${dataset name} --train_path ${train path} --val_path ${val path}  --model_path ${model save path} --lr ${learning rate}
    ```

-   ML (SVC):

Note that if you choosed to run them manually, remember to change your direction before running codes.

### Testing

#### Prerequisite

Please assure that the uncommented arguments in `Transformer/config.py` when training is again commented. (Sorry for not organizing the code more precisely)

ViT pretrained model is also needed as mentioned in training part.

#### Running testing part

`test.sh` contains a compositive testing commands to run in Linux environments (except those in `ML Models/`). You can try to run the scripts to reproduce the whole testing process.

The commands for training each model are:

-   CNN/CNN+Transformer/Transformer:
    ```bash
    python test.py --model_path ${test model path} --test_path ${test dataset path}
    ```

-   RCNN:
    ```bash
    python test.py --model_path ${test model path} --test_root ${test dataset path} --cache_root ${test region cache}
    ```

-   Statistical (GMM):
    ```bash
    python test.py --model_real ${real model path} --model_fake ${fake model path}$ --test_path ${test dataset path}
    ```

-   ML (SVC):




## Results

All the key results are in the `results/` directory in log format.

### Result Processing

`results/utils/` provides some codes to deal with the chaotic log files to make the result much clearer.

For training results, run:

```bash
python parse_deepfake_logs_train.py --log ${/path/to/train-log} --outdir ${/path/to/outdir}
```

and you may get the results processed.

Analogously, for testing results, run:

```bash
python parse_deepfake_logs_test.py --log ${/path/to/test-log} --outdir ${/path/to/outdir}
```

### Weights

To get our training weights, you can follow [this link]().

## Citation

?

## License

?

## Acknowledgements

We would like to express our sincere gratitude to:

- Prof. David Woodruff (Carnegie Mellon University), for his guidance and supervision.  
- Yupei Li, for his support and assistance as the teaching assistant.  
- Neoscholar Education, for coordinating the project and providing organizational support.  
- ChatGPT (OpenAI), for assisting in coding, debugging, and documentation.  
- SGDF, OpenForensics and 200kMDID datasets' producers and their source datasets' producers

## Contributors

- [Shiqi Yao (Soochow University)](https://github.com/GNETUX): Team leader; responsible for paper writing and implementation of the SVC model.  
- [Tianyang Liu (Zhejiang University)](https://github.com/jjsnam): Main contributor of coding and implementation; responsible for most of the experimental pipeline and result processing.  
- [Zeyu Jiang (Xi'an Jiaotong-Liverpool University)](https://github.com/Serendipity0319) and [Jinxi Li (Zhejiang University)](https://github.com/LJX-xixi): Paper writing, result analysis, and code enhancements.  