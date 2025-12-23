# Project Ovierview
Pedestrain Attribute Recognition(PAR) is the task which is used for person Re-Identification (ReID) application. [UPAR Challenge 2024](https://openaccess.thecvf.com/content/WACV2024W/RWS/papers/Cormier_UPAR_Challenge_2024_Pedestrian_Attribute_Recognition_and_Attribute-Based_Person_Retrieval_WACVW_2024_paper.pdf) is the well-known challenge in this field, and [Channel-Aware Cross-Fused Transformer-style Networks (C2T-Net)](https://github.com/caodoanh2001/upar_challenge) achieved 1st place in this challenge. In this project, I will use C2T_Net as the baseline model. The main innovation of this work is:
1. Applying 3 types of fine-tuning:
    - **Full Fine-Tuning**: All model parameters are updated.
    - **Partial Fine-Tuning**: Only the cross-fusion layers are updated.
    - **Head Fine-Tuning**: Only the fully connected (FC) classification head layers are updated.
2. Loss improvement techniques:
    - **Categorical Loss**: Introduces category-level supervision to better capture attribute group dependencies.
    - **Sample Weighting**: Adjusts loss contributions based on the imbalance of positive and negative label distributions.
    - **Logit Updating**: Refines predicted logits using recall information from positive and negative labels.
    - **GradNorm**: Learns separate weights for each attribute in addition to learning the base model weights.
    - **Focal Loss**: Reweights the loss to focus learning on harder, less frequent samples.

# Dataset
The dataset is proposed in [Sharif_PAR](https://github.com/SharifDeepLab/Sharif_PAR) repository. You could download it directly from this repo. Also, I put the data in ```.\data\NATIVE``` directory. So for training or inference of this project, you dont need to download the dataset.

# Configuration
## Prerequisite
- ```CUDA 11.8```
- ```Python 3.11.8```
## Installation
The following steps is the configuration for training and inference goals:
1. Install ```uv``` using [uv_installation](https://docs.astral.sh/uv/getting-started/installation/#standalone-installer) instruction, I used ```pip install uv``` command
2. Create ```.venv``` dir: ```uv venv```
3. Activate ```.venv```:
    - **windows**: ```.venv\Scripts\activate```
    - **mac/linux**: ```source .venv/bin/activate```
4. Install requirements(compatible with CUDA):
- ```uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118```
5. Install mmcv_full:
- ```uv pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html```

## Execution
For both training and inference, you should execute ```run.py``` file.
- ```uv run --config-file .\configs\native.yaml```
- ```.\configs\native.yaml``` defines the ```--config-file```. You could adjust parameters related to dataset, reloading, backbone, classifier, and etc.
- In ```run.py```:
    - For training: set ```run_file = "train_upar_2024.py"```
    - For inference: set ```run_file = "test_video.py"```
    - Execute using GPU: set ```os.environ["CUDA_VISIBLE_DEVICES"] = "0"```
    - Execute using CPU: set ```os.environ["CUDA_VISIBLE_DEVICES"] = ""```

## Training details
For training, you could set model parameters in ```configs\native.yaml``` file. The result of each experiment would be saved in ```exp_results\NATIVE\[name_od_experiment]``` directory. For using the pretrained weight, set ```RELOAD -> TYPE``` to True. Also for defining the folder of pretrained weights, set ```RELOAD -> MODEL_FOLDER_NAME``` with your target folder. 

## Inference details
We test our model on videos! You could set the inference parameters in ```TEST``` section in ```configs\NATIVE.yaml``` file. parameters like the detection model, video name in ```test_video.py``` file, output_path, and other related parameters.

