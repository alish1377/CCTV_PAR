Project Overview
================
Pedestrian Attribute Recognition (PAR) is a task widely used for person Re-Identification (ReID) applications. The UPAR Challenge 2024 is a well-known challenge in this field, in which Channel-Aware Cross-Fused Transformer-style Networks (C2T-Net) achieved 1st place. 

In this project, I use C2T-Net as the baseline model. The main innovations of this work are:

1. Applying 3 types of fine-tuning:
    - Full Fine-Tuning: All model parameters are updated.
    - Partial Fine-Tuning: Only the cross-fusion layers are updated.
    - Head Fine-Tuning: Only the fully connected (FC) classification head layers are updated.
2. Loss optimization and modification:
    - Categorical Loss: Introduces category-level supervision to better capture attribute group dependencies.
    - Sample Weighting: Adjusts loss contributions based on the imbalance of positive and negative label distributions.
    - Logit Updating: Refines predicted logits using recall information from positive and negative labels.
    - GradNorm: Learns separate weights for each attribute in addition to learning the base model weights.
    - Focal Loss: Reweights the loss to focus learning on harder, less frequent samples.

Dataset
=======
The dataset was introduced in the Sharif_PAR repository. You can download it directly from that repo. However, I have already included the data in the `.\data\NATIVE` directory. Therefore, you do not need to download the dataset separately for training or inference.

Configuration
=============
Prerequisites:
- CUDA 11.8
- Python 3.11.8

Installation:
The following steps outline the setup process for both training and inference:

1. Install `uv` using the official installation instructions (I used the `pip install uv` command).
2. Create a virtual environment: 
   uv venv

3. Activate the virtual environment:
    - Windows: `.\.venv\Scripts\activate`
    - Mac/Linux: `source .venv/bin/activate`

4. Install the requirements (compatible with CUDA):
   uv pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu118

5. Install `mmcv-full`:
   uv pip install mmcv-full==1.7.2 -f https://download.openmmlab.com/mmcv/dist/cu118/torch2.1/index.html

Execution
=========
For both training and inference, you need to execute the `run.py` script:

uv run run.py --config-file .\configs\native.yaml

The `.\configs\native.yaml` file defines your configuration. Inside it, you can adjust parameters related to the dataset, reloading, backbone, classifier, etc.

Inside `run.py`, set the following variables based on your goal:
- For training: Set `run_file = "train_upar_2024.py"`
- For inference: Set `run_file = "test_video.py"`
- To execute using GPU: Set `os.environ["CUDA_VISIBLE_DEVICES"] = "0"`
- To execute using CPU: Set `os.environ["CUDA_VISIBLE_DEVICES"] = ""`

Training Details
================
For training, you can configure the model parameters inside the `configs\native.yaml` file. The results of each experiment will be saved in the `exp_results\NATIVE\[name_of_experiment]` directory. 

To use pretrained weights, set `RELOAD -> TYPE` to `True`. To define the directory containing your pretrained weights, update `RELOAD -> MODEL_FOLDER_NAME` with your target folder's name.

Inference Details
=================
We test our model on videos! You can configure the inference parameters in the `TEST` section of the `configs\native.yaml` file. This includes parameters like the detection model, the output path, and other related settings. The target video name should be specified inside the `test_video.py` file.

Evaluation Results
=================

Based on our experiments, we evaluated different fine-tuning and optimization strategies. The results are summarized below.

### Table 2: Fine-Tuning Strategies Comparison

| Strategy | attr_acc (%) | mean_acc (%) | pos_recall (%) | neg_recall (%) |
| :--- | :--- | :--- | :--- | :--- |
| Full fine-tuning | $96.40\pm0.05$ | $80.79\pm2.52$ | $64.75\pm4.49$ | $96.90\pm0.58$ |
| Partial fine-tuning | $95.99\pm0.1$ | $78.96\pm0.39$ | $61.87\pm0.41$ | $96.06\pm0.65$ |
| Head fine-tuning | $96.05\pm0.23$ | $79.91\pm1.68$ | $63.05\pm3.11$ | $96.77\pm0.31$ |

*Note: Full fine-tuning outperforms partial and head fine-tuning and is used as the baseline for further optimization.*

### Table 3: Comparison of Optimization Strategies

| Strategy | attr_acc (%) | mean_acc (%) | pos_recall (%) | neg_recall (%) |
| :--- | :--- | :--- | :--- | :--- |
| Full fine-tuning (basis) | $96.40\pm0.05$ | $80.79\pm2.52$ | $64.75\pm4.49$ | $96.90\pm0.58$ |
| Categorical loss | $95.65\pm0.14$ | $79.81\pm1.7$ | $63.29\pm3.17$ | $95.94\pm0.25$ |
| Loss_weight | $96.26\pm0.29$ | $84.22\pm0.39$ | $71.37\pm1.17$ | $97.06\pm0.39$ |
| Loss_weight + Update_logits | $96.36\pm0.26$ | $83.64\pm1.29$ | $70.43\pm2.64$ | $96.85\pm0.38$ |
| Loss_weight + GradNorm | $96.13\pm0.13$ | $85.27\pm0.68$ | $74.35\pm1.42$ | $96.19\pm0.07$ |
| Loss_weight + Focal loss | $96.50\pm0.09$ | $84.15\pm1.6$ | $70.84\pm3$ | $97.45\pm0.2$ |

*Note: The combination of Loss_weight and GradNorm provides the best overall performance. It improves positive recall by $9.6\%$ and mean accuracy by $4.48\%$ compared to the baseline.*



