import argparse
import json
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import pickle

from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from models.backbone import swin_transformer2
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
from tqdm import tqdm
import time
from timeit import default_timer as timer

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr,PedesAttrPETA
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from losses import bceloss, scaledbceloss

import torchvision.transforms as T

import onnx
import onnxruntime

set_seed(605)


def main(cfg, args):
    exp_dir = os.path.join('exp_result', "UPAR")
    model_dir, log_dir = get_model_log_path(exp_dir, "_test1_usePretrained")

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    # Start time
    start_time = timer()

    # train_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
    #                         target_transform=cfg.DATASET.TARGETTRANSFORM)
    # valid_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
    #                         target_transform=cfg.DATASET.TARGETTRANSFORM)



    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=40,
        c_in=2048,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier, use_sigmoid=True)
    
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    print(model_dir)
    sum = 0
    print("##########################################################")
    model = get_reload_weight(model_dir, model, pth="best_model.pth")
    model.eval()

    # Create input tensor
    batch_size = 1
    # Assuming input image size is (256, 192) and 3 channels
    x = torch.ones((batch_size, 3, 300, 400), dtype=torch.float32) - 0.5

    normalize = T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = T.Compose([
        T.Resize((256, 128)),
        normalize
    ])
    
    x = test_transform(x)
    valid_probs = model(x)

    # Export the model to ONNX format
    input_names = ["input"]
    output_names = ["output"]
    
    # Export the model
    # onnx_program =  torch.onnx.dynamo_export(model,x) 
    # onnx_program.save("swin_b.onnx")

    torch.onnx.export(model, x, "C2T_Net.onnx", opset_version=14,
                     input_names=['input'], output_names=['output'],
                     dynamic_axes={
                        'input': {0: 'batch_size'},
                        'output': {0: 'batch_size'}
                        }
                     )
    onnx_model = onnx.load("C2T_Net.onnx")

    # Get the input and output node names
    output =[node.name for node in onnx_model.graph.output]

    input_all = [node.name for node in onnx_model.graph.input]
    input_initializer =  [node.name for node in onnx_model.graph.initializer]
    net_feed_input = list(set(input_all)  - set(input_initializer))

    print('Inputs: ', net_feed_input)
    print('Outputs: ', output)
    print("data")
    for node in net_feed_input:
        print(node)

    # onnx check the model
    onnx.checker.check_model(onnx_model)

    #onnx_input = onnx_program.adapt_torch_inputs_to_onnx(x)
    ort_session = onnxruntime.InferenceSession("C2T_Net.onnx", providers=["CPUExecutionProvider"])
    
    def to_numpy(tensor):
        print("required_grad", tensor.requires_grad)
        return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()
    
    # compute ONNX Runtime output prediction
    ort_inputs = {ort_session.get_inputs()[0].name: to_numpy(x).astype(np.float32) }
    ort_outs = ort_session.run(None, ort_inputs)
    
    # compare ONNX Runtime and PyTorch results
    print("outputs")
    print(ort_outs[0])
    print(ort_outs[0].shape)
    print(valid_probs)
    np.testing.assert_allclose(to_numpy(valid_probs), ort_outs[0], rtol=1e-03, atol=1e-05)
    
    print("Exported model has been tested with ONNXRuntime, and the result looks good!")


def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
