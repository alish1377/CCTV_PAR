import argparse
import json
import os
import random

from jupyterlab.semver import valid

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from dataset.pedes_attr.manual_pedes import ManualPedes
from metrics.ml_metrics import get_map_metrics, get_multilabel_metrics
from models.base_block import FeatClassifier
# from models.model_factory import model_dict, classifier_dict

from tools.function import get_model_log_path, get_reload_weight
from tools.utils import set_seed, str2bool, time_str
from tools.hist import save_hists
from tools.check import check_images
from tools.color_mixture import save_upper_lower_colors
from losses import bceloss, scaledbceloss
from savefig.size_instance_acc.size_acc import plot_size_acc


import pandas as pd

set_seed(605)


def main(cfg, args):
    exp_dir = os.path.join('exp_result', cfg.DATASET.NAME)
    model_dir, log_dir = get_model_log_path(exp_dir, cfg.RELOAD.MODEL_FOLDER_NAME)

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if cuda_visible == "":
        DEVICE = torch.device("cpu")
    else:
        DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Start time
    start_time = timer()

    if cfg.DATASET.NAME == "NATIVE":
        assert (cfg.DATASET.TRAIN_NATIVE_SPLIT + cfg.DATASET.VAL_NATIVE_SPLIT + \
        cfg.DATASET.TEST_NATIVE_SPLIT) == 1.0, "Invalid Split process"

        df = pd.read_excel(cfg.DATASET.NATIVE_EXCEL) 

        # Filter the DataFrame to exclude rows where Column2 has the value 2
        if cfg.DATASET.REMOVE_LOW_QUALITY:
          df = df[df['Quality'] != "low"]
          
        df = df.fillna(0)
        df_index = df.index.tolist()
        #random.shuffle(df_index)  
        n = len(df_index)

        train_end = int(cfg.DATASET.TRAIN_NATIVE_SPLIT  * n)
        val_end = int((cfg.DATASET.TRAIN_NATIVE_SPLIT + cfg.DATASET.VAL_NATIVE_SPLIT) * n)

        # Split the list
        train_list = df_index[:train_end]
        val_list = df_index[train_end:val_end]
        test_list = df_index[val_end:]

        root_path = cfg.DATASET.IMAGE_PATH

        attrs = cfg.DATASET.ATTRS

        train_set = ManualPedes(df, train_list, root_path, train_tsfm, attrs)
        valid_set = ManualPedes(df, val_list, root_path, valid_tsfm, attrs)
        test_set = ManualPedes(df, test_list, root_path, valid_tsfm, attrs)

    else:
        train_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                                target_transform=cfg.DATASET.TARGETTRANSFORM)
        valid_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                                target_transform=cfg.DATASET.TARGETTRANSFORM)

    print("length of validation set")
    print(len(valid_set))


    # Create a SubsetRandomSampler using the subset indices
    # subset_sampler = SubsetRandomSampler(subset_indices)

    valid_loader = DataLoader(
        dataset=valid_set,
        batch_size=cfg.TRAIN.BATCH_SIZE,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    print(f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}, '
          f'attr_num : {len(train_set.attrs)}')

    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)


    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=len(attrs),
        c_in=2*c_output,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale =cfg.CLASSIFIER.SCALE
    )

    model = FeatClassifier(backbone, classifier)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()

    print(model_dir)
    model = get_reload_weight(model_dir, model, pth="best_model.pth")



    model.eval()
    preds_probs = []
    gt_list = []
    # Define 2 variables for saving the x and y size for accuracy calculation
    path_list = []

    attn_list = []
    with torch.no_grad():
        for (imgs, gt_label, _) in tqdm(valid_loader):
            imgs = imgs.to(DEVICE)
            gt_label = gt_label.to(DEVICE)
            # Change second argument from gt_label to None
            valid_logits, _ = model(imgs, None)

            valid_probs = torch.sigmoid(valid_logits[0])

            gt_list.append(gt_label.cpu().numpy())
            preds_probs.append(valid_probs.cpu().numpy())


    gt_label = np.concatenate(gt_list, axis=0)
    preds_probs = np.concatenate(preds_probs, axis=0)
    print("ground truth shape")
    print(gt_label.shape)
    print("prediction shape")
    print(preds_probs.shape)
    # attn_list = np.concatenate(attn_list, axis=0)
    # End time
    end_time = timer()

    # Elapsed time
    print(f"Elapsed time: {end_time - start_time}")

    
    if cfg.METRIC.TYPE == 'pedestrian':
        valid_result = get_pedestrian_metrics(gt_label, preds_probs)
        valid_map, _ = get_map_metrics(gt_label, preds_probs)

        print(f'Evaluation on test set, \n',
              'attr_mean_acc: {:.4f}, ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  np.mean(valid_result.attr_acc), valid_result.ma, valid_map, np.mean(valid_result.label_f1), np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1)
              )
        save_hists(valid_result.label_pos_recall, valid_result.label_neg_recall, valid_result.attr_acc, attrs)

        # with open(os.path.join(model_dir, 'results_test_feat_best.pkl'), 'wb+') as f:
        #     pickle.dump([valid_result, gt_label, preds_probs, attn_list, path_list], f, protocol=4)

    elif cfg.METRIC.TYPE == 'multi_label':
        if not cfg.INFER.SAMPLING:
            valid_metric = get_multilabel_metrics(gt_label, preds_probs)

            print(
                'Performance : mAP: {:.4f}, OP: {:.4f}, OR: {:.4f}, OF1: {:.4f} CP: {:.4f}, CR: {:.4f}, '
                'CF1: {:.4f}'.format(valid_metric.map, valid_metric.OP, valid_metric.OR, valid_metric.OF1,
                                     valid_metric.CP, valid_metric.CR, valid_metric.CF1))

            with open(os.path.join(model_dir, 'results_train_feat_baseline.pkl'), 'wb+') as f:
                pickle.dump([valid_metric, gt_label, preds_probs, attn_list, path_list], f, protocol=4)

        print(f'{time_str()}')
        print('-' * 60)

def argument_parser():
    parser = argparse.ArgumentParser(description="attribute recognition",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        "--cfg", default='./configs/multilabel_baseline/coco.yaml', help="decide which cfg to use", type=str,
    )
    parser.add_argument("--debug", type=str2bool, default="true")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = argument_parser()
    update_config(cfg, args)

    main(cfg, args)
