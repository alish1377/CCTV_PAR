import argparse
import json
import os
import random
import math

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import pickle

from dataset.augmentation import get_transform
# from dataset.multi_label.coco import COCO14
from metrics.pedestrian_metrics import get_pedestrian_metrics
from metrics.box_metric import compute_mean_accuracy
from models.backbone import swin_transformer2
from models.model_factory import build_backbone, build_classifier

import numpy as np
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import torch.nn.functional as F
from tqdm import tqdm
import time
from timeit import default_timer as timer

from configs import cfg, update_config
from dataset.pedes_attr.pedes import PedesAttr, PedesAttrPETA
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
from tools.plot_tubes_losses_accs import plot_resolution_brightness_vs_loss, plot_resolution_brightness_vs_acc
from tools.plot_resolution_brightness import plot_resolution_brightness
from tools.plot_zscore import plot_zscore_resolution_brightness
from tools.boxes_selection import select_best_boxes
from tools.plot_mean_losses_accs import plot_mean_losses, plot_mean_accs
from tools.tube_utils import create_image_dict

import pandas as pd

set_seed(605)


def main(cfg, args):
    #exp_dir = os.path.join('exp_result', "UPAR")
    #model_dir, log_dir = get_model_log_path(exp_dir, "_test1_usePretrained")

    train_tsfm, valid_tsfm = get_transform(cfg)
    print(valid_tsfm)

    # Start time
    start_time = timer()

    if cfg.DATASET.NAME == "NATIVE":
        exp_dir = os.path.join('exp_result', "NATIVE")
        model_dir, log_dir = get_model_log_path(exp_dir, "swin_b_logitscale_TEST3")

        assert (cfg.DATASET.TRAIN_NATIVE_SPLIT + cfg.DATASET.VAL_NATIVE_SPLIT + \
                cfg.DATASET.TEST_NATIVE_SPLIT) == 1.0, "Invalid Split process"

        df = pd.read_excel(cfg.DATASET.NATIVE_EXCEL)

        # Filter the DataFrame to exclude rows where Column2 has the value 2
        if cfg.DATASET.REMOVE_LOW_QUALITY:
            df = df[df['Quality'] != "low"]

        df = df.fillna(0)
        df_index = df.index.tolist()
        random.shuffle(df_index)
        n = len(df_index)

        train_end = int(cfg.DATASET.TRAIN_NATIVE_SPLIT * n)
        val_end = int((cfg.DATASET.TRAIN_NATIVE_SPLIT + cfg.DATASET.VAL_NATIVE_SPLIT) * n)

        # Split the list
        train_list = df_index[:train_end]
        val_list = df_index[train_end:val_end]
        test_list = df_index[val_end:]

        root_path = cfg.DATASET.IMAGE_PATH
        root_tube_path = cfg.DATASET.TUBE_PATH

        attrs = cfg.DATASET.ATTRS



        train_sets, valid_sets, test_sets = {}, {}, {}
        #tube_names = {'tube1': 'c1_v1_191_490', 'tube2': 'c1_v1_23_37', 'tube3': 'c1_v1_196_628', 
        #              'tube4': 'c2_v2_15_1','tube5': 'c2_v2_72_6',
        #              'tube6': 'c2_v2_189_13', 'tube7': 'c2_v2_214_21', 
        #              'tube8': 'c2_v11_121_223', 'tube9': 'c2_v2_143_205', 'tube10': 'c2_v11_17_28'}

        all_tube_folder = "data/NATIVE/new_original_test_tubes"
        tube_names = create_image_dict(all_tube_folder)
    

        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff', '.webp')
        tube_paths = os.listdir(root_tube_path)

        tube_paths.sort(key=lambda name: int(name.replace("tube", "")))
        for i, tube_path in enumerate(tube_paths):
            ids_path = os.path.join(root_tube_path, tube_path)
            #print(ids_path)
            #train_set = ManualPedes(df, train_list, ids_path, train_tsfm, attrs)
            tube_idx = f'tube{i+1}'
            box_count = sum(
                1 for file in os.listdir(ids_path)
                if file.lower().endswith(image_extensions)
            )      
            print("box_count")
            print(box_count)
            #print(df['ID'])
            #print(tube_names[tube_idx])
            val_list = df.index[df['ID'] == (tube_names[tube_idx].split(".")[0])].tolist() * box_count
            #print("val_list")
            #print(val_list)
            #print(len(val_list))
            valid_set = ManualPedes(df, val_list, ids_path, valid_tsfm, 
                                    attrs, imgsize=cfg.DATASET.ACC_WITH_SIZE, tube_number=i+1)
            #test_set = ManualPedes(df, test_list, ids_path, valid_tsfm, attrs)

            #train_sets.append(train_set)
            valid_sets[ids_path] = valid_set
            #test_sets.append(test_set)
        # print("check the manual dataset")
        # check_data, check_label, _ = valid_set[6]
        # print(check_data.size())
        # print(check_label)


    else:
        train_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.TRAIN_SPLIT, transform=valid_tsfm,
                                  target_transform=cfg.DATASET.TARGETTRANSFORM)
        valid_set = PedesAttrPETA(cfg=cfg, split=cfg.DATASET.VAL_SPLIT, transform=valid_tsfm,
                                  target_transform=cfg.DATASET.TARGETTRANSFORM)

    #print("length of validation set")
    #print(len(valid_sets))
    #print(valid_sets)

    # train_loader = DataLoader(
    #     dataset=train_set,
    #     batch_size=cfg.TRAIN.BATCH_SIZE,
    #     shuffle=False,
    #     num_workers=4,
    #     pin_memory=True,
    # )

    # Define the indices of the subset you want to use
    # num_images_test = 5000
    # subset_indices = torch.randperm(len(valid_set))[:num_images_test]
    # print("subset_indices")
    # print(subset_indices)

    # Create a SubsetRandomSampler using the subset indices
    # subset_sampler = SubsetRandomSampler(subset_indices)
    backbone, c_output = build_backbone(cfg.BACKBONE.TYPE, cfg.BACKBONE.MULTISCALE)
    strict = True
    if cfg.DATASET.REMOVE_REDUNDANT_ATTRIBUTES:
        nattr = 37
        strict = False
    else:
        nattr = 40
    classifier = build_classifier(cfg.CLASSIFIER.NAME)(
        nattr=nattr,
        c_in=2048,
        bn=cfg.CLASSIFIER.BN,
        pool=cfg.CLASSIFIER.POOLING,
        scale=cfg.CLASSIFIER.SCALE
    )
    model = FeatClassifier(backbone, classifier)
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    print(model_dir)
    model = get_reload_weight(model_dir, model, pth="best_model.pth", strict=strict)
    

    mean_lower_losses, mean_upper_losses = [], []  # The mean losses for upper and lower attributes for n tubes(mean of mean)
    mean_rectified_lower_losses, mean_rectified_upper_losses = [], [] # The mean rectified losses for upper and lower attributes for n tubes(mean of mean)

    mean_lower_accs, mean_upper_accs = [], []
    mean_rectified_lower_accs, mean_rectified_upper_accs = [], []

    for tube_number, (_, valid_set) in enumerate(valid_sets.items()):
        #print("***********")
        #print(valid_set)
        valid_loader = DataLoader(
            dataset=valid_set,
            batch_size=1,
            # sampler=subset_sampler,
            shuffle=False,
            num_workers=1,
            # pin_memory=True,
        )

        print(f'{cfg.DATASET.TEST_SPLIT} set: {len(valid_loader.dataset)}')


        model.eval()
        preds_probs = []
        gt_list = []
        # Define 2 variables for saving the x and y size for accuracy calculation
        imgsize_list_x = []
        imgsize_list_y = []

        image_resolution = []
        image_brightness = []

        normalized_image_resolution = []
        normalized_image_brightness = []
        resolution_initial = 0
        brightness_initial = 0

        upper_losses = []
        lower_losses = []

        upper_accs = []
        lower_accs = []
        
        path_list = []

        attn_list = []
        box_features = []

        with torch.no_grad():
            for step, (imgs, gt_label, imgsize, v_array) in enumerate(tqdm(valid_loader)):
                for x, y in zip(imgsize[0], imgsize[1]):
                    image_resolution.append(x*y)
                    image_brightness.append(v_array.item())
                    if step == 0:   
                        resolution_initial = x*y
                        brightness_initial = v_array.item()
                        normalized_image_resolution.append(1)
                        normalized_image_brightness.append(1)
                    else:
                        normalized_image_resolution.append(math.sqrt((x*y)/resolution_initial))
                        normalized_image_brightness.append(v_array.item()/brightness_initial)
                    
                    imgsize_list_x.append(x.item())
                    imgsize_list_y.append(y.item())
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    gt_label = gt_label.cuda()
                # Change second argument from gt_label to None
                valid_logits, _ = model(imgs, None)
                valid_logits = valid_logits[0]
                if cfg.DATASET.NAME == "NATIVE" and not cfg.DATASET.REMOVE_REDUNDANT_ATTRIBUTES:
                    valid_logits = valid_logits[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 35, 36, 37, 38, 39]]
                    #valid_logits = valid_logits[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 34, 35, 36, 37, 38, 39]]

                valid_probs = torch.sigmoid(valid_logits)
                #return
                # loss_m shape would be (#images in the batch, #attrs)
                loss_m = F.binary_cross_entropy_with_logits(valid_logits, gt_label, reduction='none')
                loss_m = loss_m.cpu().numpy()

                upper_indices = [8,9,10,11,12,13,14,15,16,17,18]
                upper_loss_m = loss_m[:, upper_indices].sum(axis=1)
                upper_losses.append(upper_loss_m)
                upper_accs.append(compute_mean_accuracy(gt_label[:, upper_indices], 
                                                      valid_probs[:, upper_indices]))

                lower_indices = [20,21,22,23,24,25,26,27,28,29,30]
                lower_loss_m = loss_m[:, lower_indices].sum(axis=1)
                lower_losses.append(lower_loss_m)
                lower_accs.append(compute_mean_accuracy(gt_label[:, lower_indices], 
                                                      valid_probs[:, lower_indices]))

                # loss_m_sum shape would be (#images in the batch, )
                loos_m_sum = loss_m.sum(1)
                    # check_images(step, cfg.TRAIN.BATCH_SIZE, imgs, gt_label, valid_logits.cpu().numpy(), attrs, check_what="U_white")
                    # save_upper_lower_colors(step, cfg.TRAIN.BATCH_SIZE, imgs, gt_label, valid_logits.cpu().numpy(), attrs)
                # path_list.extend(imgname)
                gt_list.append(gt_label.cpu().numpy())
                preds_probs.append(valid_logits.cpu().numpy())
                # attn_list.append(attns.cpu().numpy())
            print("Upper loss length: ", len(upper_losses))
            print("Lower loss length: ", len(lower_losses))

            selection_method = "both_zscore"
            brightness_zscores, resolution_zscores, rectified_lower_losses, rectified_upper_losses, rectified_lower_accs, rectified_upper_accs = \
            select_best_boxes(upper_losses, lower_losses, upper_accs, lower_accs,
                              normalized_image_resolution, normalized_image_brightness,
                              cfg.BOX_SELECTION.RESOLUTION_ALPHA,
                              cfg.BOX_SELECTION.BRIGHTNESS_BETA,
                              selection_method=selection_method)

            plot_resolution_brightness(normalized_image_resolution, normalized_image_brightness, 
                                       save_path=f"./savefig/resolution_brightness/tube{tube_number+1}")
            plot_zscore_resolution_brightness(resolution_zscores, brightness_zscores,
                                              save_path=f"./savefig/resolution_brightness_zscore/tube{tube_number+1}")
            
            plot_resolution_brightness_vs_loss(normalized_image_resolution, normalized_image_brightness, 
                                                upper_losses, lower_losses,
                                                save_path = f"./savefig/resolution_brightness_vs_losses/tube{tube_number+1}")

            plot_resolution_brightness_vs_acc(normalized_image_resolution, normalized_image_brightness, 
                                    upper_accs, lower_accs,
                                    save_path = f"./savefig/resolution_brightness_vs_accs/tube{tube_number+1}")


            mean_lower_loss, mean_upper_loss = np.mean(lower_losses), np.mean(upper_losses)
            mean_rectified_lower_loss, mean_rectified_upper_loss = np.mean(rectified_lower_losses), np.mean(rectified_upper_losses)

            mean_lower_acc, mean_upper_acc = np.mean(lower_accs), np.mean(upper_accs)
            mean_rectified_lower_acc, mean_rectified_upper_acc = np.mean(rectified_lower_accs), np.mean(rectified_upper_accs)

            mean_lower_losses.append(mean_lower_loss)
            mean_upper_losses.append(mean_upper_loss)
            mean_rectified_lower_losses.append(mean_rectified_lower_loss)
            mean_rectified_upper_losses.append(mean_rectified_upper_loss)

            mean_lower_accs.append(mean_lower_acc)
            mean_upper_accs.append(mean_upper_acc)
            mean_rectified_lower_accs.append(mean_rectified_lower_acc)
            mean_rectified_upper_accs.append(mean_rectified_upper_acc)

            print("zscores")
            print("brightness_zscores")
            print(np.mean(brightness_zscores))
            print("resolution_zscores")
            print(np.mean(resolution_zscores))
            

    print("Losses difference....")
    print("overall upper losses")
    print(mean_lower_losses)
    print("overall lower losses")
    print(mean_upper_losses)
    print("rectified upper losses")
    print(mean_rectified_lower_losses)
    print("rectified lower losses")
    print(mean_rectified_upper_losses)


    print("Accs difference....")
    print("overall upper accs")
    print(mean_lower_accs)
    print("overall lower accs")
    print(mean_upper_accs)
    print("rectified upper accs")
    print(mean_rectified_lower_accs)
    print("rectified lower accs")
    print(mean_rectified_upper_accs)

    plot_mean_losses(mean_upper_losses, mean_lower_losses,
                    mean_rectified_upper_losses, mean_rectified_lower_losses,
                    save_path = f"./savefig/mean_losses/{selection_method}")

    plot_mean_accs(mean_upper_accs, mean_lower_accs,
                mean_rectified_upper_accs, mean_rectified_lower_accs,
                save_path = f"./savefig/mean_accs/{selection_method}")
    return
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

        if cfg.DATASET.ACC_WITH_SIZE:
            plot_size_acc(imgsize_list_x, imgsize_list_y, valid_result.vector_instance_acc)

        print(f'Evaluation on test set, \n',
              'attr_mean_acc: {:.4f}, ma: {:.4f},  map: {:.4f}, label_f1: {:4f}, pos_recall: {:.4f} , neg_recall: {:.4f} \n'.format(
                  np.mean(valid_result.attr_acc), valid_result.ma, valid_map, np.mean(valid_result.label_f1),
                  np.mean(valid_result.label_pos_recall),
                  np.mean(valid_result.label_neg_recall)),
              'Acc: {:.4f}, Prec: {:.4f}, Rec: {:.4f}, F1: {:.4f}'.format(
                  valid_result.instance_acc, valid_result.instance_prec, valid_result.instance_recall,
                  valid_result.instance_f1)
              )
        save_hists(valid_result.label_pos_recall, valid_result.label_neg_recall, 
                   valid_result.attr_acc, attrs)

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
