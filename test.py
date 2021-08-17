import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms
from evaluator import JerseyEvaluator, JerseyDetEvaluator
from model import JerseyModel, ResNetJersey
import glob
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('-a', '--best_acc', default=89.7, help='..')
parser.add_argument('-m', '--models_path', default='./work_dirs/r18_randaug_fulldata/',  help='..')


def main():
    args = parser.parse_args()
    models_path = args.models_path

    model = JerseyModel(inter_size=7)
    model.restore('./work_dirs/basic_randaug_fulldata/model-best.pth')
    model.cuda()


    ### This checkpoint gives 92.6 in 1.17 minutes time
    # evaluator = JerseyDetEvaluator(
    #     config_file='/home/ubuntu/oljike/BallTracking/mmdetection/configs/yolo_jersey/yolov3_d53_320_273e_jersey.py',
    #     checkpoint_file='/home/ubuntu/oljike/BallTracking/mmdetection/work_dirs/jersey_region_yolov3-320_fullData/epoch_150.pth'
    # )

    # This checkpoint gives 92.3 in 1.04 minutes time.
    # evaluator = JerseyDetEvaluator(
    #     config_file='/home/ubuntu/oljike/BallTracking/mmdetection/configs/yolo_jersey/yolov3_d53_320_273e_jersey_smallres.py',
    #     checkpoint_file='/home/ubuntu/oljike/BallTracking/mmdetection/work_dirs/jersey_region_yolov3-320_fullData_smallRes/epoch_90.pth'
    # )

    evaluator = JerseyEvaluator('./data/jersey_crops_val.txt')

    best_accuracy = args.best_acc
    accuracy = evaluator.evaluate(model)
    print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))


if __name__ == '__main__':
    main()