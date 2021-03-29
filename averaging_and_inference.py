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


def average_models(models_path):
    ### here, we will load all the models and return the final one
    print("Loading Basic model")

    all_paths = glob.glob(os.path.join(models_path, '*.pth'))
    all_models = []
    for path in all_paths:
        if 'best' in path: continue

        if 'basic' in path:
            model = JerseyModel(inter_size=7)

            dst_model = JerseyModel(inter_size=7)
            dst_dict_params = dst_model.state_dict()
        elif 'r18' in path:
            from torchvision.models.resnet import BasicBlock
            model = ResNetJersey(BasicBlock, [2, 2, 2, 2])
            dst_model = ResNetJersey(BasicBlock, [2, 2, 2, 2])
            dst_dict_params = dst_model.state_dict()


        _ = model.restore(path)
        params = sorted(list(model.state_dict().items()))
        all_models.append(params)

    beta = 1 / len(all_models)
    for param_names in zip(*all_models):
        name = param_names[0][0]

        out_weight = torch.zeros_like(param_names[0][1]).float()
        if name in dst_dict_params:
            for param in param_names:
                out_weight += beta * param[1]
            dst_dict_params[name] = out_weight


    dst_model.load_state_dict(dst_dict_params)
    dst_model.cuda()

    return dst_model

def main():
    args = parser.parse_args()
    models_path = args.models_path

    model = average_models(models_path)
    evaluator = JerseyDetEvaluator()

    best_accuracy = args.best_acc
    accuracy = evaluator.evaluate(model)
    print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

    if accuracy > best_accuracy:
        # save_path = os.path.join(models_path, 'model-best.pth')
        path_to_checkpoint_file = model.store(models_path, 'best', 100)
        print('=> Model saved to file: %s' % path_to_checkpoint_file)


if __name__ == '__main__':
    main()