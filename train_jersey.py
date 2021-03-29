import argparse
import os
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional
import torch.optim as optim
import torch.utils.data
from torch.optim.lr_scheduler import StepLR
from torchvision import transforms

from RandAugment import RandAugment
from dataset import Dataset, JerseyDataset
from evaluator import JerseyEvaluator, JerseyDetEvaluator
from model import JerseyModel, MobileJerseyModel, EfficientJerseyModel, ResNetJersey

parser = argparse.ArgumentParser()
parser.add_argument('-tf', '--train_file', help='..')
parser.add_argument('-vf', '--val_file', default=None, help='..')
parser.add_argument('-m', '--model', default='basic', help='..')
parser.add_argument('-e', '--exp', default='basic_model', help='directory to write logs')
parser.add_argument('-r', '--restore_checkpoint', default=None,
                    help='path to restore checkpoint, e.g. ./logs/model-100.pth')
parser.add_argument('-bs', '--batch_size', default=32, type=int,  help='Default 32')
parser.add_argument('-lr', '--learning_rate', default=1e-2, type=float, help='Default 1e-2')
parser.add_argument('-p', '--patience', default=100, type=int, help='Default 100, set -1 to train infinitely')
parser.add_argument('-ds', '--decay_steps', default=10000, type=int, help='Default 10000')
parser.add_argument('-dr', '--decay_rate', default=0.9, type=float, help='Default 0.9')


def _loss(length_logits, digit1_logits, digit2_logits, length_labels, digits_labels):
    length_cross_entropy = torch.nn.functional.cross_entropy(length_logits, length_labels)
    digit1_cross_entropy = torch.nn.functional.cross_entropy(digit1_logits, digits_labels[0])
    digit2_cross_entropy = torch.nn.functional.cross_entropy(digit2_logits, digits_labels[1])
    loss = length_cross_entropy + digit1_cross_entropy + digit2_cross_entropy
    return loss


def _train(path_to_train_file, path_to_val_file, path_to_log_dir,
           path_to_restore_checkpoint_file, training_options):
    model_name = training_options['model']
    batch_size = training_options['batch_size']
    initial_learning_rate = training_options['learning_rate']
    initial_patience = training_options['patience']
    num_steps_to_show_loss = 100
    num_steps_to_check = 1000

    step = 0
    patience = initial_patience
    best_accuracy = 0.0
    duration = 0.0

    if 'basic' in model_name:
        print("Loading Basic model")
        model = JerseyModel(inter_size=7)
        model.cuda()
    elif 'mobile' in model_name:
        print("Loading MobileV2 model")
        model = MobileJerseyModel()
        model.cuda()
    elif 'eff' in model_name:
        print("Loading Efficient model")
        from efficientnet_pytorch import get_model_params
        blocks_args, global_params = get_model_params('efficientnet-b0', {'image_size': 56})
        model = EfficientJerseyModel(blocks_args, global_params)
        model.cuda()
    elif 'r50' in model_name:
        print("Loading ResNet50 model...")
        from torchvision.models.resnet import Bottleneck
        model = ResNetJersey(Bottleneck, [3, 4, 6, 3])
        model.cuda()
    elif 'r18' in model_name:
        print("Loading ResNet18 model...")
        from torchvision.models.resnet import BasicBlock
        model = ResNetJersey(BasicBlock, [2, 2, 2, 2])
        model.cuda()
    else:
        print("Wrong model name!")
        exit()


    size = 64
    crop_size = 54
    transform = transforms.Compose([
        RandAugment(2, 9),
        transforms.RandomCrop([crop_size, crop_size]),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    train_loader = torch.utils.data.DataLoader(JerseyDataset(path_to_train_file, transform, size),
                                               batch_size=batch_size, shuffle=True,
                                               num_workers=4, pin_memory=True)

    # The following evaluator test the accuracy of the whole pipeline
    # including the detection model
    evaluator = JerseyDetEvaluator()

    optimizer = optim.SGD(model.parameters(), lr=initial_learning_rate, momentum=0.9, weight_decay=0.0005)
    scheduler = StepLR(optimizer, step_size=training_options['decay_steps'], gamma=training_options['decay_rate'])

    if path_to_restore_checkpoint_file is not None:
        assert os.path.isfile(path_to_restore_checkpoint_file), '%s not found' % path_to_restore_checkpoint_file
        step = model.restore(path_to_restore_checkpoint_file)
        scheduler.last_epoch = step
        print('Model restored from file: %s' % path_to_restore_checkpoint_file)

    path_to_losses_npy_file = os.path.join(path_to_log_dir, 'losses.npy')
    if os.path.isfile(path_to_losses_npy_file):
        losses = np.load(path_to_losses_npy_file)
    else:
        losses = np.empty([0], dtype=np.float32)

    print('Number of training samples is , ', len(train_loader.dataset))
    while True:
        for batch_idx, (images, length_labels, digits_labels) in enumerate(train_loader):
            start_time = time.time()
            images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
            length_logits, digit1_logits, digit2_logits = model.train()(images)
            loss = _loss(length_logits, digit1_logits, digit2_logits, length_labels, digits_labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            step += 1
            duration += time.time() - start_time

            if step % num_steps_to_show_loss == 0:
                examples_per_sec = batch_size * num_steps_to_show_loss / duration
                duration = 0.0
                print('=> %s: step %d, loss = %f, learning_rate = %f (%.1f examples/sec)' % (
                    datetime.now(), step, loss.item(), scheduler.get_lr()[0], examples_per_sec))

            if step % num_steps_to_check != 0:
                continue

            losses = np.append(losses, loss.item())
            np.save(path_to_losses_npy_file, losses)

            print('=> Evaluating on validation dataset...')
            accuracy = evaluator.evaluate(model)
            print('==> accuracy = %f, best accuracy %f' % (accuracy, best_accuracy))

            if accuracy > best_accuracy:
                path_to_checkpoint_file = model.store(path_to_log_dir, step=step)
                print('=> Model saved to file: %s' % path_to_checkpoint_file)
                patience = initial_patience
                best_accuracy = accuracy
            else:
                patience -= 1

            print('=> patience = %d' % patience)
            if patience == 0:
                return


def main(args):
    path_to_train_file = args.train_file
    path_to_val_file = args.val_file
    path_to_log_dir = args.exp
    path_to_restore_checkpoint_file = args.restore_checkpoint
    training_options = {
        'batch_size': args.batch_size,
        'learning_rate': args.learning_rate,
        'patience': args.patience,
        'decay_steps': args.decay_steps,
        'decay_rate': args.decay_rate,
        'model': args.model
    }

    path_to_log_dir = os.path.join('./work_dirs', path_to_log_dir)
    if not os.path.exists(path_to_log_dir):
        os.makedirs(path_to_log_dir)

    print('Start training')
    _train(path_to_train_file, path_to_val_file, path_to_log_dir,
           path_to_restore_checkpoint_file, training_options)
    print('Done')


if __name__ == '__main__':
    main(parser.parse_args())
