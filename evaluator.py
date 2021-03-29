import torch
import torch.utils.data
from torchvision import transforms
import os
import cv2
from tqdm import tqdm
import numpy as np
from dataset import Dataset, JerseyDataset
from PIL import Image

class Evaluator(object):
    def __init__(self, path_to_lmdb_dir):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(Dataset(path_to_lmdb_dir, transform), batch_size=128, shuffle=False)

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits, digit3_logits, digit4_logits, digit5_logits = model.eval()(images)

                length_prediction = length_logits.max(1)[1]
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]
                digit3_prediction = digit3_logits.max(1)[1]
                digit4_prediction = digit4_logits.max(1)[1]
                digit5_prediction = digit5_logits.max(1)[1]

                if needs_include_length:
                    num_correct += (length_prediction.eq(length_labels) &
                                    digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cpu().sum()
                else:
                    num_correct += (digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1]) &
                                    digit3_prediction.eq(digits_labels[2]) &
                                    digit4_prediction.eq(digits_labels[3]) &
                                    digit5_prediction.eq(digits_labels[4])).cpu().sum()

        accuracy = num_correct.item() / len(self._loader.dataset)
        return accuracy


class JerseyEvaluator(object):
    def __init__(self, path_to_anno_file, size=64):
        transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
        self._loader = torch.utils.data.DataLoader(JerseyDataset(path_to_anno_file, transform, size), batch_size=128, shuffle=False)

    def evaluate(self, model):
        num_correct = 0
        needs_include_length = False

        with torch.no_grad():
            for batch_idx, (images, length_labels, digits_labels) in enumerate(self._loader):
                images, length_labels, digits_labels = images.cuda(), length_labels.cuda(), [digit_labels.cuda() for digit_labels in digits_labels]
                length_logits, digit1_logits, digit2_logits = model.eval()(images)

                length_prediction = length_logits.max(1)[1]
                digit1_prediction = digit1_logits.max(1)[1]
                digit2_prediction = digit2_logits.max(1)[1]

                # if length_prediction.item() == 1:
                #     res = [digit1_prediction.item(), None]
                # elif length_prediction.item() == 2:
                #     res = [digit1_prediction.item(), digit2_prediction.item()]
                #
                # res = int(''.join([str(x) for x in res if x is not None]))
                # labels = int(''.join([str(x) for x in digits_labels if x!=10]))
                # if res == labels:
                #     num_correct += 1
                if needs_include_length:
                    num_correct += (length_prediction.eq(length_labels) &
                                    digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1])).cpu().sum()
                else:
                    num_correct += (digit1_prediction.eq(digits_labels[0]) &
                                    digit2_prediction.eq(digits_labels[1])).cpu().sum()

        print('Number of samples is ', len(self._loader.dataset))
        accuracy = num_correct.item() / len(self._loader.dataset)
        # accuracy = num_correct / len(self._loader.dataset)
        return accuracy

from mmdet.apis import init_detector, inference_detector, inference_batch_detector
class JerseyDetEvaluator(object):
    def __init__(self,):
        ### Config and model weights path
        det_paths = '/home/ubuntu/oljike/ocr_jersey/two_stage_pipeline/data/detector_val_path.txt'
        with open(det_paths) as f:
            self.det_paths = f.readlines()

        config_file = '/home/ubuntu/oljike/BallTracking/mmdetection/configs/yolo_jersey/yolov3_d53_320_273e_jersey.py'
        # checkpoint_file = '/home/ubuntu/oljike/BallTracking/mmdetection/work_dirs/jersey_region_yolov3-320/epoch_80.pth'
        checkpoint_file = '/home/ubuntu/oljike/BallTracking/mmdetection/work_dirs/jersey_region_yolov3-320_fullData/epoch_150.pth'

        # build the model from a config file and a checkpoint file
        self.det_model = init_detector(config_file, checkpoint_file, device='cuda:0')

        self.transform = transforms.Compose([
            transforms.CenterCrop([54, 54]),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    def evaluate(self, class_model):

        all_ = 0
        crct = 0
        offset = 2
        bs = 16

        chunks = (len(self.det_paths) - 1) // bs + 1
        for i in tqdm(range(chunks)):
            batch = self.det_paths[i * bs:(i + 1) * bs]

            inp_data = []
            img_paths = []
            for img_path in batch:
                img_paths.append(img_path.strip())
                img = cv2.imread(img_path.strip())
                inp_data.append(img.copy())

            all_results = inference_batch_detector(self.det_model, inp_data)
            for idx, result in enumerate(all_results):

                if len(result[0]) == 0:
                    all_ += 1
                    continue

                img = inp_data[idx]
                max_res = max(result[0], key=lambda x: x[-1]).astype(np.int)
                jersey_crop = img[max_res[1] - offset: max_res[3] + offset,
                              max_res[0] - offset: max_res[2] + offset, :]

                if 0 in jersey_crop.shape:
                    jersey_crop = img[max_res[1]: max_res[3],
                                  max_res[0]: max_res[2], :]
                if 0 in jersey_crop.shape:
                    all_ += 1
                    continue

                jersey_res = self.infer_jersey(class_model, jersey_crop)
                jersey_res = ''.join([str(x) for x in jersey_res if x != 10])

                label = os.path.basename(img_paths[idx]).split('_')[-1].replace('.jpg', '')

                if jersey_res == label:
                    crct += 1
                all_ += 1

        return 100 * (crct / all_)

    def infer_jersey(self, model, numpy_image):

        with torch.no_grad():
            numpy_image = cv2.resize(numpy_image, (64, 64))

            image = Image.fromarray(numpy_image)
            image = self.transform(image)
            images = image.unsqueeze(dim=0).cuda()

            length_logits, digit1_logits, digit2_logits = model.eval()(images)

            digit1_prediction = digit1_logits.max(1)[1]
            digit2_prediction = digit2_logits.max(1)[1]

        return [digit1_prediction.item(), digit2_prediction.item()]
