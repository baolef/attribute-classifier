# Created by Baole Fang at 2/21/23
import argparse
import os

import numpy as np
from tqdm import tqdm
import torch
import models
from data.base import create_dataloader, create_folder
import yaml


def test(model, dataloader, acc=False):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Test')
    test_results = []
    num_correct = 0
    for i, (images, labels) in enumerate(dataloader):
        # TODO: Finish predicting on the test set.
        images = images.cuda()

        with torch.inference_mode():
            outputs = model(images)
        if acc:
            labels = labels.cuda()
            num_correct += int(((outputs > 0) == labels).sum())

        outputs = model.activation(outputs)
        outputs = outputs.detach().cpu().numpy().tolist()
        test_results.extend(outputs)

        batch_bar.update()
    if acc:
        acc = 100 * num_correct / (len(test_results[-1]) * len(dataloader.dataset))
        print(f'Test accuracy: {acc}')
    batch_bar.close()
    return test_results


def test_and_submit(model_path, dataloader, data_path=''):
    sv_file = torch.load(model_path)
    model = models.make(sv_file['model'], load_sd=True).cuda()
    test_results = test(model, dataloader, not data_path)
    if data_path:
        np.save(os.path.join(data_path, 's.npy'), test_results)
    else:
        np.save(os.path.join(os.path.dirname(model_path), 's.npy'), test_results)
    name = os.path.join(os.path.dirname(model_path), 'classification.csv')
    with open(name, "w+") as f:
        f.write("id,scores\n")
        for i in range(len(test_results)):
            f.write("{},{}\n".format(str(i).zfill(5) + ".jpg", test_results[i]))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--model')
    parser.add_argument('--data')
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')
    if args.data:
        test_loader = create_folder(args.data, config.get('dataset').get('batch_size'))
    else:
        _, _, test_loader = create_dataloader(**config.get('dataset'))
    test_and_submit(args.model, test_loader, args.data)
