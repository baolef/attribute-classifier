# Created by Baole Fang at 2/20/23

import argparse
import gc
import os

import yaml
import torch
from torchsummary import summary

import models
import utils
from torch.optim import lr_scheduler
from torch import nn
from tqdm import tqdm
from data.base import create_dataloader


def prepare_training():
    if config.get('restart') is not None:
        sv_file = torch.load(config['restart'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        epoch_start = sv_file['epoch'] + 1
        if config.get('scheduler') is None:
            scheduler = None
        else:
            scheduler_class = lr_scheduler.__dict__[config.get('scheduler')['name']]
            scheduler = scheduler_class(optimizer=optimizer, **config.get('scheduler')['args'])
        if config.get('optimizer').get('force_lr'):
            optimizer.param_groups[0]['lr'] = config.get('optimizer').get('force_lr')
        max_val_v = sv_file['accuracy']
    elif config.get('resume') is not None:
        sv_file = torch.load(config['resume'])
        model = models.make(sv_file['model'], load_sd=True).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), sv_file['optimizer'], load_sd=True)
        if config.get('optimizer').get('force_lr'):
            optimizer.param_groups[0]['lr'] = config.get('optimizer').get('force_lr')
        epoch_start = sv_file['epoch'] + 1
        if config.get('scheduler') is None:
            scheduler = None
        else:
            scheduler = utils.make_scheduler(optimizer, sv_file['scheduler', True])
            # scheduler_class=lr_scheduler.__dict__[config.get('scheduler')['name']]
            # scheduler = scheduler_class(optimizer=optimizer, **config.get('scheduler')['args'],last_epoch=epoch_start-1)
        # for _ in range(epoch_start - 1):
        #     scheduler.step()
        max_val_v = sv_file['accuracy']
    else:
        model = models.make(config['model']).cuda()
        optimizer = utils.make_optimizer(
            model.parameters(), config['optimizer'])
        epoch_start = 1
        if config.get('scheduler') is None:
            scheduler = None
        else:
            scheduler_class = lr_scheduler.__dict__[config.get('scheduler')['name']]
            scheduler = scheduler_class(optimizer=optimizer, **config.get('scheduler')['args'])
        if config['precision'] == 'half':
            model = model.half()
        max_val_v = -1e18

    log('model: #params={}'.format(utils.compute_num_params(model, text=True)))
    return model, optimizer, epoch_start, scheduler, max_val_v


def train(model, dataloader, optimizer, criterion=nn.BCELoss()):
    model.train()

    # Progress Bar
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, leave=False, position=0, desc='Train', ncols=5)

    num_correct = 0
    count = 0
    total_loss = 0

    for i, (images, labels) in enumerate(dataloader):
        optimizer.zero_grad()  # Zero gradients

        images, labels = images.cuda(), labels.cuda()

        with torch.cuda.amp.autocast():  # This implements mixed precision. Thats it!
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Update no. of correct predictions & loss as we iterate
        num_correct += int(((outputs > 0) == labels).sum())
        total_loss += float(loss.item())
        count += len(images)

        # tqdm lets you add some details so you can monitor training as you train.
        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (count * labels.shape[-1])),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct,
            lr="{:.04f}".format(float(optimizer.param_groups[0]['lr']))
        )

        loss.backward()
        optimizer.step()
        # scaler.scale(loss).backward()  # This is a replacement for loss.backward()
        # scaler.step(optimizer)  # This is a replacement for optimizer.step()
        # scaler.update()

        batch_bar.update()  # Update tqdm bar

    batch_bar.close()  # You need this to close the tqdm bar

    acc = 100 * num_correct / (labels.shape[-1] * len(dataloader.dataset))
    total_loss = float(total_loss / len(dataloader))

    return acc, total_loss


def validate(model, dataloader, criterion):
    model.eval()
    batch_bar = tqdm(total=len(dataloader), dynamic_ncols=True, position=0, leave=False, desc='Val', ncols=5)

    num_correct = 0
    total_loss = 0.0

    for i, (images, labels) in enumerate(dataloader):
        # Move images to device
        images, labels = images.cuda(), labels.cuda()

        # Get model outputs
        with torch.inference_mode():
            outputs = model(images)
            loss = criterion(outputs, labels)

        num_correct += int(((outputs > 0) == labels).sum())
        total_loss += float(loss.item())

        batch_bar.set_postfix(
            acc="{:.04f}%".format(100 * num_correct / (config['dataset']['batch_size'] * labels.shape[-1] * (i + 1))),
            loss="{:.04f}".format(float(total_loss / (i + 1))),
            num_correct=num_correct)

        batch_bar.update()

    batch_bar.close()
    acc = 100 * num_correct / (config['dataset']['batch_size'] * labels.shape[-1] * len(dataloader))
    total_loss = float(total_loss / len(dataloader))
    return acc, total_loss


def main(config_, save_path_):
    global config, log, scaler
    config = config_
    remove = config.get('resume') is None and config.get('restart') is None
    log = utils.set_save_path(save_path, remove)
    with open(os.path.join(save_path_, 'config.yaml'), 'w') as f:
        yaml.dump(config, f, sort_keys=False)

    train_loader, valid_loader, test_loader = create_dataloader(**config.get('dataset'))
    model, optimizer, epoch_start, scheduler, max_val_v = prepare_training()

    summary(model, (3, 224, 224))

    n_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(','))
    if n_gpus > 1:
        model = nn.parallel.DataParallel(model)

    epoch_max = config['epoch_max']
    epoch_save = config.get('epoch_save')

    criterion = nn.BCEWithLogitsLoss()

    timer = utils.Timer()
    scaler = torch.cuda.amp.GradScaler()

    gc.collect()  # These commands help you when you face CUDA OOM error
    torch.cuda.empty_cache()

    for epoch in range(epoch_start, epoch_max + 1):
        t_epoch_start = timer.t()
        log_info = ['epoch {}/{}'.format(epoch, epoch_max)]

        curr_lr = float(optimizer.param_groups[0]['lr'])
        train_acc, train_loss = train(model, train_loader, optimizer, criterion)
        if scheduler is not None:
            scheduler.step()

        log_info.append('train: acc={:.4f} loss={:.4f} lr={:.4f}'.format(train_acc, train_loss, curr_lr))

        val_acc, val_loss = validate(model, valid_loader, criterion)
        log_info.append("val: acc={:.04f} loss={:.04f}".format(val_acc, val_loss))

        if n_gpus > 1:
            model_ = model.module
        else:
            model_ = model
        model_spec = config['model']
        model_spec['sd'] = model_.state_dict()
        optimizer_spec = config['optimizer']
        optimizer_spec['sd'] = optimizer.state_dict()
        if scheduler:
            scheduler_spec = config['scheduler']
            scheduler_spec['sd'] = scheduler.state_dict()
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'scheduler': scheduler_spec,
                'epoch': epoch,
                'accuracy': max(max_val_v, val_acc)
            }
        else:
            sv_file = {
                'model': model_spec,
                'optimizer': optimizer_spec,
                'scheduler': None,
                'epoch': epoch,
                'accuracy': max(max_val_v, val_acc)
            }

        torch.save(sv_file, os.path.join(save_path_, 'epoch-last.pth'))

        if (epoch_save is not None) and (epoch % epoch_save == 0):
            torch.save(sv_file, os.path.join(save_path_, 'epoch-{}.pth'.format(epoch)))

        if n_gpus > 1 and (config.get('eval_bsize') is not None):
            model_ = model.module
        else:
            model_ = model

        if val_acc > max_val_v:
            max_val_v = val_acc
            torch.save(sv_file, os.path.join(save_path_, 'epoch-best.pth'))

        t = timer.t()
        prog = (epoch - epoch_start + 1) / (epoch_max - epoch_start + 1)
        t_epoch = utils.time_text(t - t_epoch_start)
        t_elapsed, t_all = utils.time_text(t), utils.time_text(t / prog)
        log_info.append('{} {}/{}\n'.format(t_epoch, t_elapsed, t_all))

        log(', '.join(log_info))

    log('max_memory_allocated=' + str(torch.cuda.max_memory_allocated()))
    log('max_memory_reserved=' + str(torch.cuda.max_memory_reserved()))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config')
    parser.add_argument('--name', default=None)
    parser.add_argument('--tag', default=None)
    parser.add_argument('--gpu', default='0')
    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    with open(args.config, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        print('config loaded.')

    save_name = args.name
    if save_name is None:
        save_name = args.config.split('/')[-1][:-len('.yaml')]
    if args.tag is not None:
        save_name += '_' + args.tag
    save_path = os.path.join('./save', save_name)
    main(config, save_path)
