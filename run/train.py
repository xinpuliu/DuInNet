import logging
import os
import torch
from torch.utils.data import DataLoader
## 引用模型
from models.EGIInet import EGIInet

#import utils.helpers
import argparse
from datetime import datetime
from tqdm import tqdm
from tensorboardX import SummaryWriter
from run.test import test_net
from utils.ShapeNetViPCDataloader import ShapeNetViPCDataloader
from utils.ShapeNetMPCDataloader import ShapeNetMPCDataloader
from utils.ModelNetMPCDataloader import ModelNetMPCDataloader
from utils.average_meter import AverageMeter
from utils.loss_utils import *
from utils.schedular import GradualWarmupScheduler
from torch.optim.lr_scheduler import *


class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()


def train_net(cfg):
    torch.backends.cudnn.benchmark = True
    ViewAlign = cfg.DATASETS.view_align
    if cfg.DATASETS.NAME == 'ShapeNetViPC':
        Dataset_train = ShapeNetViPCDataloader(r'TrainTest_List/ShapeNetViPC/train_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetViPC, status='train',
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)

        Dataset_test = ShapeNetViPCDataloader(r'TrainTest_List/ShapeNetViPC/test_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetViPC, status='test',
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)
        workers = cfg.CONST.NUM_WORKERS
        perfetch = cfg.CONST.DATA_perfetch

    if cfg.DATASETS.NAME == 'ModelNetMPC':
        if cfg.DATASETS.zero_shot == False:
            train_list_path = r'TrainTest_List/ModelNetMPC/train_list.txt'
            test_list_path = r'TrainTest_List/ModelNetMPC/test_list.txt'
        else:
            train_list_path = r'TrainTest_List/ModelNetMPC/zero_shot/train_list.txt'
            if cfg.DATASETS.seen == True:
                test_list_path = r'TrainTest_List/ModelNetMPC/zero_shot/seen_test_list.txt'
            else:
                test_list_path = r'TrainTest_List/ModelNetMPC/zero_shot/unseen_test_list.txt'

        Dataset_train = ModelNetMPCDataloader(train_list_path,
                                    data_path=cfg.DATASETS.PATH_ModelNetMPC, status='train', noise_partial=cfg.DATASETS.denoise,
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)

        Dataset_test = ModelNetMPCDataloader(test_list_path,
                                    data_path=cfg.DATASETS.PATH_ModelNetMPC, status='test', noise_partial=cfg.DATASETS.denoise,
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)
        workers = cfg.CONST.NUM_WORKERS
        perfetch = cfg.CONST.DATA_perfetch
        
    if cfg.DATASETS.NAME == 'ShapeNetMPC':
        Dataset_train = ShapeNetMPCDataloader(r'TrainTest_List/ShapeNetMPC/train_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetMPC, status='train',
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)

        Dataset_test = ShapeNetMPCDataloader(r'TrainTest_List/ShapeNetMPC/test_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetMPC, status='test',
                                    view_align=ViewAlign, category=cfg.TRAIN.CATE)
        workers = 2 * cfg.CONST.NUM_WORKERS
        perfetch = 2 * cfg.CONST.DATA_perfetch
    
    train_data_loader = DataLoader(Dataset_train,
                            batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=workers,
                            shuffle=True,
                            drop_last=True,
                            prefetch_factor=perfetch)
    val_data_loader = DataLoader(Dataset_test,
                            batch_size=cfg.TRAIN.BATCH_SIZE,
                            num_workers=workers,
                            shuffle=True,
                            drop_last=True,
                            prefetch_factor=perfetch)

    


    # Set up folders for logs and checkpoints
    output_dir = os.path.join(cfg.DIR.OUT_PATH, cfg.NETWORK.NAME, cfg.TRAIN.CATE, '%s', 'BS'+str(cfg.TRAIN.BATCH_SIZE)+'_'+datetime.now().strftime('%y-%m-%d-%H-%M-%S'))
    cfg.DIR.CHECKPOINTS = output_dir % 'checkpoints'
    cfg.DIR.LOGS = output_dir % 'logs'
    if not os.path.exists(cfg.DIR.CHECKPOINTS):
        os.makedirs(cfg.DIR.CHECKPOINTS)
    if not os.path.exists(cfg.DIR.LOGS):
        os.makedirs(cfg.DIR.LOGS)
    
    log_file_path = cfg.DIR.LOGS + '/traintest.log'
    # if not os.path.exists(log_file_path):
    #     open(log_file_path, 'w').close()

    io = IOStream(log_file_path)

    # Create tensorboard writers
    train_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'train'))
    val_writer = SummaryWriter(os.path.join(cfg.DIR.LOGS, 'test'))

    
    model = EGIInet()#.apply(weights_init_normal)
    model.cuda()
    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model, device_ids=cfg.CONST.GPUs, output_device=cfg.CONST.GPUs[0])
    
    # Create the optimizers
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                                 lr=cfg.TRAIN.LEARNING_RATE,
                                 weight_decay=cfg.TRAIN.WEIGHT_DECAY,
                                 betas=cfg.TRAIN.BETAS)

    # lr scheduler
    scheduler_steplr = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
    lr_scheduler = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=cfg.TRAIN.WARMUP_STEPS,
                                          after_scheduler=scheduler_steplr)
    init_epoch = 0
    best_metrics = float('inf')
    steps = 0
    BestEpoch = 0

    if 'WEIGHTS' in cfg.CONST:
        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        steps = cfg.TRAIN.WARMUP_STEPS+1
        lr_scheduler = MultiStepLR(optimizer,milestones=cfg.TRAIN.LR_DECAY_STEP, gamma=cfg.TRAIN.GAMMA)
        optimizer.param_groups[0]['lr']= cfg.TRAIN.LEARNING_RATE
        logging.info('Recover complete.')

    print('recon_points: ',cfg.DATASETS.N_POINTS, 'Parameters: ', sum(p.numel() for p in model.parameters()))
    #exit()
    # Training/Testing the network
    for epoch_idx in range(init_epoch + 1, cfg.TRAIN.N_EPOCHS + 1):

        model.train()

        total_cd_pc = 0
        total_style = 0
        total_loss = 0

        n_batches = len(train_data_loader)
        print('epoch: ', epoch_idx, 'optimizer: ', optimizer.param_groups[0]['lr'])
        io.cprint("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! NEW EPOCH !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        with tqdm(train_data_loader) as t:
            for batch_idx, (view,gt_pc,part_pc,category,) in enumerate(t):

                partial = part_pc.cuda()#[16,2048,3]
                gt = gt_pc.cuda()#[16,2048,3]
                png = view.cuda()
                partial = farthest_point_sample(partial,cfg.DATASETS.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.N_POINTS)
                      
                recon,style_loss=model(partial,png)
                cd = chamfer_sqrt(recon,gt)
                loss_total=cd+style_loss*1e-2
                optimizer.zero_grad()
                loss_total.mean().backward()
                optimizer.step()

                cd_pc_item = cd.mean().item() * 1e3
                total_cd_pc += cd_pc_item
                style_item = style_loss.mean().item() * 1e1
                total_style += style_item
                loss_item = loss_total.mean().item() * 1e3
                total_loss += loss_item
                n_itr = (epoch_idx - 1) * n_batches + batch_idx
                train_writer.add_scalar('Loss/Batch/cd_pc', cd_pc_item, n_itr)
                train_writer.add_scalar('Loss/Batch/style', style_item, n_itr)
                train_writer.add_scalar('Loss/Batch/loss', loss_item, n_itr)
                t.set_description(
                    '[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                t.set_postfix(loss='%s' % ['%.4f' % l for l in [cd_pc_item, style_item, loss_item]])
                if batch_idx % 50 == 0:
                    io.cprint('[Epoch %d/%d][Batch %d/%d]' % (epoch_idx, cfg.TRAIN.N_EPOCHS, batch_idx + 1, n_batches))
                    io.cprint('Losses = %s' % ['%.4f' % l for l in [cd_pc_item, style_item, loss_item]])
                if steps <= cfg.TRAIN.WARMUP_STEPS:
                    lr_scheduler.step()
                    steps += 1

        avg_cdc = total_cd_pc / n_batches
        avg_style = total_style / n_batches
        avg_loss = total_loss / n_batches

        lr_scheduler.step()
        train_writer.add_scalar('Loss/Epoch/cd_pc', avg_cdc, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/style', avg_style, epoch_idx)
        train_writer.add_scalar('Loss/Epoch/loss', avg_loss, epoch_idx)
        logging.info(
                '[Epoch %d/%d]  Losses = %s' %
                (epoch_idx, cfg.TRAIN.N_EPOCHS,
                 ['%.4f' % l for l in [avg_cdc, avg_style, avg_loss]]))
        io.cprint('[Epoch %d/%d]  Losses = %s' % (epoch_idx, cfg.TRAIN.N_EPOCHS, ['%.4f' % l for l in [avg_cdc, avg_style, avg_loss]]))
        io.cprint("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!! TEST !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

        # Validate the current model
        cd_eval = test_net(cfg, epoch_idx, val_data_loader, val_writer, model, io)
        # Save checkpoints
        if epoch_idx % cfg.TRAIN.SAVE_FREQ == 0 or cd_eval < best_metrics:
            if cd_eval < best_metrics:
                best_metrics = cd_eval
                BestEpoch = epoch_idx
                file_name = 'ckpt-best.pth'
                io.cprint("!! BEST !!")

            else:
                file_name = 'ckpt-epoch-%03d.pth' % epoch_idx
            output_path = os.path.join(cfg.DIR.CHECKPOINTS, file_name)
            torch.save({
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict()
            }, output_path)

            logging.info('Saved checkpoint to %s ...' % output_path)
        logging.info('Best Performance: Epoch %d -- CD %.4f' % (BestEpoch,best_metrics))

    train_writer.close()
    val_writer.close()
