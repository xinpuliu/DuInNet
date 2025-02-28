import logging
import torch
from torch.utils.data import DataLoader
## 引用模型
from models.EGIInet import EGIInet
import numpy as np
#import utils.data_loaders
#import utils.helpers
from tqdm import tqdm
from utils.ShapeNetViPCDataloader import ShapeNetViPCDataloader
from utils.ShapeNetMPCDataloader import ShapeNetMPCDataloader
from utils.ModelNetMPCDataloader import ModelNetMPCDataloader
from utils.average_meter import AverageMeter
from utils.loss_utils import *

def dictionary_loss(categ,dict_cd,dict_fi,loss,f1):
    
    i = 0
    for cat in categ:
        if cat in dict_cd:
            dict_cd[cat].append(loss[i].item())
            dict_fi[cat].append(f1[i].item())
        else:
            dict_cd[cat] = []
            dict_fi[cat] = []
            dict_cd[cat].append(loss[i].item())
            dict_fi[cat].append(f1[i].item())   
        dict_cd['all'].append(loss[i].item())
        dict_fi['all'].append(f1[i].item())
        
        i+=1 
     
    return dict_cd, dict_fi

class IOStream():
    def __init__(self, path):
        self.f = open(path, 'a')

    def cprint(self, text):
        # print(text)
        self.f.write(text+'\n')
        self.f.flush()

    def close(self):
        self.f.close()

def test_net(cfg, epoch_idx=-1, test_data_loader=None, test_writer=None, model=None, io=None):
    torch.backends.cudnn.benchmark = True
    ViewAlign = cfg.DATASETS.view_align
    if test_data_loader is None:
        # Set up data loader
        if cfg.DATASETS.NAME == 'ShapeNetViPC':
            Dataset_test = ShapeNetViPCDataloader(r'TrainTest_List/ShapeNetViPC/test_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetViPC,
                                    status='test',
                                    view_align=ViewAlign, category=cfg.TEST.CATE)
            workers = cfg.CONST.NUM_WORKERS
            perfetch = cfg.CONST.DATA_perfetch
        
        if cfg.DATASETS.NAME == 'ModelNetMPC':
            Dataset_test = ModelNetMPCDataloader(r'TrainTest_List/ModelNetMPC/test_list.txt',
                                        data_path=cfg.DATASETS.PATH_ModelNetMPC, status='test', noise_partial=cfg.DATASETS.denoise,
                                        view_align=ViewAlign, category=cfg.TRAIN.CATE)
            workers = cfg.CONST.NUM_WORKERS
            perfetch = cfg.CONST.DATA_perfetch

        if cfg.DATASETS.NAME == 'ShapeNetMPC':
            Dataset_test = ShapeNetMPCDataloader(r'TrainTest_List/ShapeNetMPC/test_list.txt',
                                    data_path=cfg.DATASETS.PATH_ShapeNetMPC,
                                    status='test',
                                    view_align=ViewAlign, category=cfg.TEST.CATE)
            workers = 2 * cfg.CONST.NUM_WORKERS
            perfetch = 2 * cfg.CONST.DATA_perfetch
            
        test_data_loader = DataLoader(Dataset_test,
                                    batch_size=cfg.TEST.BATCH_SIZE,
                                    num_workers=workers,
                                    shuffle=True,
                                    drop_last=True,
                                    prefetch_factor=perfetch)


    # Setup networks and initialize networks
    if model is None:
        model = EGIInet()#.apply(weights_init_normal)
        model.cuda()
        if torch.cuda.is_available():
            model = torch.nn.DataParallel(model, device_ids=cfg.CONST.GPUs, output_device=cfg.CONST.GPUs[0])

        logging.info('Recovering from %s ...' % (cfg.CONST.WEIGHTS))
        checkpoint = torch.load(cfg.CONST.WEIGHTS)
        model.load_state_dict(checkpoint['model'])

    # Switch models to evaluation mode
    model.eval()

    n_samples = len(test_data_loader)
    test_losses = AverageMeter(['CDl1', 'CDl2', 'F1'])
    test_metrics = AverageMeter(['CDl1', 'CDl2', 'F1'])
    category_metrics = dict()

    # Testing loop
    loss_dict = {}
    loss_dict['all'] = []
    f1_dict ={}
    f1_dict['all'] = []
    with tqdm(test_data_loader) as t:
        for model_idx, (view,gt_pc,part_pc,category) in enumerate(t):

            with torch.no_grad():

                partial = part_pc.cuda()  # [16,2048,3]
                gt = gt_pc.cuda()  # [16,2048,3]
                png = view.cuda()
                partial = farthest_point_sample(partial,cfg.DATASETS.N_POINTS)
                gt = farthest_point_sample(gt,cfg.DATASETS.N_POINTS)

                model.eval()
              
                pcds_pred,_ = model(partial, png)
                cdl1, cdl2, f1 = calc_cd(pcds_pred, gt, calc_f1=True)
                   
                loss_dict, f1_dict = dictionary_loss(categ=category,dict_cd=loss_dict,dict_fi=f1_dict,loss=cdl2* 1e3,f1=f1)

                
                cdl1 = cdl1.mean().item() * 1e3
                cdl2 = cdl2.mean().item() * 1e3
                f1 = f1.mean().item()

                _metrics = [cdl1, cdl2, f1]
                test_losses.update([cdl1, cdl2, f1])

                test_metrics.update(_metrics)

    # Print testing results
    print('============================ TEST RESULTS ============================')

    """
    print('Taxonomy', end='\t')
    print('#Sample', end='\t')
    for metric in test_metrics.items:
        print(metric, end='\t')
    print()

    for taxonomy_id in category_metrics:
        print(taxonomy_id, end='\t')
        print(category_metrics[taxonomy_id].count(0), end='\t')
        for value in category_metrics[taxonomy_id].avg():
            print('%.4f' % value, end='\t')
        print()

    """
    loss_dict = {key: np.mean(np.array(value)) for key, value in loss_dict.items()}
    f1_dict = {key: np.mean(np.array(value)) for key, value in f1_dict.items()}

    print("\n")  
    for key, value in loss_dict.items():
        print(f'Category: {key}'.ljust(30), f'CD is: {value}')    
    # print('all category loss is : ', loss_dict['all'])

    print('\n***************************************************************')    
    print('***************************************************************\n')    

    for key, value in f1_dict.items():
        print(f'Category: {key}'.ljust(30), f'F1_Score is: {value}')
    
    print('Overall', end='\t\t\t')
    for value in test_metrics.avg():
        print('%.4f' % value, end='\t')
        if io is not None:
            io.cprint('%.4f' % value)
    print('\n')

    print('Epoch ', epoch_idx, end='\t')
    for value in test_losses.avg():
        print('%.4f' % value, end='\t')
    print('\n')

    # Add testing results to TensorBoard
    if test_writer is not None:
        test_writer.add_scalar('Loss/Epoch/cd', test_losses.avg(0), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/dcd', test_losses.avg(1), epoch_idx)
        test_writer.add_scalar('Loss/Epoch/f1', test_losses.avg(2), epoch_idx)
        for i, metric in enumerate(test_metrics.items):
            test_writer.add_scalar('Metric/%s' % metric, test_metrics.avg(i), epoch_idx)

    return test_losses.avg(0)