import os
from torchvision import transforms
import os.path
from torch.utils.data import Dataset
import torch
from PIL import Image
import numpy as np
import random
from tqdm import tqdm


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)

def read_pts(file_path):
    points = []
    with open(file_path, 'r') as file:
        for line in file:
            points.append([float(value) for value in line.split()])
    return np.array(points)

class ShapeNetMPCDataloader(Dataset):
    def __init__(self, filepath, data_path, status, pc_input_num=2048, view_align=False, category='all'):
        super(ShapeNetMPCDataloader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.filelist = []
        self.cat = []
        self.key = []
        self.category = category
        self.view_align = view_align
        self.cat_map = {
            'plane':'02691156',
            'trash_bin':'02747177',
            'bag':'02773838',
            'basket':'02801938',
            'bathtub':'02808440',
            'bed':'02818832',
            'bench':'02828884',
            'birdhouse':'02843684',
            'bookshelf':'02871439',
            'bottle':'02876657',
            'bowl':'02880940',
            'bus':'02924116',
            'cabinet':'02933112',
            'camera':'02942699',
            'can':'02946921',
            'cap':'02954340',
            'car':'02958343',
            'cellphone':'02992529',
            'chair':'03001627',
            'clock':'03046257',
            'keyboard':'03085013',
            'dishwasher':'03207941',
            'display':'03211117',
            'earphone':'03261776',
            'faucet':'03325088',
            'file cabinet':'03337140',
            'guitar':'03467517',
            'helmet':'03513137',
            'jar':'03593526',
            'knife':'03624134',
            'lamp':'03636649',
            'laptop':'03642806',
            'loudspeaker':'03691459',
            'mailbox':'03710193',
            'microphone':'03759954',
            'microwaves':'03761084',
            'motorbike':'03790512',
            'mug':'03797390',
            'piano':'03928116',
            'pillow':'03938244',
            'pistol':'03948459',
            'flowerpot':'03991062',
            'printer':'04004475',
            'remote':'04074963',
            'rifle':'04090263',
            'rocket':'04099429',
            'skateboard':'04225987',
            'sofa':'04256520',
            'stove':'04330267',
            'table':'04379243',
            'telephone':'04401088',
            'tower':'04460130',
            'train':'04468005',
            'watercraft':'04530566',
            'washer':'04554684'
        }
        with open(filepath,'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.path = data_path
        
        for key in self.filelist:
            if category !='all':
                if key.split('/')[0]!= self.cat_map[category]:
                    continue
            self.cat.append(key.split(';')[0])
            self.key.append(key)

        self.transform = transforms.Compose([
            transforms.Resize(224),
            transforms.ToTensor()
        ])

        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):
        
        key = self.key[idx]
        pc_part_path = os.path.join(self.path,key.split('/')[0]+'/pc/'+ key.split('/')[1]+'-perview-'+key.split('/')[-1].replace('\n', '')+'.pts')
        # view_align = True, means the view of image equal to the view of partial points
        # view_align = False, means the view of image is different from the view of partial points
        if self.view_align:
            ran_key = key        
        else:
            ran_key = key[:-2]+str(random.randint(0,31))
       
        pc_path = os.path.join(self.path,key.split('/')[0]+'/pc/'+ key.split('/')[1]+'-complete'+'.pts')
        view_path = os.path.join(self.path,key.split('/')[0]+'/color/'+ key.split('/')[1]+'-color-'+key.split('/')[-1].replace('\n', '')+'.jpg')
        

        views = self.transform(Image.open(view_path))
        views = views[:3,:,:]
        # load gt
        pc = read_pts(pc_path).astype(np.float32)[:2048,:3]   #### 选择前2048个点和3个坐标
        # load partial points
        pc_part = read_pts(pc_part_path).astype(np.float32)[:2048,:3]


        # normalize partial point cloud and GT to the same scale
        gt_mean = pc.mean(axis=0) 
        pc = pc -gt_mean
        pc_L_max = np.max(np.sqrt(np.sum(abs(pc ** 2), axis=-1)))
        pc = pc/pc_L_max

        pc_part = pc_part-gt_mean
        pc_part = pc_part/pc_L_max

        return views.float(), torch.from_numpy(pc).float(), torch.from_numpy(pc_part).float(), key.split('/')[0]

    def __len__(self):
        return len(self.key)


if __name__ == "__main__":
    from torch.utils.data import DataLoader
    category = "table"
    MPCDataset = ShapeNetMPCDataloader(r'TrainTest_List/ShapeNetViPC/train_list.txt',data_path='/media/fenglang/Elements/datasets/PCshibie/ShapeNetCoreV2_55_MPC',status='test', category = category)
    train_loader = DataLoader(MPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    


    for image, gt, partial in tqdm(train_loader):
        
        print(image.shape)
        
        pass
    