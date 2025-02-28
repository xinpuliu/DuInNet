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


def rotation_z(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, -sin_theta, 0.0],
                                [sin_theta, cos_theta, 0.0],
                                [0.0, 0.0, 1.0]])
    return pts @ rotation_matrix.T


def rotation_y(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[cos_theta, 0.0, sin_theta],
                                [0.0, 1.0, 0.0],
                                [-sin_theta, 0.0, cos_theta]])
    return pts @ rotation_matrix.T


def rotation_x(pts, theta):
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    rotation_matrix = np.array([[1.0, 0.0, 0.0],
                                [0.0, cos_theta, -sin_theta],
                                [0.0, sin_theta, cos_theta]])
    return pts @ rotation_matrix.T


class ModelNetMPCDataloader(Dataset):
    def __init__(self, filepath, data_path, status, pc_input_num=2048, noise_partial=False, category='all', view_align=True):
        super(ModelNetMPCDataloader,self).__init__()
        self.pc_input_num = pc_input_num
        self.status = status
        self.view_align = view_align
        self.noise = noise_partial
        self.filelist = []
        # self.cat = []
        self.key = []
        self.category = category
        self.cat_map = ['airplane', 'bathtub', 'bed', 'bench', 'bookshelf', 'bottle', 'bowl', 'car',
                        'chair', 'cone', 'cup', 'curtain', 'desk', 'door', 'dresser', 'flower_pot', 
                        'glass_box', 'guitar', 'keyboard', 'lamp', 'laptop', 'mantel', 'monitor', 
                        'night_stand', 'person', 'piano', 'plant', 'radio', 'range_hood', 'sink', 
                        'sofa', 'stairs', 'stool', 'table', 'tent', 'toilet', 'tv_stand','vase','wardrobe','xbox'
                       ]
        self.angle_view = np.array([[0.0, 1.5707963267948966], 
                           [0.0, -1.5707963267948966], 
                           [0.0, 0.0], 
                           [0.0, 0.7853981633974483], 
                           [0.0, -0.7853981633974483], 
                           [0.6283185307179586, 0.0], 
                           [0.6283185307179586, 0.7853981633974483], 
                           [0.6283185307179586, -0.7853981633974483], 
                           [1.2566370614359172, 0.0], 
                           [1.2566370614359172, 0.7853981633974483], 
                           [1.2566370614359172, -0.7853981633974483], 
                           [1.8849555921538759, 0.0], 
                           [1.8849555921538759, -0.7853981633974483], 
                           [1.8849555921538759, 0.7853981633974483], 
                           [2.5132741228718345, 0.0],
                           [2.5132741228718345, -0.7853981633974483],
                           [2.5132741228718345, 0.7853981633974483],
                           [3.141592653589793, 0.0],
                           [3.141592653589793, -0.7853981633974483],
                           [3.141592653589793, 0.7853981633974483],
                           [3.7699111843077517, 0.0],
                           [3.7699111843077517, -0.7853981633974483],
                           [3.7699111843077517, 0.7853981633974483],
                           [4.39822971502571, 0.0],
                           [4.39822971502571, -0.7853981633974483],
                           [4.39822971502571, 0.7853981633974483],
                           [5.026548245743669, 0.0],
                           [5.026548245743669, 0.7853981633974483],
                           [5.026548245743669, -0.7853981633974483],
                           [5.654866776461628, 0.0],
                           [5.654866776461628, 0.7853981633974483],
                           [5.654866776461628, -0.7853981633974483]
                           ])
        with open(filepath, 'r') as f:
            line = f.readline()
            while (line):
                self.filelist.append(line)
                line = f.readline()
        
        self.imcomplete_path = os.path.join(data_path,'Partial')
        self.noise_partial = os.path.join(data_path,  'Noise_Partial')
        self.gt_path = os.path.join(data_path,'GT')               
        self.view_path = os.path.join(data_path,'Img')      

        for key in self.filelist:
            if category !='all':
                if not key.split('/')[0] == self.category:
                    continue

            self.key.append(key)

        self.transform = transforms.Compose([   
                            # transforms.Resize([224, 224]),     
                            transforms.ToTensor()      
                            ])


        print(f'{status} data num: {len(self.key)}')


    def __getitem__(self, idx):

        key = self.key[idx].strip('\n')                    
        
        if self.noise:
            pc_part_path = os.path.join(self.noise_partial, key)
        else:  
            pc_part_path = os.path.join(self.imcomplete_path, key)

        
        if self.view_align:  # view_align = False
            ran_key = key[:-4]        
        else:
            ran_key = key[:-7] + str(random.randint(1, 32)).rjust(3, '0')    
        view_path = self.view_path + '/' + ran_key + '.png'         

        gt_key = key[:-8]  
        if self.status == 'train':
            pc_path = self.gt_path + '/' + gt_key + '.xyz'          # use gt_points of training
        else:
            pc_path = self.gt_path + '/' + gt_key + '_fps.xyz'      # use gt_points of testing

        ### load data ###
        views = self.transform(Image.open(view_path)) 
        views[views >= 1] = 0
        with open(pc_path,'rb') as f:                       
            pc = np.loadtxt(f).astype(np.float32)

        with open(pc_part_path,'rb') as f:
            pc_part = np.loadtxt(f).astype(np.float32)
        # incase some item point number less than 3500
        if pc_part.shape[0]<self.pc_input_num:
            pc_part = np.repeat(pc_part,(self.pc_input_num//pc_part.shape[0])+1,axis=0)[0:self.pc_input_num]
        
        # load the view metadata
        x_rotation_angle1 = -1.5707963267948966
        x_rotation_matrix1 = np.array([[1, 0, 0],
                            [0, np.cos(x_rotation_angle1), -np.sin(x_rotation_angle1)],
                            [0, np.sin(x_rotation_angle1), np.cos(x_rotation_angle1)]])
        pc_part = np.dot(pc_part, x_rotation_matrix1.T)
        pc = np.dot(pc, x_rotation_matrix1.T)

        y_rotation_angle1 = -1.5707963267948966
        y_rotation_matrix1 = np.array([[np.cos(y_rotation_angle1), 0, np.sin(y_rotation_angle1)],
                                    [0, 1, 0],
                                    [-np.sin(y_rotation_angle1), 0, np.cos(y_rotation_angle1)]])
        pc_part = np.dot(pc_part, y_rotation_matrix1.T)
        pc = np.dot(pc, y_rotation_matrix1.T)

        image_view_id = view_path.split('.')[0].split('/')[-1].split('_')[-1]
        x_rotation_angle = self.angle_view[(int(image_view_id) - 1),1]  
        y_rotation_angle = - self.angle_view[(int(image_view_id) - 1),0] 
        
        x_rotation_matrix = np.array([[1, 0, 0],
                            [0, np.cos(x_rotation_angle), -np.sin(x_rotation_angle)],
                            [0, np.sin(x_rotation_angle), np.cos(x_rotation_angle)]])
        
        pc_part = np.dot(pc_part, x_rotation_matrix.T)     
        pc =  np.dot(pc, x_rotation_matrix.T) 

        y_rotation_matrix = np.array([[np.cos(y_rotation_angle), 0, np.sin(y_rotation_angle)],
                                    [0, 1, 0],
                                    [-np.sin(y_rotation_angle), 0, np.cos(y_rotation_angle)]])
        pc_part = np.dot(pc_part, y_rotation_matrix.T)
        pc =  np.dot(pc, y_rotation_matrix.T)

            
        # pc_part = rotation_y(rotation_z(pc_part, theta_img), phi_img) 
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
    category = "all"
    filepah = '/root/OWN_modelnet/dataset/ModelNet40ViPC_list/train_list.txt'
    data_path='/root/autodl-tmp/ModelNet40ViPC'
    ViPCDataset = ModelNetMPCDataloader(filepah, data_path, status='test', category = category)
    train_loader = DataLoader(ViPCDataset,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True,
                              drop_last=True)
    
    for image, gt, partial in tqdm(train_loader):
        
        print(image.shape)
        
        pass
    