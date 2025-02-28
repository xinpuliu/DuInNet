from easydict import EasyDict as edict

__C                                              = edict()
cfg                                              = __C

#
# Dataset Config
#
__C.DATASETS                                     = edict()
__C.DATASETS.NAME                                = 'ShapeNetViPC'  # ShapeNetViPC or ModelNetMPC or ShapeNetMPC 
__C.DATASETS.N_POINTS                            = 2048
__C.DATASETS.PATH_ShapeNetViPC                   = '/root/autodl-tmp/ShapeNetViPC'
__C.DATASETS.PATH_ModelNetMPC                    = '/root/autodl-tmp/ModelNetMPC'
__C.DATASETS.PATH_ShapeNetMPC                    = '/root/autodl-tmp/ShapeNetMPC'
__C.DATASETS.view_align                          = False  # True or False

# if DATASETS.NAME == 'ModelNetMPC'
__C.DATASETS.denoise                             = False  # True or False
__C.DATASETS.zero_shot                           = False  # True or False

# if DATASETS.zero_shot == True
__C.DATASETS.seen                                = False  # True or False

#
# Constants
#
__C.CONST                                        = edict()
__C.CONST.NUM_WORKERS                            = 8
__C.CONST.DATA_perfetch                          = 8
#
# Directories
#
__C.DIR                                          = edict()
__C.DIR.OUT_PATH                                 = 'project_logs'#path to save checkpoints and logs
__C.CONST.DEVICE                                 = '0,1'
__C.CONST.GPUs                                   = [0,1]

#
# Network
#
__C.NETWORK                                      = edict()
__C.NETWORK.NAME                                 = 'EGIInet'

__C.NETWORK.EGIInet                              = edict()
__C.NETWORK.EGIInet.embed_dim                    = 192
__C.NETWORK.EGIInet.depth                        = 6
__C.NETWORK.EGIInet.img_patch_size               = 14
__C.NETWORK.EGIInet.pc_sample_rate               = 0.125
__C.NETWORK.EGIInet.pc_sample_scale              = 2
__C.NETWORK.EGIInet.fuse_layer_num               = 2
__C.NETWORK.shared_encoder                       = edict()
__C.NETWORK.shared_encoder.block_head            = 12
__C.NETWORK.shared_encoder.pc_h_hidden_dim       = 192
#
# Train
#
__C.TRAIN                                        = edict()
__C.TRAIN.BATCH_SIZE                             = 128  ### init:128
__C.TRAIN.N_EPOCHS                               = 160
__C.TRAIN.SAVE_FREQ                              = 40
__C.TRAIN.LEARNING_RATE                          = 0.001
__C.TRAIN.LR_MILESTONES                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.LR_DECAY_STEP                          = [16,32,48,64,80,96,112,128,144]
__C.TRAIN.WARMUP_STEPS                           = 1
__C.TRAIN.GAMMA                                  = 0.7
__C.TRAIN.BETAS                                  = (.9, .999)
__C.TRAIN.WEIGHT_DECAY                           = 0
__C.TRAIN.CATE                                   = 'plane'  # care about the seen CATE of ModelNetMPC
__C.TRAIN.d_size                                 = 1
#
# Test
#
__C.TEST                                         = edict()
__C.TEST.METRIC_NAME                             = 'ChamferDistance'
__C.TEST.CATE                                    = 'plane'  # care about the unseen CATE of ModelNetMPC
__C.TEST.BATCH_SIZE                              = 64
#__C.CONST.WEIGHTS = r"/project/EGIInet/checkpoints/plane-ckpt-best.pth" #path to pre-trained checkpoints


'''

ShapeNet-ViPC Samples:
    plane           :77442
    cabinet         :29989
    car             :67635
    chair           :129324
    lamp            :44277
    sofa            :60689
    table           :161211
    watercraft      :36969

ModelNetMPC Samples:
    airplane	    :726
    bathtub         :156
    bed             :615
    bench           :193
    bookshelf       :672
    bottle          :435
    bowl (un)       :84
    car             :297
    chair           :989
    cone            :187
    cup (un)        :99
    curtain (un)    :158
    desk            :286
    door            :129
    dresser         :286
    flower_pot      :169
    glass_box       :271
    guitar          :255
    keyboard (un)   :165
    lamp            :144
    laptop          :169
    mantel          :384
    monitor         :565
    night_stand     :286
    person          :108
    piano           :331
    plant           :340
    radio (un)      :124
    range_hood      :215
    sink (un)       :148
    sofa            :780
    stairs (un)     :144
    stool (un)      :110
    table           :492
    tent (un)       :183
    toilet          :444
    tv_stand        :367
    vase            :575
    wardrobe (un)   :107
    xbox            :123

ShapeNetMPC Samples:
    airplane	    :4045
    trash bin	    :343
    bag	            :83
    basket	        :113
    bathtub	        :856
    bed	            :233
    bench	        :1813
    birdhouse	    :73
    bookshelf	    :452
    bottle	        :498
    bowl	        :186
    bus	            :939
    cabinet	        :1571
    camera	        :113
    can	            :108
    cap	            :56
    car	            :3510
    cellphone	    :831
    chair	        :6778
    clock	        :651
    keyboard	    :65
    dishwasher	    :93
    display	        :1093
    earphone	    :73
    faucet	        :744
    file cabinet	:298
    guitar	        :797
    helmet	        :162
    jar	            :596
    knife	        :423
    lamp	        :2317
    laptop	        :460
    loudspeaker	    :1597
    mailbox	        :94
    microphone	    :67
    microwaves	    :152
    motorbike	    :337
    mug	            :214
    piano	        :239
    pillow	        :96
    pistol	        :307
    flowerpot	    :602
    printer	        :166
    remote	        :66
    rifle	        :2373
    rocket          :85
    skateboard	    :152
    sofa	        :3173
    stove	        :218
    table	        :8436
    telephone	    :1089
    tower	        :133
    train	        :389
    watercraft	    :1939
    washer	        :169

'''