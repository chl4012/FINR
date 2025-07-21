import argparse
import os
import sys
import json
import h5py
import shutil
import numpy as np
from datetime import datetime

def parse_args():        
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--gpu_num", type=int, default=0)
    parser.add_argument("--gpu_num_2", type=int, default=0) #gpu_numThe second device for 3D case
    parser.add_argument("--use_multi_gpus", type=bool, default=True) #How many gpus are there
    parser.add_argument('--gpu_ids', nargs='+', type=int, default=[2,3], help='gpu ids for parallel computing')
    parser.add_argument('--fp16', type = bool, default = True)
    parser.add_argument('--ff', action='store_true', help="use fully-fused MLP")
    parser.add_argument('--tcnn', action='store_true', help="use TCNN backend")


    

    # models------------------------------2D CNN----------------------------------------
    parser.add_argument("--Nr", type=int, default=4) #depth of cnn
    parser.add_argument("--ndf", type=int, default=512) #channel of cnn
    parser.add_argument("--input_nch", type=int, default=1) # output channel of the mapper
    parser.add_argument("--output_nch", type=int, default=8) #4 singular images
    parser.add_argument("--need_bias", action="store_true")
    parser.add_argument("--up_factor", type=int, default=4) 
    parser.add_argument("--upsample_mode", type=str, default="nearest")    
    parser.add_argument("--isMGRE", type=bool, default=False)    
    parser.add_argument("--solver_type", type=str, default="MRF_fat_water_STN")    
    """
    'MRF_fat_water_STN'
    """
    
    # models------------------------------HashGrid----------------------------------------
    parser.add_argument('--HashGrid_hidden_dim', type=int, default=64,
                        help='Hidden dimension of MLP') # mapping net HashGrid_num_layers
    parser.add_argument('--HashGrid_num_layers', type=int, default=4,
                        help='number of layers of MLP')
    parser.add_argument('--HashGrid_num_lavels', type=int, default=16,
                        help='number of levels of singular images')
    parser.add_argument('--attachmag2_r2s', type=bool, default=False,
                        help='')
    parser.add_argument('--attachmixed', type=bool, default=True,
                        help='')
    parser.add_argument('--use_siren', type=bool, default=False,
                        help='use sine activation')
    parser.add_argument('--use_inits', type=bool, default=False,
                        help='use initializations')
    parser.add_argument('--dropout_rate', type=float, default=0.0,
                        help='dropout rate for MLP') # mapping net HashGrid_num_layers
    

    #--------------------------------------Hashgrid_Dfield-----------------------------------
    parser.add_argument('--D_hidden_dim', type=int, default=32,
                        help='Hidden dimension of MLP for Displacement field') # mapping net HashGrid_num_layers
    parser.add_argument('--D_num_layers', type=int, default=4,
                        help='number of layers of MLP for Displacement field')
    parser.add_argument('--D_num_levels', type=int, default=4,
                        help='number of levels of MLP for Displacement field')
    

    #--------------------------------------------for 3D-------------------------------------------
    parser.add_argument("--network_type_3D", type=str, default="MDDIP3D")#true for MGRE recon
    parser.add_argument("--Necho", type=int, default=5) 
    parser.add_argument("--T1_stacks", type=int, default=8) 
    parser.add_argument("--T2_stacks", type=int, default=8) 
    parser.add_argument("--MGRE_stacks", type=int, default=8) 
    parser.add_argument("--nslice_se", type=int, default=20) 
    parser.add_argument("--nslice_me", type=int, default=5) 
    parser.add_argument("--Nslice", type=int, default=64) 
    
    
    
    #-----------------------------------------FGN---------------------------------------
    parser.add_argument("--FGN_num_layers", type=int, default=2)
    parser.add_argument("--FGN_hidden_dim", type=int, default=32)
    parser.add_argument("--FGN_dropout_rate", type=float, default=0.0,
                        help='dropout rate for FGN MLP') # mapping net HashGrid_num_layers
    
    
    #-----------------------------------------PEN---------------------------------------
    #hyperparameters for parameter estimation network
    parser.add_argument("--PEN_num_layers", type=int, default=4)
    parser.add_argument("--PEN_hidden_dim", type=int, default=64)
    parser.add_argument("--PEN_dropout_rate", type=float, default=0.0, 
                        help='dropout rate for PEN MLP') # mapping net HashGrid_num_layers
    


    #------------------------------------------dataset------------------------------------------------------------
    parser.add_argument("--dataset", type=str, default="Retrospective")
    parser.add_argument("--file_dir", type=str, default= "")

    parser.add_argument("--used_fr_min", type=int, default=1)      
    parser.add_argument("--used_fr_max", type=int, default=1600) 
    parser.add_argument("--window_size", type=int, default=5) #Chao Li: number of leaves in a sliding window
    parser.add_argument("--window_size_MGRE", type=int, default=20) #Chao Li: number of leaves in a sliding window for MGRE part
    parser.add_argument("--data_usage_ratio", type=float, default=1) #Chao Li: number of leaves in a sliding window for MGRE part
    parser.add_argument("--echo_used", type=int, default=5) #Chao Li: number of leaves in a sliding window for MGRE part
    parser.add_argument("--use_in_phase", type=bool, default=False) 
    parser.add_argument("--use_direct", type=bool, default=False) 
    parser.add_argument("--motion_averaged", type=bool, default=False)
    parser.add_argument("--Nsingular_used", type=int, default=5) #Chao Li: number of leaves in a sliding window for MGRE part
    parser.add_argument("--Dnet_ckp", type=str, default=None)
    

    # ---------------------------------------------training setups----------------------------------------------
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=2000) # scheduler
    parser.add_argument("--gamma", type=float, default=0.5) # scheduler
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--max_steps", type=int, default=6) 
    parser.add_argument('--slice', type=int,  default=14)   
    parser.add_argument('--change_route_epoch', type=int,  default=4)   


    # misc
    parser.add_argument("--noPLOT", action="store_true")
    parser.add_argument("--isdev", action="store_true")  
    parser.add_argument("--istest", type=bool, default=False) 
    parser.add_argument("--isresume", type=str , default=None)    
    parser.add_argument("--resumeMRF_4_r2s", type=str, default="./logs/")  
    parser.add_argument("--ckpt_root", type=str, default="./logs/")    
    parser.add_argument("--save_period", type=int, default=200)
    parser.add_argument("--memo", type=str, default="")
    
    #3D
    parser.add_argument("--Q_kdata_fname", type=str, default="Q_kdata_8_bins")
    return parser.parse_args() 


def make_template(opt):

    if opt.isdev:
        ckpt_folder = "retro_"
    else:
        now = datetime.now()
        curr_time = now.strftime("%Y%m%d_%H%M%S")    
        subject_name = opt.file_dir.split("/")[-3]
        ckpt_folder = "{}_{}".format(curr_time,subject_name)

    opt.ckpt_root = opt.ckpt_root + ckpt_folder
    print(opt.ckpt_root)
    os.makedirs(opt.ckpt_root, exist_ok=True)

    with open(os.path.join(opt.ckpt_root, 'myparam.json'), 'w') as f:
        json.dump(vars(opt), f)

    with open(opt.ckpt_root+"/command_line_log.txt", "w") as log_file:
        log_file.write("python %s" % " ".join(sys.argv))

    shutil.copy(os.path.join(os.getcwd(),__file__),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'solver.py'),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'option.py'),opt.ckpt_root)
    shutil.copy(os.path.join(os.getcwd(),'models','{}.py'.format(opt.model)),opt.ckpt_root) 
    
    # model
    if "netowrk" in opt.model:
        opt.memo = '%s: basic net, %s' % (opt.dataset, opt.input_type) + opt.memo    
    else:
        raise NotImplementedError('what is it?')


def get_option():    
    opt = parse_args()
    if opt.isresume is None:
        make_template(opt)
    else: 
        with open(os.path.join(os.path.dirname(opt.isresume), 'myparam.json'), 'rt') as f:
            t_args = argparse.Namespace()
            t_args.__dict__.update(json.load(f))
            t_args.isresume = opt.isresume
            t_args.ckpt_root = os.path.dirname(opt.isresume)
            t_args.istest = opt.istest
            t_args.gpu_ids = opt.gpu_ids
            t_args.gpu_num = opt.gpu_num
            t_args.window_size = opt.window_size
            t_args.Dnet_ckp = opt.Dnet_ckp
            t_args.change_route_epoch = opt.change_route_epoch
            t_args.max_steps = opt.max_steps
            t_args.solver_type = opt.solver_type
            t_args.file_dir = opt.file_dir
            t_args.lr = opt.lr
            opt = t_args
        if opt.istest:    
            opt.use_multi_gpus = False
        
            
    return opt

