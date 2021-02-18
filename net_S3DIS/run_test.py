import argparse
import os
import test

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--log_dir', default='log', help='Log dir [default: log]')
parser.add_argument('--num_class', type=int, default=9, help='number of class')
parser.add_argument('--num_point', type=int, default=16384, help='Point number [default: 4096]')
parser.add_argument('--channel', type=int, default=6, help='number of feature channel [default: 6]')
parser.add_argument('--testdataset_dir', type=str, default='/mnt/cm-nas03/backup/students/qbai/Datasets/point_clouds/Hannover_AreaMapping_4_labeled_feat_type_h5/val', help='testdataset path')
parser.add_argument('--test_dir', type=str, default='/mnt/cm-nas03/backup/students/qbai/Datasets/point_clouds/Hannover_AreaMapping_5_predicted/pred_files_val_feat', help='The file contains original test data')
parser.add_argument('--outdir', type=str, default='/mnt/cm-nas03/backup/students/qbai/Datasets/point_clouds/Hannover_AreaMapping_5_predicted/GACNet/labels_pred_true')

args = parser.parse_args()

#graph_inf contents parameters for grapg building and coarsing
graph_inf = {'stride_list': [4, 4, 4, 2],
             'radius_list': [0.7, 1.4, 2.8, 5.6, 11.2],
             'maxsample_list': [12, 21, 21, 21, 12]
}

# number of units for each mlp layer
forward_parm = [
                [ [32,32,64], [64] ],
                [ [64,64,128], [128] ],
                [ [128,128,256], [256] ],
                [ [256,256,512], [512] ],
                [ [256,256], [256] ]
]

# for feature interpolation stage 
upsample_parm = [
                  [128, 128],
                  [128, 128],
                  [256, 256],
                  [256, 256]
]

# parameters for fully connection layer
fullconect_parm = 128

net_inf = {'forward_parm': forward_parm,
           'upsample_parm': upsample_parm,
           'fullconect_parm': fullconect_parm
}


if not os.path.exists(args.outdir): os.mkdir(args.outdir)

test.test(args, graph_inf, net_inf)
test.interpolate(args)
test.acc_report(args)
