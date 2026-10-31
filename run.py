import torch
import argparse
import time

import os
import sys

import logging
import json


from utils import graph_utils
from data_provider import data_prepare, build_prior
from engine import Exp


parser = argparse.ArgumentParser()

# load configuration from files
parser.add_argument('--config_file', type=str, default='none')

# the information about the data preprocessing
parser.add_argument('--dataset_name', type=str, default='none')
parser.add_argument('--division_name', type=str, default='div1')
parser.add_argument('--input_length', type=int, default=-1)
parser.add_argument('--predict_length', type=int, default=-1)
parser.add_argument('--scaler_type', type=str,default='zscore')
parser.add_argument('--slice_size_per_day', type=int, default=-1)
parser.add_argument('--scale_channel_wise', type=int, default=0)

parser.add_argument('--dataset_seed', type=int, default=1)
parser.add_argument('--adj_type', type=str, default='auto')
parser.add_argument('--adj_link_or_distance', type=str, default='distance')


parser.add_argument('--test_stages', type=str, default='3,4,5,6')

### for ST-ICA
parser.add_argument('--prompt_type', type=str, default='DA') ## prompt_type == 'DA' means to use ST-ICA
parser.add_argument('--static_func_type', type=str, default='featMLP')
parser.add_argument('--dynamic_func_type', type=str, default='gcn2')
parser.add_argument('--adj_dropout', type=float, default=0.2)

parser.add_argument('--random_prior_dim', type=int, default=64)
parser.add_argument('--random_prior_sigma', type=float, default=1.0)
parser.add_argument('--random_prior_type', type=int, default=2)

# the hyper-parameter-setting in the model
### default for all models (if applied). -1 means it should be loaded from config json file.
parser.add_argument('--modelid', type=str, default='STID')
parser.add_argument('--hid_dim', type=int, default=-1) 
parser.add_argument('--n_heads', type=int, default=8) 
parser.add_argument('--num_layers', type=int, default=-1) # 
parser.add_argument('--tau', type=int, default=3, help='0:no TCN')
parser.add_argument('--hasTemb', type=int, default=1)
parser.add_argument('--hasRawSemb', type=int, default=0) #### for expanding-node forecasting, original spatial embeddings are not used
parser.add_argument('--hasSTencoder', type=int, default=1)
parser.add_argument('--enc_dropout', type=float, default=0.1) 
parser.add_argument('--att_dropout', type=float, default=0.1)
parser.add_argument('--emb_dropout', type=float, default=0.1)
parser.add_argument('--te_emb_dropout', type=float, default=-1)
parser.add_argument('--se_emb_dropout', type=float, default=-1)
parser.add_argument('--return_att', type=int, default=0)
parser.add_argument('--revin', type=int, default=0)
parser.add_argument('--revin_type', type=str, default='ST')
parser.add_argument('--finetune_epochs', type=int, default=-1) 
parser.add_argument('--adapter_mlp_layers', type=int, default=1)


#### for GWNET
parser.add_argument('--addaptadj', type=int, default=0)
parser.add_argument('--aptonly', type=int, default=0)
parser.add_argument('--use_ST-ICA_as_adjadp', type=int, default=0)
parser.add_argument('--addST-ICAemb_GWN', type=int, default=0)

#### for DUET
parser.add_argument('--noisy_gating', type=int, default=1)
parser.add_argument('--num_experts', type=int, default=4)
parser.add_argument('--DUET_k', type=int, default=3)
parser.add_argument('--CI', type=int, default=1)
parser.add_argument('--moving_avg', type=int, default=3)

# PatchTST    
parser.add_argument('--d_ff', type=int, default=128, help='dimension of fcn')
parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')
parser.add_argument('--patch_len', type=int, default=4, help='patch length')
parser.add_argument('--stride', type=int, default=2, help='stride')
parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
# parser.add_argument('--revin', type=int, default=0, help='RevIN; True 1 False 0')
parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
parser.add_argument('--decomposition', type=int, default=0, help='decomposition; True 1 False 0')
parser.add_argument('--kernel_size', type=int, default=25, help='decomposition-kernel')
parser.add_argument('--tcn_output_dim', type=int, default=320, help='decomposition-kernel')
parser.add_argument('--tcn_layer', type=int, default=2, help='decomposition-kernel')
parser.add_argument('--tcn_hidden', type=int, default=160, help='decomposition-kernel')
parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')

## for OneNet_minus
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')

## for FlashST
parser.add_argument('--FlashST_embed_dim', type=int, default=32)

# the hyper-parameter-setting of the training process
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--learning_rate', type=float,default=0.001)
parser.add_argument('--weight_decay', type=float,default=0.0001)
parser.add_argument('--epochs', type=int, default=100)
parser.add_argument('--print_every', type=int, default=50)
parser.add_argument('--early_stop', type=bool, default=False)
parser.add_argument('--early_stop_step', type=int, default=20)
parser.add_argument('--lr_decay', type=int, default=1)
parser.add_argument('--huber_delta', type=int, default=2, help='delta in huber loss')
parser.add_argument('--loss_type', type=str, default="huber")
parser.add_argument('--finetune_type', type=str, default="finetune")
parser.add_argument('--eval_mask', type=int, default=0, help='eval_mask; {-1: not mask; 0: exclude zero; x: exclude values below x}')
parser.add_argument('--lr_scheduler_type', type=str, default='none')
parser.add_argument('--save_epochs', type=str, default='')
parser.add_argument('--save_output', type=int, default=0)
parser.add_argument('--note', type=str, default='')
parser.add_argument('--load_expid', type=str, default=None)
parser.add_argument('--load_note', type=str, default='')
parser.add_argument('--exp_name', type=str, default='Ori')


parser.add_argument('--online_adapt', type=int, default=1)
parser.add_argument('--adapt_modules', type=str, default='DA_dynamic_prior')
parser.add_argument('--online_st_ttc', type=int, default=0) # use ST-TTC

parser.add_argument('--testing', type=int, default=1)

def merge_args(args1, args2):
    add_flag = False
    merged_args = argparse.Namespace(**vars(args1))
    for key, value in vars(args2).items():
        if not hasattr(merged_args, key):
            add_flag = True
            setattr(merged_args, key, value)
    setattr(merged_args, 'note', args2.note)
    setattr(merged_args, 'update_type', args2.update_type)
    setattr(merged_args, 'finetune_epochs', args2.finetune_epochs)
    return merged_args, add_flag

def get_logger(log_dir, log_filename, name=None):
    logfilepath = os.path.join(log_dir, log_filename)

    logger = logging.getLogger(name)

    log_level = 'DEBUG'

    if log_level.lower() == 'info':
        level = logging.INFO
    elif log_level.lower() == 'debug':
        level = logging.DEBUG
    elif log_level.lower() == 'error':
        level = logging.ERROR
    elif log_level.lower() == 'warning':
        level = logging.WARNING
    elif log_level.lower() == 'critical':
        level = logging.CRITICAL
    else:
        level = logging.INFO

    logger.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler(logfilepath)
    file_handler.setFormatter(formatter)

    console_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(console_formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info('Log directory: %s', log_dir)
    return logger

def main():
    
    args = parser.parse_args()

    current = time.strftime("%m%d%H%M%S", time.localtime())
    if args.load_expid is None:
        expid = current 
    else:
        expid = args.load_expid
    

    device = torch.device(args.device)

    args.cache_data_path = 'datasets'
    if len(args.save_epochs)>0:
        save_epoch_list = args.save_epochs.split('_')
        save_epoch_list = [int(it) for it in save_epoch_list]
    else:
        save_epoch_list = None
    
    if os.path.exists(f'./configurations/{args.config_file}'):
        with open(f'./configurations/{args.config_file}', 'r') as f:
            x = json.load(f)
            data_config = x['data_config']
            model_config = x['model_config']
            training_config = x['training_config']

            args.dataset_name =data_config['dataset_name']
            args.slice_size_per_day = data_config['slice_size_per_day']
            args.in_dim = data_config['in_dim']
            args.pre_dim = data_config['pre_dim']
            args.input_length = data_config['input_length']
            args.predict_length = data_config['predict_length']
            
            args.hid_dim = model_config['hid_dim'] 
            args.n_heads = model_config['n_heads']
            args.num_layers = model_config['num_layers']
            args.se_emb_dropout = model_config['se_emb_dropout'] if args.se_emb_dropout == -1 else args.se_emb_dropout
            args.te_emb_dropout = model_config['te_emb_dropout'] if args.te_emb_dropout == -1 else args.te_emb_dropout

            args.loss_type = training_config['loss_type']
            
    else:
        print('there is no config file!')
    
    if args.modelid == 'GWNET':
        if args.prompt_type == 'DA':
            args.addaptadj = False
            args.use_ST-ICA_as_adjadp = False
            args.addST-ICAemb_GWN = True

    random_prior_dim_dict ={'EPeMS':512,'AQI':64,'METR-LA':256,'UrbanEV_s':512,'GBA':64}
    args.random_prior_dim = random_prior_dim_dict[args.dataset_name]
    args.expid = expid
    args.norm_y = 0
    args.inverse = True

    suffix = '' if args.note =='' else f'_{args.note}'
    args.note = f'bs{args.batch_size}_d{args.hid_dim}_L{args.num_layers}{suffix}'
    save_path = f'./EXP_experiments/{args.modelid}/{args.dataset_name}/{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp' + str(
        args.expid) + "/"
    log_path = save_path

    if args.load_expid is None:
        log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{args.expid}.log'
    else:
        log_filename = f'{args.modelid}_{args.dataset_name}_{args.input_length}to{args.predict_length}_exp{args.expid}_load_ON{current}.log'
    
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    logger = get_logger(log_path, log_filename)
    exp_name = f'{args.modelid}_{args.dataset_name}'
    logger.info(exp_name)

    print('args.slice_size_per_day*7:', args.slice_size_per_day*7)
    

    full_data, full_time, full_df_time, full_valid_mask, node_ids_full = data_prepare.load_full_data_time(args.dataset_name,1)
    full_adj = graph_utils.load_full_adj(args.dataset_name, node_ids_full, 'distance')


    stage_spec = json.load(open(f"datasets/{args.dataset_name}/{args.division_name}/stage_spec.json","r",encoding="utf-8"))
    stage_adj_dict, stage_ori_adj, stage_lpls_dict, stage_nadj_dict = graph_utils.build_stage_adj_dict(full_adj, stage_spec, device=device, lpls=args.prompt_type=='FlashST')
    args.support_len = len(stage_adj_dict['1'])


    # L, N, C = full_data.shape
    
    train_stages = [str(it) for it in [1]]
    val_stages = [str(it) for it in [2]]
    test_stages = [str(it) for it in [3,4,5,6]]
    if args.test_stages != '3,4,5,6':
        test_stages = args.test_stages.split(',')
    
    dataloader, scaler_list = data_prepare.build_stagewise_splits(full_data, full_time, stage_spec, 
                                                                  train_stages, val_stages, test_stages, 
                                                                  args.input_length, args.predict_length, 
                                                                  batch_size=args.batch_size, scaler_mode='concat')
    
    prior_dict = {}   
    train_scaler= scaler_list[-1]
    
    torch.cuda.empty_cache()
    logger.info(args)
    
    
    args.first_stage_node_idx_dict = data_prepare.get_stage1_survivor_index_dict(stage_spec, stage1_id='1')
    args.note = f'{args.note}_{args.load_note}'
    args.stage_adj_dict = stage_adj_dict
    args.stage_lpls_dict = stage_lpls_dict
    args.stage_nadj_dict = stage_nadj_dict

    args.num_nodes = stage_adj_dict['1'][0].shape[0]
    args.stage_spec = stage_spec    
    if args.random_prior_type == 2:
        prior_dict = build_prior.build_stage_prior_dict_calibrated(args.stage_spec, args.random_prior_dim, sigma=args.random_prior_sigma, seed=42)
    
    elif args.random_prior_type == 3:
        prior_dict = build_prior.build_stage_prior_dict_calibrated_from_first(args.stage_spec, args.random_prior_dim, sigma=args.random_prior_sigma, seed=42)
    elif args.random_prior_type == 4:
        logger.info(f'prior type: All Same')
        prior_dict = build_prior.build_stage_prior_dict_all_same(args.stage_spec, args.random_prior_dim, sigma=args.random_prior_sigma, seed=42)
    elif args.random_prior_type == 5:
        logger.info(f'prior type: Stage New')
        prior_dict = build_prior.build_stage_prior_dict_stage_new(args.stage_spec, args.random_prior_dim, sigma=args.random_prior_sigma, seed=42)
    
    args.stage_prior_dict = prior_dict
    args.prior_feat_dim = prior_dict['1'].shape[-1]

    logger.info(args.note)
    logger.info("start training...")
    engine = Exp(scaler= train_scaler, device=device, args=args)

    engine.training(train_stages, val_stages, dataloader, save_path, mode='train')


    if bool(args.testing):
        logger.info("start testing...")
        print(f'{save_path}/best.pth')
        engine.model.load_state_dict(torch.load(save_path + "best.pth"))
        engine.testing(test_stages, dataloader['test'], stage_spec, exp_name=args.exp_name)
    if bool(args.online_adapt):
        logger.info("start online adapting...")
        print(f'{save_path}/best.pth')
        engine.model.load_state_dict(torch.load(save_path + "best.pth"))
        engine.testing(test_stages, dataloader['test'], stage_spec, test_finetuning=True, exp_name=args.exp_name)
    if bool(args.online_st_ttc):
        logger.info("start online adapting with st_ttc...")
        engine.model.load_state_dict(torch.load(save_path + "best.pth"))
        engine.testing(test_stages, dataloader['test'], stage_spec, test_ttc=True, exp_name=args.exp_name)

    

if __name__ == "__main__":
    t1 = time.time()
    main()
    t2 = time.time()
    print("Total time spent: {:.4f}".format(t2 - t1))
