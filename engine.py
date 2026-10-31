import csv
import torch
from torch import nn
import torch.optim as optim

from models import DUET, iTransformer, DLinear, PatchTST
from models import GWNET, DCRNN
from models import STID, STAEformer
from models import STOP
from models import Time_FSNet, Time_TCN, OneNet_minus

import torch.nn.functional as F
from tqdm import tqdm
from logging import getLogger
from utils import  metric
from functools import partial
import pickle
import sys

import time
import numpy as np
import os
import queue
from layers.ST_TTC import DynamicFRPlusModule


class Exp():
    def __init__(self, args, device, scaler):
        self._logger = getLogger()
        self.args = args
        self.scaler = scaler
        self.device = device
        self.clip = 1
        self.eval_mask = args.eval_mask
        self.inverse = args.inverse
        self.epochs = args.epochs
        
        self.current_pid = os.getpid()

        self.model = self._build_model(args.modelid)
        self.loss = self._build_loss_func(args.loss_type)
        self.store_grad = args.modelid in ['Time_FSNet']
        self.first_stage_node_order = args.stage_spec['1']['node_order']

    def _build_model(self, modelid):
        model_dict = {
            'GWNET':GWNET,
            'STID':STID,
            'STAEformer':STAEformer,
            'iTransformer':iTransformer,
            'DUET':DUET,
            'DLinear':DLinear,
            'DCRNN':DCRNN,
            'Time_FSNet':Time_FSNet,
            'Time_TCN':Time_TCN,
            'OneNet_minus':OneNet_minus,
            'PatchTST':PatchTST
        }
        self.modelid = modelid
        if modelid == 'STOP':
            base = STOP.MLP(# node_num=node_num,
                num_layer=3, 
                model_dim=64, 
                prompt_dim=32, 
                tod_size=self.args.slice_size_per_day, 
                kernel_size=3,
                args=self.args)
            model = STOP.Model(self.args, stmodel=base)
        elif modelid in model_dict.keys():
            model = model_dict[modelid].Model(self.args) 
        else:
            self._logger.error(
                f'no model named `{modelid}`, please try again!')
            sys.exit('error')

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        model.to(self.device)

        self._logger.info(model)
        for name, param in model.named_parameters():
            self._logger.info(str(name) + '\t' + str(param.shape) + '\t' +
                              str(param.device) + '\t' + str(param.requires_grad))
        total_para_num, trainable_para_num = get_parameter_number(model)
        self.total_para_num, self.trainable_para_num = total_para_num, trainable_para_num
        self._logger.info(f'total_parameter_num is {total_para_num}, trainable_parameter_num is {trainable_para_num}')   
        return model
    
    def _build_loss_func(self, loss_type):
        if loss_type == 'huber':
            loss_func = partial(metric.huber_loss, delta=self.args.huber_delta)
        elif loss_type == 'masked_huber_0':
            loss_func = partial(metric.masked_huber_loss, delta=self.args.huber_delta, null_val = 0.)
        elif loss_type == 'masked_mae_0':
            loss_func = partial(metric.masked_mae_torch, null_val = 0.)
        elif loss_type == 'masked_huber_eval':
            loss_func = partial(metric.masked_huber_loss, delta=self.args.huber_delta, null_val = 0., mask_val= self.args.eval_mask)
        return loss_func


    def _forward(self, input, mode=None):
        output, emb_or_loss = self.model(input, mode=mode)  # output:(B,T,N,C) #real: (B,T,N,C)
        if self.args.inverse:
            predict = self.scaler.inverse_transform(output)
        else:
            predict = output

        return predict, emb_or_loss

    def _metric(self, predict, real):
        filter_num = 1e-5
        if self.eval_mask == -1:
            mae = metric.mae(predict, real).item()
            rmse = metric.rmse(predict, real).item()
            mape = metric.masked_mape_torch(predict, real, 0.0).item()
        elif self.eval_mask == 0:
            mae = metric.masked_mae_torch(predict, real, mask_val=0.0, filter_num=filter_num).item()
            rmse = metric.masked_rmse_torch(predict, real, mask_val=0.0, filter_num=filter_num).item()
            mape = metric.masked_mape_torch(predict, real, mask_val=0.0, filter_num=filter_num).item()
        return mae, mape, rmse

    def train_epoch(self, input, real_val, mode=None):  # input(B,T,N,C), real_val(B,T,N,C)
        self.model.train()
        self.optimizer.zero_grad()
        predict, emb = self._forward(input, mode=mode)
        predict_ready = predict
        real_val_ready = real_val


        loss = self.loss(predict_ready, real_val_ready)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        

        self.optimizer.step()
        if self.store_grad:
            self.model.store_grad()

        mae, mape, rmse = self._metric(predict_ready, real_val_ready)
        return [loss.item(), mae, mape, rmse], emb

    def eval_epoch(self, input, real_val, test_node_set=None, stage=None):
        self.model.eval()
        predict, _ = self._forward(input, mode='val')
        if test_node_set is not None:
            predict_ready = predict[:,:,test_node_set,:]
            real_val_ready = real_val[:,:,test_node_set,:]
        else:
            predict_ready = predict
            real_val_ready = real_val
        loss = self.loss(predict_ready, real_val_ready)
        mae, mape, rmse = self._metric(predict_ready, real_val_ready)
        return loss.item(), mae, mape, rmse

    def train_one_stage(self, loader, stage_id, mode='train'):
        self.model.train()
        
        aux_data = {'adj_mx':self.args.stage_adj_dict[stage_id],
                    'prior':self.args.stage_prior_dict[stage_id].to(self.device)}
        if self.args.prompt_type == 'FlashST':
            aux_data['lpls']= self.args.stage_lpls_dict[stage_id]
            aux_data['nadj']= self.args.stage_nadj_dict[stage_id]


        for iter, batch in enumerate(loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            tx = batch["tx"].to(self.device)
            
            if self.args.modelid in ['HimNet', 'DCRNN']:
                ty = batch["ty"].to(self.device)
                trainx = [x, tx, y, ty, self.batches_seen, aux_data]
            elif self.args.modelid in ['GMAN']:
                ty = batch["ty"].to(self.device)
                aux_data['SE'] = self.args.stage_GMAN_SE_dict[stage_id].to(self.device)
                trainx = [x,tx,ty,aux_data]
            else:
                trainx = [x,tx,aux_data]

            trainy = y
            metrics, emb = self.train_epoch(trainx, trainy, mode=mode)
            
            self.batches_seen += 1
            self.train_loss.append(metrics[0])
            self.train_mae.append(metrics[1])
            self.train_mape.append(metrics[2])
            self.train_rmse.append(metrics[3])
            if iter % self.args.print_every == 0:
                log = 'Iter: {:03d} [{:d}], Train Loss: {:.4f},Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f}'
                self._logger.info(log.format(
                    iter, self.batches_seen, self.train_loss[-1], self.train_mae[-1], self.train_mape[-1], self.train_rmse[-1]))
        return 

    def eval_one_stage(self, loader, stage_id, node_set=None):
        self.model.eval()
        aux_data= {'adj_mx':self.args.stage_adj_dict[stage_id],
                    'prior':self.args.stage_prior_dict[stage_id].to(self.device)}
        if self.args.prompt_type == 'FlashST':
            aux_data['lpls']= self.args.stage_lpls_dict[stage_id]
            aux_data['nadj']= self.args.stage_nadj_dict[stage_id]

        if self.args.modelid in ['GMAN']:
            aux_data['SE'] = self.args.stage_GMAN_SE_dict[stage_id].to(self.device)

        if node_set is not None:                
            aux_data['adj_mx'] = [it[:,node_set][node_set, :] for it in self.args.stage_adj_dict[stage_id]]
            aux_data['prior'] = aux_data['prior'][node_set]
            if self.args.modelid in ['GMAN']:
                aux_data['SE'] = aux_data['SE'][node_set]
            if self.args.prompt_type == 'FlashST':
                aux_data['lpls'] = aux_data['lpls'][node_set]
                aux_data['nadj'] = aux_data['nadj'][:,node_set][node_set, :]


        for iter, batch in enumerate(loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            tx = batch["tx"].to(self.device)
            if node_set is not None:
                x = x[:,:,node_set]
                y = y[:,:,node_set]
            

            if self.args.modelid in ['DCRNN']:
                ty = batch["ty"].to(self.device)
                valx = [x, tx, y, ty, self.batches_seen, aux_data]
            else:
                valx = [x,tx,aux_data]
            
            metrics = self.eval_epoch(valx, y)
            self.valid_loss.append(metrics[0])
            self.valid_mae.append(metrics[1])
            self.valid_mape.append(metrics[2])
            self.valid_rmse.append(metrics[3])

    def training(self, train_stages, val_stages, dataloader,  save_path, is_finetune=False, log_tip='Forecasting',  mode=None, save_epoch_list=None, keep_train_epochs=None):
        all_start_time = time.time()
        his_loss = []
        trainloss_record = []
        val_time = []
        train_time = []
        best_validate_loss = np.inf
        validate_score_non_decrease_count = 0
        self.log_in_train_details = []
        
        epochs = self.args.epochs if keep_train_epochs is None else keep_train_epochs

        self.optimizer = optim.AdamW(self.model.parameters(), lr=self.args.learning_rate, weight_decay=self.args.weight_decay)

        for stage_id in train_stages:
            print(f'start training stage {stage_id}....')
            self.batches_seen = 0
            for i in range(1, epochs + 1):
                i = int(stage_id)*1000 + i
                # training
                self.train_loss, self.train_mae, self.train_mape, self.train_rmse = [], [], [], []
                t1 = time.time()
                train_mode = 'train'
                self.train_one_stage(dataloader['train'][stage_id], stage_id, mode=train_mode)
                t2 = time.time()
                train_time.append(t2 - t1)
                mtrain_loss = np.mean(self.train_loss)
                mtrain_mae = np.mean(self.train_mae)
                mtrain_mape = np.mean(self.train_mape)
                mtrain_rmse = np.mean(self.train_rmse)
                
                # validation
                self.valid_loss, self.valid_mae, self.valid_mape, self.valid_rmse = [], [], [], []
                s1 = time.time()
                for vs in val_stages:
                    self.eval_one_stage(dataloader['val'][vs], vs, node_set=self.args.first_stage_node_idx_dict[vs])
                s2 = time.time()
                log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
                self._logger.info(log.format(i, (s2 - s1)))
                val_time.append(s2 - s1)
                mvalid_loss = np.mean(self.valid_loss)
                mvalid_mae = np.mean(self.valid_mae)
                mvalid_mape = np.mean(self.valid_mape)
                mvalid_rmse = np.mean(self.valid_rmse)


                his_loss.append(mvalid_loss)
                trainloss_record.append(
                    [mtrain_loss, mtrain_mae, mtrain_mape, mtrain_mape, mvalid_loss, mvalid_mae, mvalid_mape, mvalid_rmse])
                log = f'Stage:{stage_id}, Epoch: {i}, Train Loss: {mtrain_loss:.4f}, Train MAE: {mtrain_mae:.4f} Train MAPE: {mtrain_mape:.4f}, Train RMSE: {mtrain_rmse:.4f}\nTraining Time: {(t2 - t1):.4f}/epoch\n'
                log += f'Valid Loss: {mvalid_loss:.4f}, Valid MAE: {mvalid_mae:.4f}, Valid MAPE: {mvalid_mape:.4f}, Valid RMSE: {mvalid_rmse:.4f}'
                self._logger.info(log)
                self.log_in_train_details.append([stage_id, i, mtrain_loss, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_loss, mvalid_mae, mvalid_mape,
                                            mvalid_rmse, (t2 - t1)])

                if best_validate_loss > mvalid_loss:
                    best_validate_loss = mvalid_loss
                    validate_score_non_decrease_count = 0
                    torch.save(self.model.state_dict(), save_path + "best.pth")

                    self._logger.info('got best validation result: {:.4f}, {:.4f}, {:.4f}'.format(
                        mvalid_loss, mvalid_mape, mvalid_rmse))
                    

                if save_epoch_list is not None:
                    if i in save_epoch_list:
                        torch.save(self.model.state_dict(), save_path+f'epoch{i}.pth')
    


                self._logger.info(f'---- {self.args.expid}_{self.args.modelid}_{self.args.dataset_name}_{self.args.note} ----')
                
                if self.args.early_stop and validate_score_non_decrease_count >= self.args.early_stop_step:
                    break
        


        avg_train_time = np.mean(train_time)
        avg_inference_time = np.mean(val_time)
    

        self._logger.info(
            "Average Training Time: {:.4f} secs/epoch".format(avg_train_time))
        self._logger.info("Average Valid Inference Time: {:.4f} secs".format(avg_inference_time))

        training_time = (time.time() - all_start_time) / 60

        bestid = np.argmin(his_loss)
        self._logger.info("Training finished")
        self._logger.info(
            f"The valid loss on best model is {str(round(his_loss[bestid], 4))}")
        train_metric = trainloss_record[bestid][0:4]
        valid_metric = trainloss_record[bestid][4:]

        try:
            with open(save_path + 'train_record.csv', 'a+') as f:
                fcsv = csv.writer(f)
                fcsv.writerow(['bestid',bestid,'train_loss_mae_mape_rmse']+ train_metric +['valid_loss_mae_mape_rmse'] + valid_metric)
        except:
            print('error in save train_record')

        return training_time, train_metric, valid_metric

    def test_one_stage(self, loader, stage_node_info, stage_id, inverse = True):
        outputs = []
        realy_list = []
        y_valid_mask_list =[]
        inference_time = 0
        self.model.eval()
        aux_data= {'adj_mx':self.args.stage_adj_dict[stage_id],
                'prior':self.args.stage_prior_dict[stage_id].to(self.device)}
        if self.args.prompt_type == 'FlashST':
            aux_data['lpls']= self.args.stage_lpls_dict[stage_id]
            aux_data['nadj']= self.args.stage_nadj_dict[stage_id]

        for iter, batch in enumerate(loader):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            tx = batch["tx"].to(self.device)
            
            if self.args.modelid in ['DCRNN']:
                ty = batch["ty"].to(self.device)
                testx = [x, tx, None, ty, None, aux_data]
            else:
                testx = [x,tx,aux_data]

            with torch.no_grad():
                t1 = time.time()
                preds, _ = self.model(testx, mode='test')
                inference_time += time.time()-t1
            outputs.append(preds)
            realy_list.append(y)

        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy_list, dim=0)
        if inverse:
            yhat = self.scaler.inverse_transform(yhat)

        if len(y_valid_mask_list)>0:
            y_valid_mask = torch.cat(y_valid_mask_list, dim=0)
            assert y_valid_mask.shape == realy.shape
        else:
            y_valid_mask = None
        print('yhat.shape:',yhat.shape)
        print('realy.shape:', realy.shape)
        print('inference_time:', inference_time)

        if self.args.predict_length == 0:
            ready_length = self.args.input_length
        else:
            ready_length = self.args.predict_length

        
        all_metrics = []
        all_metrics_remain = []
        all_metrics_newadd = []

        filter_num = 1e-5
        feature_idx = 0
        pred_feature = yhat[..., feature_idx]
        real_feature = realy[..., feature_idx]
        if y_valid_mask is not None:
            y_valid_mask_feature = y_valid_mask[..., feature_idx]

        for i in range(ready_length):
            pred = pred_feature[:, i]
            real = real_feature[:, i]

            if y_valid_mask is not None:
                y_valid_mask_feature_ready = y_valid_mask_feature[:, i].to(pred.device)
            else:
                y_valid_mask_feature_ready = None
            metrics = metric.metric_torch(pred, real, mask_val=self.args.eval_mask, valid_mask=y_valid_mask_feature_ready, filter_num=filter_num)
            log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            self._logger.info(log.format(feature_idx, i + 1, metrics[0], metrics[1], metrics[2]))
            all_metrics.append(metrics)

            if stage_node_info is not None:
                num_remain_nodes = stage_node_info['num_visible_nodes'] - stage_node_info['num_new_nodes']
                num_newadd_nodes = stage_node_info['num_new_nodes']
                pred_subset1 = pred[:,:num_remain_nodes]
                real_subset1 = real[:,:num_remain_nodes]
                pred_subset2 = pred[:,-num_newadd_nodes:]
                real_subset2 = real[:,-num_newadd_nodes:]
                metrics_subset1 = metric.metric_torch(pred_subset1, real_subset1, mask_val=self.args.eval_mask,filter_num=filter_num)
                metrics_subset2 = metric.metric_torch(pred_subset2, real_subset2, mask_val=self.args.eval_mask,filter_num=filter_num)
                all_metrics_remain.append(metrics_subset1)
                all_metrics_newadd.append(metrics_subset2)

        amae, amape, armse, amre, ar2 = map(list, zip(*all_metrics))
        
        MAE = np.mean(amae)
        MAPE = np.mean(amape)
        RMSE = np.mean(armse)
        MRE = np.mean(amre)
        R2 = np.mean(ar2)

        log = f'[dim{feature_idx:d}] On average over {ready_length:d} horizons, Test MAE: {MAE:.4f}, Test MAPE: {MAPE:.4f}, Test RMSE: {RMSE:.4f} Test MRE: {MRE:.4f} Test R2: {R2:.4f}'
        self._logger.info(log)

        if stage_node_info is not None:
            amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1 = map(list, zip(*all_metrics_remain))
            amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2 = map(list, zip(*all_metrics_newadd))
            MAE_sub1 = np.mean(amae_sub1)
            MAPE_sub1 = np.mean(amape_sub1)
            RMSE_sub1 = np.mean(armse_sub1)
            MRE_sub1 = np.mean(amre_sub1)
            R2_sub1 = np.mean(ar2_sub1)
            MAE_sub2 = np.mean(amae_sub2)
            MAPE_sub2 = np.mean(amape_sub2)
            RMSE_sub2 = np.mean(armse_sub2)
            MRE_sub2 = np.mean(amre_sub2)
            R2_sub2 = np.mean(ar2_sub2)
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                'remain_mean':[MAE_sub1, MAPE_sub1, RMSE_sub1, MRE_sub1, R2_sub1],
                'newadd_mean':[MAE_sub2, MAPE_sub2, RMSE_sub2, MRE_sub2, R2_sub2],
                    'all_full':[amae, amape, armse, amre, ar2],
                'remain_full':[amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1],
                'newadd_full':[amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2],}, inference_time
        else:
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                    'all_full':[amae, amape, armse, amre, ar2]}, inference_time





    def test_finetuning_one_stage(
        self, loader, stage_node_info, stage_id, inverse=True
    ):
        outputs = []
        realy_list = []
        inference_time = 0.0
        adapt_time = 0.0
        self.model.eval()
        # ---------- online adaptation config ----------
        online_adapt = True
        queue_size = int(getattr(self.args, "adapt_queue_size", self.args.input_length))
        adapt_steps = int(getattr(self.args, "adapt_steps", 1))
        adapt_every = int(getattr(self.args, "adapt_every", 1))  
        adapt_lr = float(getattr(self.args, "adapt_lr", 1e-4))
        adapt_wd = float(getattr(self.args, "adapt_wd", 0.0))
        grad_clip = float(getattr(self.args, "adapt_grad_clip", 0.0))
        loss_type = getattr(self.args, "adapt_loss", "huber")  # "mse" | "mae" | "huber"
        adapt_modules = self.args.adapt_modules 

        def _build_input(_x, _tx, _aux):
            if self.args.modelid in ['DCRNN']:
                ty = batch["ty"].to(self.device)
                testx = [_x, _tx, None, ty, None, _aux]
            else:
                testx = [_x,_tx,_aux]
            return testx

        def _get_loss(pred, target):
            if loss_type == "mae":
                return F.l1_loss(pred, target)
            if loss_type == "huber":
                return F.smooth_l1_loss(pred, target)
            return F.mse_loss(pred, target)

        # stage-wise constant aux data
        prior = self.args.stage_prior_dict[stage_id].to(self.device)
        adj_mx = self.args.stage_adj_dict[stage_id]
        aux_data = {"adj_mx": adj_mx, "prior": prior}
        if self.args.prompt_type == 'FlashST':
            aux_data['lpls']= self.args.stage_lpls_dict[stage_id]
            aux_data['nadj']= self.args.stage_nadj_dict[stage_id]

        node_remain_idx = []
        for i, node_id in enumerate(stage_node_info['node_order']):
            if node_id in self.first_stage_node_order:
                node_remain_idx.append(i)

        # prepare queue, optimizer, and freeze params if needed
        q = queue.Queue(maxsize=queue_size)
        opt = None
        trainable_params = []
        online_adapt = online_adapt and adapt_modules.lower()!= 'none'
        if online_adapt:
            self.model.set_finetuning_parameters(online = self.args.adapt_modules)
            for name, param in self.model.named_parameters():
                if param.requires_grad:
                    trainable_params.append(param)
                    self._logger.info(f'{name}')
            opt = torch.optim.AdamW(trainable_params, lr=adapt_lr, weight_decay=adapt_wd)

        self.model.eval()

        for it, batch in enumerate(tqdm(loader)):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            tx = batch["tx"].to(self.device)

            # inference
            with torch.no_grad():
                t1 = time.time()
                preds, _ = self.model(_build_input(x, tx, aux_data), mode="test")
                inference_time += time.time() - t1
                preds = self.scaler.inverse_transform(preds)

            outputs.append(preds)
            realy_list.append(y)
            # push into queue (batch_size=1)
            if online_adapt:
                q.put({"x": x, "y": y, "tx": tx})
            
            # online training when queue is full
            if q.full() and ((it + 1) % adapt_every == 0):
                e = q.get()
                t2 = time.time()
                self.model.train()

                xb = e["x"]
                yb = e["y"]
                txb = e["tx"]
                # print('len(q):',q.qsize,'xb.shape:',xb.shape)
                for _ in range(adapt_steps):
                    opt.zero_grad(set_to_none=True)
                    pred_b, emb = self.model(_build_input(xb, txb, aux_data), mode="train")
                    pred_b = self.scaler.inverse_transform(pred_b)

                    loss = self.loss(pred_b, yb)
                    
                    loss.backward()

                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    
                    opt.step()
                    if self.store_grad:
                        self.model.store_grad()

                self.model.eval()
                adapt_time += time.time() - t2

            
        if not os.path.exists(f'EXP_{self.args.exp_name}_results_Time'):
            os.makedirs(f'EXP_{self.args.exp_name}_results_Time')
        with open(f'EXP_{self.args.exp_name}_results_Time/{self.args.dataset_name}_efficiency.csv', 'a+') as f:
            fcsv = csv.writer(f)
            row = [self.args.expid, self.args.dataset_name, self.args.modelid, f'online_{adapt_modules}', f'stage-{stage_id}', 'param_nums-Total_Finetuning:', self.total_para_num, sum(p.numel() for p in trainable_params),'time-inference_adapt_noth:', inference_time, adapt_time, inference_time+adapt_time]
            
            fcsv.writerow(row) 
        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy_list, dim=0)


        y_valid_mask = None
        print('yhat.shape:',yhat.shape)
        print('realy.shape:', realy.shape)
        print('inference_time:', inference_time)

        if self.args.predict_length == 0:
            ready_length = self.args.input_length
        else:
            ready_length = self.args.predict_length

        
        all_metrics = []
        all_metrics_remain = []
        all_metrics_newadd = []

        filter_num = 1e-5
        feature_idx = 0
        pred_feature = yhat[..., feature_idx]
        real_feature = realy[..., feature_idx]
        if y_valid_mask is not None:
            y_valid_mask_feature = y_valid_mask[..., feature_idx]

        for i in range(ready_length):
            pred = pred_feature[:, i]
            real = real_feature[:, i]

            if y_valid_mask is not None:
                y_valid_mask_feature_ready = y_valid_mask_feature[:, i].to(pred.device)
            else:
                y_valid_mask_feature_ready = None
            metrics = metric.metric_torch(pred, real, mask_val=self.args.eval_mask, valid_mask=y_valid_mask_feature_ready, filter_num=filter_num)
            log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            self._logger.info(log.format(feature_idx, i + 1, metrics[0], metrics[1], metrics[2]))
            all_metrics.append(metrics)

            if stage_node_info is not None:
                num_remain_nodes = stage_node_info['num_visible_nodes'] - stage_node_info['num_new_nodes']
                num_newadd_nodes = stage_node_info['num_new_nodes']
                pred_subset1 = pred[:,:num_remain_nodes]
                real_subset1 = real[:,:num_remain_nodes]
                pred_subset2 = pred[:,-num_newadd_nodes:]
                real_subset2 = real[:,-num_newadd_nodes:]
                metrics_subset1 = metric.metric_torch(pred_subset1, real_subset1, mask_val=self.args.eval_mask,filter_num=filter_num)
                metrics_subset2 = metric.metric_torch(pred_subset2, real_subset2, mask_val=self.args.eval_mask,filter_num=filter_num)
                all_metrics_remain.append(metrics_subset1)
                all_metrics_newadd.append(metrics_subset2)

        amae, amape, armse, amre, ar2 = map(list, zip(*all_metrics))
        
        MAE = np.mean(amae)
        MAPE = np.mean(amape)
        RMSE = np.mean(armse)
        MRE = np.mean(amre)
        R2 = np.mean(ar2)

        log = f'[dim{feature_idx:d}] On average over {ready_length:d} horizons, Test MAE: {MAE:.4f}, Test MAPE: {MAPE:.4f}, Test RMSE: {RMSE:.4f} Test MRE: {MRE:.4f} Test R2: {R2:.4f}'
        self._logger.info(log)

        if stage_node_info is not None:
            amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1 = map(list, zip(*all_metrics_remain))
            amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2 = map(list, zip(*all_metrics_newadd))
            MAE_sub1 = np.mean(amae_sub1)
            MAPE_sub1 = np.mean(amape_sub1)
            RMSE_sub1 = np.mean(armse_sub1)
            MRE_sub1 = np.mean(amre_sub1)
            R2_sub1 = np.mean(ar2_sub1)
            MAE_sub2 = np.mean(amae_sub2)
            MAPE_sub2 = np.mean(amape_sub2)
            RMSE_sub2 = np.mean(armse_sub2)
            MRE_sub2 = np.mean(amre_sub2)
            R2_sub2 = np.mean(ar2_sub2)
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                'remain_mean':[MAE_sub1, MAPE_sub1, RMSE_sub1, MRE_sub1, R2_sub1],
                'newadd_mean':[MAE_sub2, MAPE_sub2, RMSE_sub2, MRE_sub2, R2_sub2],
                    'all_full':[amae, amape, armse, amre, ar2],
                'remain_full':[amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1],
                'newadd_full':[amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2],}, inference_time, adapt_time
        else:
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                    'all_full':[amae, amape, armse, amre, ar2]}, inference_time, adapt_time



    def test_ttc_one_stage(
        self, loader, stage_node_info, stage_id, FRP, inverse=True
    ):
        outputs = []
        realy_list = []
        inference_time = 0.0
        adapt_time = 0.0
        self.model.eval()
        # ---------- online adaptation config ----------
        online_adapt = True
        queue_size = int(getattr(self.args, "adapt_queue_size", self.args.input_length))
        adapt_steps = int(getattr(self.args, "adapt_steps", 1))
        adapt_every = int(getattr(self.args, "adapt_every", 1)) 
        adapt_lr = float(getattr(self.args, "adapt_lr", 1e-4))
        adapt_wd = float(getattr(self.args, "adapt_wd", 0.0))
        grad_clip = float(getattr(self.args, "adapt_grad_clip", 0.0))
        loss_type = getattr(self.args, "adapt_loss", "masked_mae")  # "mse" | "mae" | "huber"
        adapt_modules = self.args.adapt_modules 
        node_ids_stage = stage_node_info["node_order"]

        def _build_input(_x, _tx, _aux, batch):
            if self.args.modelid in ['DCRNN']:
                ty = batch["ty"].to(self.device)
                testx = [_x, _tx, None, ty, None, _aux]
            else:
                testx = [_x,_tx,_aux]
            return testx

        def _get_loss(pred, target):
            if loss_type == "mae":
                return F.l1_loss(pred, target)
            if loss_type == "huber":
                return F.smooth_l1_loss(pred, target)
            if loss_type == "masked_mae":
                return metric.masked_mae_torch(pred, target, mask_val=0.0)
            return F.mse_loss(pred, target)

        # stage-wise constant aux data
        prior = self.args.stage_prior_dict[stage_id].to(self.device)
        adj_mx = self.args.stage_adj_dict[stage_id]
        aux_data = {"adj_mx": adj_mx, "prior": prior}
        if self.args.prompt_type == 'FlashST':
            aux_data['lpls']= self.args.stage_lpls_dict[stage_id]
            aux_data['nadj']= self.args.stage_nadj_dict[stage_id]

        # prepare queue, optimizer, and freeze params if needed
        q = queue.Queue(maxsize=queue_size)
        opt = None
        trainable_params = []
        ok_num = 0
        gpu_usage = 0

        self.model.set_finetuning_parameters(online = 'none')
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                trainable_params.append(param)
                self._logger.info(f'{name}')
                
        FRP.ensure_nodes(node_ids_stage, device=self.device)
        keys = [str(int(n) if torch.is_tensor(n) else n) for n in node_ids_stage]

        params = []
        for k in keys:
            params += list(FRP.node_params[k].parameters())
        opt = torch.optim.AdamW(params, lr=adapt_lr, weight_decay=adapt_wd)

        self.model.eval()

        for it, batch in enumerate(tqdm(loader)):
            x = batch["x"].to(self.device)
            y = batch["y"].to(self.device)
            tx = batch["tx"].to(self.device)

            # inference
            with torch.no_grad():
                t1 = time.time()
                yb, _ = self.model(_build_input(x, tx, aux_data, batch), mode="test")
                FRP.eval()
                yc = FRP(yb, node_ids_stage)
                inference_time += time.time() - t1
                preds = self.scaler.inverse_transform(yc)

            outputs.append(preds)
            realy_list.append(y)
            # push into queue (batch_size=1)
            if online_adapt:
                q.put({"x": x, "y": y, "tx": tx})
            
            # online training when queue is full
            if q.full() and ((it + 1) % adapt_every == 0):
                e = q.get()
                t2 = time.time()
                self.model.eval()

                xb = e["x"]
                yb = e["y"]
                txb = e["tx"]
                # print('len(q):',q.qsize,'xb.shape:',xb.shape)
                for _ in range(adapt_steps):
                    opt.zero_grad(set_to_none=True)
                    with torch.no_grad():
                        yb_o, _ = self.model(_build_input(xb, txb, aux_data, batch), mode="train")
                    FRP.train()
                    yc_o = FRP(yb_o, node_ids_stage)
                    pred_b = self.scaler.inverse_transform(yc_o)
                    loss = _get_loss(pred_b, yb)
                    loss.backward()

                    if grad_clip and grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(trainable_params, grad_clip)
                    
                    opt.step()

                self.model.eval()
                adapt_time += time.time() - t2


        yhat = torch.cat(outputs, dim=0)
        realy = torch.cat(realy_list, dim=0)


        y_valid_mask = None
        print('yhat.shape:',yhat.shape)
        print('realy.shape:', realy.shape)
        print('inference_time:', inference_time)

        if self.args.predict_length == 0:
            ready_length = self.args.input_length
        else:
            ready_length = self.args.predict_length

        
        all_metrics = []
        all_metrics_remain = []
        all_metrics_newadd = []

        filter_num = 1e-5
        feature_idx = 0
        pred_feature = yhat[..., feature_idx]
        real_feature = realy[..., feature_idx]
        if y_valid_mask is not None:
            y_valid_mask_feature = y_valid_mask[..., feature_idx]

        for i in range(ready_length):
            pred = pred_feature[:, i]
            real = real_feature[:, i]

            if y_valid_mask is not None:
                y_valid_mask_feature_ready = y_valid_mask_feature[:, i].to(pred.device)
            else:
                y_valid_mask_feature_ready = None
            metrics = metric.metric_torch(pred, real, mask_val=self.args.eval_mask, valid_mask=y_valid_mask_feature_ready, filter_num=filter_num)
            log = 'Evaluate best model on test data for [dim{:d}] horizon {:d}, Test MAE: {:.4f}, Test MAPE: {:.4f}, Test RMSE: {:.4f}'
            self._logger.info(log.format(feature_idx, i + 1, metrics[0], metrics[1], metrics[2]))
            all_metrics.append(metrics)

            if stage_node_info is not None:
                num_remain_nodes = stage_node_info['num_visible_nodes'] - stage_node_info['num_new_nodes']
                num_newadd_nodes = stage_node_info['num_new_nodes']
                pred_subset1 = pred[:,:num_remain_nodes]
                real_subset1 = real[:,:num_remain_nodes]
                pred_subset2 = pred[:,-num_newadd_nodes:]
                real_subset2 = real[:,-num_newadd_nodes:]
                metrics_subset1 = metric.metric_torch(pred_subset1, real_subset1, mask_val=self.args.eval_mask,filter_num=filter_num)
                metrics_subset2 = metric.metric_torch(pred_subset2, real_subset2, mask_val=self.args.eval_mask,filter_num=filter_num)
                all_metrics_remain.append(metrics_subset1)
                all_metrics_newadd.append(metrics_subset2)

        amae, amape, armse, amre, ar2 = map(list, zip(*all_metrics))
        
        MAE = np.mean(amae)
        MAPE = np.mean(amape)
        RMSE = np.mean(armse)
        MRE = np.mean(amre)
        R2 = np.mean(ar2)

        log = f'[dim{feature_idx:d}] On average over {ready_length:d} horizons, Test MAE: {MAE:.4f}, Test MAPE: {MAPE:.4f}, Test RMSE: {RMSE:.4f} Test MRE: {MRE:.4f} Test R2: {R2:.4f}'
        self._logger.info(log)

        if stage_node_info is not None:
            amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1 = map(list, zip(*all_metrics_remain))
            amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2 = map(list, zip(*all_metrics_newadd))
            MAE_sub1 = np.mean(amae_sub1)
            MAPE_sub1 = np.mean(amape_sub1)
            RMSE_sub1 = np.mean(armse_sub1)
            MRE_sub1 = np.mean(amre_sub1)
            R2_sub1 = np.mean(ar2_sub1)
            MAE_sub2 = np.mean(amae_sub2)
            MAPE_sub2 = np.mean(amape_sub2)
            RMSE_sub2 = np.mean(armse_sub2)
            MRE_sub2 = np.mean(amre_sub2)
            R2_sub2 = np.mean(ar2_sub2)
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                'remain_mean':[MAE_sub1, MAPE_sub1, RMSE_sub1, MRE_sub1, R2_sub1],
                'newadd_mean':[MAE_sub2, MAPE_sub2, RMSE_sub2, MRE_sub2, R2_sub2],
                    'all_full':[amae, amape, armse, amre, ar2],
                'remain_full':[amae_sub1, amape_sub1, armse_sub1, amre_sub1, ar2_sub1],
                'newadd_full':[amae_sub2, amape_sub2, armse_sub2, amre_sub2, ar2_sub2],}, inference_time, adapt_time
        else:
            return {'all_mean':[MAE, MAPE, RMSE, MRE, R2],
                    'all_full':[amae, amape, armse, amre, ar2]}, inference_time, adapt_time


    def testing(self, test_stages, test_dataloader, stage_node_info=None, test_finetuning = False, test_ttc=False, exp_name='Ori'):
        
        online_tip = 'none'
        if test_ttc:
            online_tip ='_online_stttc'
            T = self.args.input_length
            M = T // 2 + 1
            groups = 4
            FRP = DynamicFRPlusModule(M, groups).to(self.device)
        elif test_finetuning:
            online_tip = '_online'

        
        stage_metric_dict = {}
        stage_infer_time_list = []
        stage_adapt_time_list = []
        
        for stage_id, loader in test_dataloader.items():
            self.model.eval()
            stage_id = str(stage_id)
            print(f'start testing {stage_id}...')
            node_info = stage_node_info[stage_id] if stage_node_info is not None else None
            
            if test_ttc:
                metric_dict, infer_time, adapt_time = self.test_ttc_one_stage(loader, node_info, stage_id, FRP=FRP)
            elif test_finetuning:
                metric_dict, infer_time, adapt_time = self.test_finetuning_one_stage(loader, node_info, stage_id)
            else:
                metric_dict, infer_time = self.test_one_stage(loader, node_info, stage_id)
                adapt_time = 0


            stage_metric_dict[stage_id] = metric_dict
            stage_infer_time_list.append(infer_time)
            stage_adapt_time_list.append(adapt_time)

        if not os.path.exists(f'./EXP_{exp_name}_results/{self.args.modelid}/'):
            os.makedirs(f'./EXP_{exp_name}_results/{self.args.modelid}/')
        if not os.path.exists(f'./EXP_{exp_name}_results_avg/{self.args.modelid}/'):
            os.makedirs(f'./EXP_{exp_name}_results_avg/{self.args.modelid}/')
            
        result_csv_path = f'./EXP_{exp_name}_results/{self.args.modelid}/{self.args.modelid}_{self.args.dataset_name}_{self.args.input_length}to{self.args.predict_length}_{self.args.division_name}_Results.csv'
        result_csv_path_avg = f'./EXP_{exp_name}_results_avg/{self.args.modelid}/{self.args.modelid}_{self.args.dataset_name}_{self.args.input_length}to{self.args.predict_length}_{self.args.division_name}_Results.csv'
       

        def write_to_csv(csv_path, Detail=True, Avg=False):
            with open(csv_path, 'a+', newline='')as f0:
                f_csv = csv.writer(f0)
                # --- before loop: collectors ---
                collect_all = []
                collect_remain = []
                collect_newadd = []

                for stage_id in test_stages:
                    row = [self.args.expid, self.args.dataset_name, self.args.note, online_tip,
                        self.args.hid_dim, 'test', f'stage-{stage_id}']

                    all_mean = stage_metric_dict[stage_id]['all_mean']
                    if Detail:
                        row.extend(['all-mae_mape_rmse_mre_r2'] + all_mean)
                    collect_all.append(all_mean)

                    if stage_node_info is not None:
                        remain_mean = stage_metric_dict[stage_id]['remain_mean']
                        newadd_mean  = stage_metric_dict[stage_id]['newadd_mean']
                        if Detail:
                            row.extend(['remain-mae_mape_rmse_mre_r2'] + remain_mean)
                            row.extend(['newadd-mae_mape_rmse_mre_r2'] + newadd_mean)
                        collect_remain.append(remain_mean)
                        collect_newadd.append(newadd_mean)

                    if Detail: 
                        f_csv.writerow(row)

                # --- after loop: write stage-avg row ---
                if Avg:
                    avg_row = [self.args.expid, self.args.dataset_name, self.args.note, online_tip,
                            self.args.hid_dim, 'test', 'stage-avg']
                    all_avg = np.mean(np.asarray(collect_all, dtype=np.float64), axis=0).tolist()
                    avg_row.extend(['all-mae_mape_rmse_mre_r2'] + all_avg)
                    if stage_node_info is not None and len(collect_remain) > 0:
                        remain_avg = np.mean(np.asarray(collect_remain, dtype=np.float64), axis=0).tolist()
                        newadd_avg = np.mean(np.asarray(collect_newadd, dtype=np.float64), axis=0).tolist()
                        avg_row.extend(['remain-mae_mape_rmse_mre_r2'] + remain_avg)
                        avg_row.extend(['newadd-mae_mape_rmse_mre_r2'] + newadd_avg)

                    f_csv.writerow(avg_row)
        
        print('result_csv_path:',result_csv_path)
        write_to_csv(result_csv_path)
        write_to_csv(result_csv_path_avg, Detail=False, Avg=True)

        self._logger.info('log and results saved.')




def get_parameter_number(model):
    total_num = sum(p.numel() for p in model.parameters())
    trainable_num = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total_num, trainable_num

