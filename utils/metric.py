import numpy as np
import torch
import math

def huber_loss(preds, labels, delta=1.0):
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    return torch.mean(torch.where(condition, small_res, large_res))


def masked_huber_loss(preds, labels, delta=1.0, null_val=np.nan, mask_val=np.nan, compare_flag='ne', valid_mask=None,filter_num=1e-5):
    # labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):        
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    residual = torch.abs(preds - labels)
    condition = torch.le(residual, delta)
    small_res = 0.5 * torch.square(residual)
    large_res = delta * residual - 0.5 * delta * delta
    loss = torch.where(condition, small_res, large_res)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def mae_torch(preds,labels):
    loss = torch.mean(torch.abs(preds - labels))
    return loss
def mse_torch(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss
def rmse_torch(preds,labels):
    loss = torch.mean(torch.square(preds - labels))
    return loss**0.5



def masked_mae_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne', reduce=True, valid_mask = None, filter_num=1e-5):
    if filter_num is not None:
        labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    if valid_mask is not None:
        mask &= valid_mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    if reduce:
        return torch.mean(loss)
    else:
        return loss


def masked_mape_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne', valid_mask=None, filter_num=1e-5):
    if filter_num is not None: 
        labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):        
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    if valid_mask is not None:
        mask &= valid_mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs((preds - labels) / labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_mse_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne',valid_mask=None,filter_num=1e-5):
    if filter_num is not None: labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):        
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    if valid_mask is not None:
        mask &= valid_mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.square(torch.sub(preds, labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne', valid_mask=None,filter_num=1e-5):
    if filter_num is not None: labels[torch.abs(labels) < filter_num] = 0
    return torch.sqrt(masked_mse_torch(preds=preds, labels=labels,
                                       null_val=null_val, mask_val=mask_val, compare_flag=compare_flag, valid_mask=valid_mask))

def masked_mre_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne',valid_mask=None,filter_num=1e-5):
    if filter_num is not None: labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    if valid_mask is not None:
        mask &= valid_mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.sum(torch.abs(preds - labels)) / torch.sum(torch.abs(labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    # print(loss.shape)
    return torch.mean(loss)


def masked_r2_torch(preds, labels, null_val=np.nan, mask_val=np.nan, compare_flag='ne',valid_mask=None,filter_num=1e-5):
    if filter_num is not None: labels[torch.abs(labels) < filter_num] = 0
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = labels.ne(null_val)
    if not np.isnan(mask_val):
        if compare_flag == 'gt':
            mask &= labels.gt(mask_val)
        elif compare_flag == 'ge':
            mask &= labels.ge(mask_val)
        elif compare_flag == 'ne':
            mask &= labels.ne(mask_val)
    if valid_mask is not None:
        mask &= valid_mask
    mask = mask.float()
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = 1. - torch.sum(torch.square(labels - preds)) / torch.sum(torch.square(torch.mean(labels, dim=0) - labels))
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric_torch(predict, real, mask_val=0, compare_flag='ne', valid_mask=None,filter_num=1e-5):
    nmae = masked_mae_torch(predict, real, mask_val = 0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    nrmse = masked_rmse_torch(predict, real, mask_val = 0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    mape = masked_mape_torch(predict, real, mask_val = 0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    mre = masked_mre_torch(predict, real, mask_val = 0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    r2 = masked_r2_torch(predict, real, mask_val = 0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    return nmae, mape, nrmse, mre, r2

def mae_np(preds,labels):
    loss = np.mean(np.abs(preds - labels))
    return loss
def mse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss
def rmse_np(preds,labels):
    loss = np.mean(np.square(preds - labels))
    return loss**0.5

def masked_r2_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan, compare_flag='ne', valid_mask=None,filter_num=1e-5):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            if compare_flag == 'gt':
                mask &= np.greater(y_true, mask_val)
            elif compare_flag == 'ge':
                mask &= np.greater_equal(y_true, mask_val)
            elif compare_flag == 'ne':
                mask &= np.not_equal(y_true, mask_val)
        if valid_mask is not None:
            mask &= valid_mask
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        r2 = 1. - (np.square(y_true - y_pred)).sum() / (np.square(y_true.mean(0) - y_true)).sum()
        r2 = np.nan_to_num(mask * r2)
        return np.mean(r2)


def masked_mape_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan ,compare_flag='ne', valid_mask=None,filter_num=1e-5):
    if filter_num is not None:
        y_true[y_true<filter_num] = 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            if compare_flag == 'gt':
                mask &= np.greater(y_true, mask_val)
            elif compare_flag == 'ge':
                mask &= np.greater_equal(y_true, mask_val)
            elif compare_flag == 'ne':
                mask &= np.not_equal(y_true, mask_val)
        if valid_mask is not None:
            mask &= valid_mask
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mape = np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'),
                      y_true))
        mape = np.nan_to_num(mask * mape)
        return np.mean(mape)

def masked_mre_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan ,compare_flag='ne', valid_mask=None,filter_num=1e-5):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            if compare_flag == 'gt':
                mask &= np.greater(y_true, mask_val)
            elif compare_flag == 'ge':
                mask &= np.greater_equal(y_true, mask_val)
            elif compare_flag == 'ne':
                mask &= np.not_equal(y_true, mask_val)
        if valid_mask is not None:
            mask &= valid_mask
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mre = np.divide(np.sum(np.abs(np.abs(y_pred - y_true).astype('float32'))),
                      np.sum(np.abs(y_true)))
        mre = np.nan_to_num(mask * mre)
        return np.mean(mre)

def masked_mae_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan ,compare_flag='ne', valid_mask=None,filter_num=1e-5):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            if compare_flag == 'gt':
                mask &= np.greater(y_true, mask_val)
            elif compare_flag == 'ge':
                mask &= np.greater_equal(y_true, mask_val)
            elif compare_flag == 'ne':
                mask &= np.not_equal(y_true, mask_val)
        if valid_mask is not None:
            mask &= valid_mask
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(y_pred, y_true).astype('float32'))
        mae = np.nan_to_num(mask * mae)
        return np.mean(mae)

def masked_mse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan ,compare_flag='ne', valid_mask=None,filter_num=1e-5):
    with np.errstate(divide='ignore', invalid='ignore'):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        if not np.isnan(mask_val):
            if compare_flag == 'gt':
                mask &= np.greater(y_true, mask_val)
            elif compare_flag == 'ge':
                mask &= np.greater_equal(y_true, mask_val)
            elif compare_flag == 'ne':
                mask &= np.not_equal(y_true, mask_val)
        if valid_mask is not None:
            mask &= valid_mask
        mask = mask.astype('float32')
        mask /= np.mean(mask)
        mse = np.square(np.subtract(y_pred, y_true).astype('float32'))
        mse = np.nan_to_num(mask * mse)
        return np.mean(mse)

def masked_rmse_np(y_pred, y_true, null_val=np.nan, mask_val=np.nan ,compare_flag='ne', valid_mask=None,filter_num=1e-5):
    return np.sqrt(masked_mse_np(y_pred=y_pred, y_true=y_true,
                                       null_val=null_val, mask_val=mask_val, compare_flag=compare_flag, valid_mask=valid_mask))


def masked_metric_np(predict, real, mask_val=-1, compare_flag ='ne', valid_mask=None,filter_num=1e-5):
    print('in metric_np MASK:',mask_val)
    nmae = masked_mae_np(predict, real, mask_val=0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    nrmse = masked_rmse_np(predict, real, mask_val=0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    mape = masked_mape_np(predict, real, mask_val=0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    mre = masked_mre_np(predict, real, mask_val=0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    r2 = masked_r2_np(predict, real, mask_val=0.0, compare_flag=compare_flag, valid_mask=valid_mask,filter_num=filter_num).item()
    return nmae, mape, nrmse, mre, r2

