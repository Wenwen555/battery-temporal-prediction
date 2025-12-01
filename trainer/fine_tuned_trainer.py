import os
import sys

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import NTXentLoss
from dataloader.augmentations import DataTransform

# 定义一个MAPE loss
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        return loss


def Trainer_f(model, temporal_contr_model, model_optimizer, train_dl, test_dl, device,
            logger, config, experiment_log_dir, training_mode):
    # Start training
    if logger != None:
        logger.debug("Fine tuning started ....")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')
    if training_mode == "plot":
        predictions, labels = model_plot(model, temporal_contr_model, test_dl, device)
        return predictions, labels
    if training_mode == "plot_metric":
        mape, rmse = model_metric_plot(model, temporal_contr_model, test_dl, device)
        return mape, rmse

    if training_mode != "source_only":
        for epoch in range(1, config.num_epoch + 1):
            # Train and validate
            train_rmse_loss, train_mape_loss, train_real_rmse = model_train(model, temporal_contr_model, model_optimizer, train_dl, config, device, training_mode)
            scheduler.step(train_rmse_loss)
            if logger != None:
                logger.debug(f'\nEpoch : {epoch}\n'
                            f'Train RMSE Loss     : {train_rmse_loss:2.8f}\t | \tTrain MAPE Loss     : {train_mape_loss:2.8f}\n'
                            f'Real RMSE Loss     : {train_real_rmse:2.8f}\t\n'
                            f'lr                 : {scheduler.get_last_lr()}\n'
                )

        # save the model after training ...
        os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
        chkpoint = {'model_state_dict': model.state_dict(),
                    'temporal_contr_model_state_dict': temporal_contr_model.state_dict()}
        torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # evaluate on the test set
    if logger != None:
        logger.debug('\nEvaluate on the Test set:')
        test_mape, test_rmse = model_evaluate(model, temporal_contr_model, test_dl, device)
        logger.debug(f'Test MAPE loss      :{test_mape:2.8f}\t | Test RMSE loss      : {test_rmse:2.8f}')
        logger.debug("\n################## Training is Done! #########################")
        return (test_mape, test_rmse)



def model_train(model, temporal_contr_model, model_optimizer, train_loader, config,
                device, training_mode):
    model.train()
    #此时并不训练TC模块，只是利用其功能来计算
    
    total_loss_mse = []
    total_loss_mape = []
    total_loss_rmse = []
    total_loss_real_rmse = []
    total_acc = []
    criterion_1 = nn.MSELoss()
    criterion_2 = MAPELoss()
    
    for batch_idx, (data, labels) in enumerate(train_loader):
        # send to device
        cycle_data =data.float().to(device)
        cycle_labels = labels.float().to(device)
        # optimizer
        model_optimizer.zero_grad()
        
        # calculation
        predictions, features = model(cycle_data)
        aug1, aug2 = DataTransform(features, config)
        
        temp_cont_loss1, _ = temporal_contr_model(aug1, features)
        temp_cont_loss2, _ = temporal_contr_model(aug2, features)
        cont_loss = temp_cont_loss1 + temp_cont_loss2

        loss_mse_supervised = criterion_1(predictions, cycle_labels)
        loss_mape_supervised = criterion_2(predictions, cycle_labels)
        loss_rmse_supervised = torch.sqrt(loss_mse_supervised)
    
        loss_rmse = loss_rmse_supervised/loss_rmse_supervised.detach() + 0.5 * cont_loss/cont_loss.detach()
        loss_mape = loss_mape_supervised

        total_loss_rmse.append(loss_rmse.item())
        total_loss_real_rmse.append(loss_rmse_supervised.item())
        total_loss_mape.append(loss_mape.item())
        loss_rmse.backward()
        # 不backword loss_mape,只将其作为检测指标考虑.
        # loss_mape.backward()
        model_optimizer.step()

    total_loss_rmse = torch.tensor(total_loss_rmse).mean()
    total_loss_mape = torch.tensor(total_loss_mape).mean()
    total_loss_real_rmse = torch.tensor(total_loss_real_rmse).mean()
    return total_loss_rmse, total_loss_mape, total_loss_real_rmse


def model_evaluate(model, temporal_contr_model ,test_dl, device):
    model.eval()
    temporal_contr_model.eval()

    total_loss_mape = []
    total_loss_mse = []
    total_loss_rmse = []
    
    criterion_1 = MAPELoss()
    criterion_2 = nn.MSELoss()

    with torch.no_grad():
        for data, labels in test_dl:
            cycle_data,cycle_labels = data.float().to(device), labels.float().to(device)
            predictions, _ = model(cycle_data)
            # predictions, _ = temporal_contr_model(features)
            loss_mape = criterion_1(predictions, cycle_labels)
            loss_mse = criterion_2(predictions, cycle_labels)
            loss_rmse = torch.sqrt(loss_mse)
            total_loss_mape.append(loss_mape.item())
            total_loss_rmse.append(loss_rmse.item())
        total_loss_mape = torch.tensor(total_loss_mape).mean()  # average loss
        total_loss_rmse = torch.tensor(total_loss_rmse).mean()
    return total_loss_mape, total_loss_rmse
    

def model_metric_plot(model, temporal_contr_model, test_dl, device='cuda'):
    model.eval()
    temporal_contr_model.eval()

    total_loss_mape = []
    total_loss_mse = []
    total_loss_rmse = []
    
    criterion_1 = MAPELoss()
    criterion_2 = nn.MSELoss()

    with torch.no_grad():
        for data, labels in test_dl:
            cycle_data,cycle_labels = data.float().to(device), labels.float().to(device)
            predictions, _ = model(cycle_data)
            # predictions, _ = temporal_contr_model(features)
            loss_mape = criterion_1(predictions, cycle_labels)
            loss_mse = criterion_2(predictions, cycle_labels)
            loss_rmse = torch.sqrt(loss_mse)
            total_loss_mape.append(loss_mape.item())
            total_loss_rmse.append(loss_rmse.item())
    return total_loss_mape, total_loss_rmse

def model_plot(model, temporal_contr_model, test_dl, device='cuda'):
    model.eval()
    temporal_contr_model.eval()

    total_predictions = []
    total_labels = []

    with torch.no_grad():
        for data, labels in test_dl:
            cycle_data,cycle_labels = data.float().to(device), labels.float().to(device)
            predictions, _ = model(cycle_data)
            
            for idx in range(len(predictions)):
                total_predictions.append(predictions[idx])
                total_labels.append(cycle_labels[idx])
    tensor_predictions = torch.stack([window for window in total_predictions])
    tensor_labels = torch.stack([window for window in total_labels])
    return tensor_predictions, tensor_labels
