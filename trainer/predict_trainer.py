import os
import sys

from mpmath.tests.test_hp import log_a

sys.path.append("..")
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.loss import NTXentLoss

# 定义一个MAPE loss
class MAPELoss(nn.Module):
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, y_pred, y_true):
        loss = torch.mean(torch.abs((y_true - y_pred) / y_true)) * 100
        return loss


def predict_Trainer(model, model_optimizer,temporal_contr_model, temporal_contr_optimizer,
                    encoder,encoder_optimizer, train_dl, valid_dl, test_dl, device,
            logger, config, experiment_log_dir):
    # Start training
    logger.debug("Training started ....")

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(model_optimizer, 'min')

    for epoch in range(1, config.num_epoch + 1):
        # Train and validate
        train_mse_loss, train_mape_loss = model_train(model, model_optimizer, temporal_contr_model,temporal_contr_optimizer,
                                                      encoder,encoder_optimizer,train_dl, config, device)
        valid_loss_mape, valid_loss_mse = model_evaluate(model,temporal_contr_model,encoder, valid_dl, device)
        scheduler.step(valid_loss_mse)

        logger.debug(f'\nEpoch : {epoch}\n'
                     f'Train MSE Loss     : {train_mse_loss:2.8f}\t | \tTrain MAPE Loss     : {train_mape_loss:2.8f}\n'
                     f'Valid MSE Loss     : {valid_loss_mse:2.8f}\t | \tValid MAPE Loss     : {valid_loss_mape:2.8f}\n'
                     f'lr                 : {scheduler.get_last_lr()}\n'
        )


    # save the model after training ...
    os.makedirs(os.path.join(experiment_log_dir, "saved_models"), exist_ok=True)
    chkpoint = {'predict_model_state_dict': model.state_dict()}
    torch.save(chkpoint, os.path.join(experiment_log_dir, "saved_models", f'ckp_last.pt'))

    # evaluate on the test set
    logger.debug('\nEvaluate on the Test set:')
    test_loss, test_acc = model_evaluate(model,temporal_contr_model,encoder,test_dl,device)
    logger.debug(f'Test MAPE loss      :{test_loss:2.8f}\t | Test MSE loss      : {test_acc:2.8f}')

    logger.debug("\n################## Training is Done! #########################")



def model_train(model, model_optimizer,temporal_contr_model,temporal_contr_optimizer,encoder,encoder_optimizer,train_loader, config,
                device):
    model.train()
    encoder.train()
    temporal_contr_model.train()
    total_loss_mse = []
    total_loss_mape = []
    total_acc = []
    criterion_1 = nn.MSELoss()
    criterion_2 = MAPELoss()

    for batch_idx, (data, labels, aug1, aug2) in enumerate(train_loader):
        # send to device
        data, labels = data.float().to(device), labels.float().to(device)

        # optimizer
        model_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        temporal_contr_optimizer.zero_grad()
        # todo: 个人怀疑此处训练效果不佳的原因是features normalized,让它返回的权重受到影响.
        _, features = encoder(data)
        # normalize projection feature vectors

        predictions,_ = model(features.reshape(128,-1))
        temp_cont_loss1, temp_cont_feat1 = temporal_contr_model(F.normalize(features, dim=1), F.normalize(features, dim=1))
        temp_cont_loss2, temp_cont_feat2 = temporal_contr_model(F.normalize(features, dim=1), F.normalize(features, dim=1))

        lambda1 = 1
        lambda2 = 0.7
        nt_xent_criterion = NTXentLoss(device, config.batch_size, config.Context_Cont.temperature,
                                       config.Context_Cont.use_cosine_similarity)
        loss_con = (temp_cont_loss1 + temp_cont_loss2) * lambda1 + \
                   nt_xent_criterion(temp_cont_feat1, temp_cont_feat2) * lambda2
        loss_mse_supervised = criterion_1(predictions, labels.view(-1, 1))
        loss_mape_supervised = criterion_2(predictions, labels.view(-1, 1))
        loss_mse = loss_mse_supervised/loss_mse_supervised.detach() + loss_con/(loss_con/loss_mse_supervised).detach()
        loss_mape = loss_mape_supervised

        total_loss_mse.append(loss_mse.item())
        total_loss_mape.append(loss_mape.item())
        loss_mse.backward()
        # 不backword loss_mape,只将其作为检测指标考虑.
        # loss_mape.backward()
        model_optimizer.step()
        temporal_contr_optimizer.step()
        encoder_optimizer.step()

    total_loss_mse = torch.tensor(total_loss_mse).mean()
    total_loss_mape = torch.tensor(total_loss_mape).mean()

    return total_loss_mse, total_loss_mape


def model_evaluate(model,temporal_contr_model, encoder ,test_dl, device):
    model.eval()
    encoder.eval()
    temporal_contr_model.eval()
    total_loss_mape = []
    total_loss_mse = []

    criterion_1 = MAPELoss()
    criterion_2 = nn.MSELoss()

    with torch.no_grad():
        for data, labels, _, _ in test_dl:
            data, labels = data.float().to(device), labels.float().to(device)
            _,features = encoder(data)
            # features = F.normalize(features, dim=1)
            # compute loss
            # predictions, _ = output
            predictions,_ = model(features.reshape(128,-1))
            loss_mape = criterion_1(predictions, labels.view(-1, 1))
            loss_mse = criterion_2(predictions, labels.view(-1, 1))
            total_loss_mape.append(loss_mape.item())
            total_loss_mse.append(loss_mse.item())

    total_loss_mape = torch.tensor(total_loss_mape).mean()  # average loss
    total_loss_mse = torch.tensor(total_loss_mse).mean()
    return total_loss_mape, total_loss_mse
