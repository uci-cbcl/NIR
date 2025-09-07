'''Implements a generic training loop.
'''

from ast import Gt
import re
import torch
from torch.optim import lr_scheduler
import util
from torch.utils.tensorboard import SummaryWriter
from tqdm.autonotebook import tqdm
import time
import numpy as np
import os
import shutil
from loss_functions import *


def train_pairwise(deform_model, moving_model, fixed_model,
                   train_dataloader, epochs, lr, steps_til_summary, 
                   epochs_til_checkpoint, model_dir, summary_fn, 
                   val_dataloader=None, double_precision=False, clip_grad=False, 
                   loss_schedules=None, loss_types=['image_mse'], loss_weight=10,  maxlen=[256, 256, 256]):

    optim = torch.optim.Adam(lr=lr, params=deform_model.deform_model.parameters())
    lr_scheduler = LinearDecaySchedule(lr, 0.1*lr, 2000)

    if os.path.exists(model_dir):
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    util.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    util.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=epochs) as pbar:
        train_losses = []
        while total_steps < epochs:

            adjust_learning_rate(lr_scheduler, optim, total_steps)

            if not total_steps % epochs_til_checkpoint and total_steps:
                torch.save(deform_model.state_dict(),
                            os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % total_steps))
                np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % total_steps),
                            np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                #TODO
                # gt['img'] = fixed_model(model_input)['model_out'] 

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                reg_output = deform_model(model_input, moving_model)

                losses = {}
                if 'image_mse' in loss_types:
                    mse_loss = image_mse(None, reg_output, gt)
                    losses = {**losses, **mse_loss}
                if 'l1' in loss_types:
                    l1_loss = image_l1(None, reg_output, gt)
                    losses = {**losses, **l1_loss}
                if 'ncc' in loss_types:
                    ncc_loss = ncc(reg_output, gt)
                    losses = {**losses, **ncc_loss}
                if 'discrete_jacobian' in loss_types:
                    discrete_jacobian_reg = discrete_jacobian_det(reg_output, gt, maxlen, loss_weight)
                    losses = {**losses, **discrete_jacobian_reg}

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                        single_loss *= loss_schedules[loss_name](total_steps)

                    # writer.add_scalar(loss_name, single_loss, total_steps)
                    train_loss += single_loss

                train_losses.append(train_loss.item())

                optim.zero_grad()
                train_loss.backward()

                if clip_grad:
                    if isinstance(clip_grad, bool):
                        torch.nn.utils.clip_grad_norm_(deform_model.deform_model.parameters(), max_norm=1.)
                    else:
                        torch.nn.utils.clip_grad_norm_(deform_model.deform_model.parameters(), max_norm=clip_grad)

                optim.step()

                pbar.update(1)

                if steps_til_summary:
                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))
                else:
                    if not total_steps % 100:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))

                # writer.add_scalar("total_train_loss", train_loss, total_steps)
                if steps_til_summary:
                    if not total_steps % steps_til_summary:
                        torch.save(deform_model.state_dict(),
                                os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(deform_model, moving_model, writer, total_steps)

                total_steps += 1

        torch.save(deform_model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def train_efficient_pairwise(deform_model, moving_model, fixed_model,
                   train_dataloader, epochs, lr, steps_til_summary, 
                   epochs_til_checkpoint, model_dir, summary_fn, 
                   val_dataloader=None, loss_schedules=None, loss_types=['image_mse'], 
                   loss_weight=10,  maxlen=[256, 256, 256]):

    scaler = torch.cuda.amp.GradScaler()

    optim = torch.optim.Adam(lr=lr, params=deform_model.deform_model.parameters())
    # lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(optim, max_lr=lr, steps_per_epoch=len(train_dataloader), epochs=epochs)
    lr_scheduler = LinearDecaySchedule(lr, 0.1*lr, 2000)

    if os.path.exists(model_dir):
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    util.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    util.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=epochs) as pbar:
        train_losses = []
        while total_steps < epochs:

            adjust_learning_rate(lr_scheduler, optim, total_steps)

            # if not total_steps % epochs_til_checkpoint and total_steps:
            #     torch.save(deform_model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % total_steps))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % total_steps),
            #                np.array(train_losses))

            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                #TODO
                # gt['img'] = fixed_model(model_input)['model_out'] 

                with torch.cuda.amp.autocast(enabled=True):
                    reg_output = deform_model(model_input, moving_model)

                    losses = {}
                    if 'image_mse' in loss_types:
                        mse_loss = image_mse(None, reg_output, gt)
                        losses = {**losses, **mse_loss}
                    if 'l1' in loss_types:
                        l1_loss = image_l1(None, reg_output, gt)
                        losses = {**losses, **l1_loss}
                    if 'ncc' in loss_types:
                        ncc_loss = ncc(reg_output, gt)
                        losses = {**losses, **ncc_loss}
                    if 'discrete_jacobian' in loss_types:
                        discrete_jacobian_reg = discrete_jacobian_det(reg_output, gt, maxlen, loss_weight)
                        losses = {**losses, **discrete_jacobian_reg}

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        # writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps)
                optim.zero_grad()
                # with torch.cuda.amp.autocast():
                scaler.scale(train_loss).backward()
                # Unscales gradients and calls
                scaler.step(optim)
                # Updates the scale for next iteration
                scaler.update()

                # lr_scheduler.step()

                if steps_til_summary:
                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))
                else:
                    if not total_steps % 100:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))

                # if steps_til_summary:
                #     if not total_steps % steps_til_summary:
                #         torch.save(deform_model.state_dict(),
                #                 os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(deform_model, moving_model, writer, total_steps)

                pbar.update(1)

                total_steps += 1

        torch.save(deform_model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def train_efficient_mss_pairwise(deform_model, moving_model, fixed_model,
                   train_dataloader, epochs, lr, steps_til_summary, 
                   epochs_til_checkpoint, model_dir, summary_fn, 
                   val_dataloader=None, loss_schedules=None, loss_types=['image_mse'], 
                   loss_weight=10, maxlen=[256, 256, 256]):

    scaler = torch.cuda.amp.GradScaler()

    optim = torch.optim.Adam(lr=lr, params=deform_model.deform_model_1.parameters())
    lr_scheduler = LinearDecaySchedule(lr, 0.1*lr, 2000)

    if os.path.exists(model_dir):
        # val = input("The model directory %s exists. Overwrite? (y/n)"%model_dir)
        # if val == 'y':
        #     shutil.rmtree(model_dir)
        shutil.rmtree(model_dir)

    os.makedirs(model_dir)

    summaries_dir = os.path.join(model_dir, 'summaries')
    util.cond_mkdir(summaries_dir)

    checkpoints_dir = os.path.join(model_dir, 'checkpoints')
    util.cond_mkdir(checkpoints_dir)

    writer = SummaryWriter(summaries_dir)

    total_steps = 0
    with tqdm(total=epochs) as pbar:
        train_losses = []
        while total_steps < epochs:

            adjust_learning_rate(lr_scheduler, optim, total_steps)

            # if not total_steps % epochs_til_checkpoint and total_steps:
            #     torch.save(deform_model.state_dict(),
            #                os.path.join(checkpoints_dir, 'model_epoch_%04d.pth' % total_steps))
            #     np.savetxt(os.path.join(checkpoints_dir, 'train_losses_epoch_%04d.txt' % total_steps),
            #                np.array(train_losses))

            for step, (model_input_1, gt_1) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input_1 = {key: value.cuda() for key, value in model_input_1.items()}
                gt_1 = {key: value.cuda() for key, value in gt_1.items()}

                #TODO
                # gt['img'] = fixed_model(model_input)['model_out'] 

                with torch.cuda.amp.autocast(enabled=True):
                    reg_output_1 = deform_model.training_forward(model_input_1, moving_model)

                    losses = {}
                    if 'image_mse' in loss_types:
                        mse_loss = image_mse(None, reg_output_1, gt_1)
                        losses = {**losses, **mse_loss}
                    if 'l1' in loss_types:
                        l1_loss = image_l1(None, reg_output_1, gt_1)
                        losses = {**losses, **l1_loss}
                    if 'ncc' in loss_types:
                        ncc_loss = ncc(reg_output_1, gt_1)
                        losses = {**losses, **ncc_loss}
                    if 'discrete_jacobian' in loss_types:
                        discrete_jacobian_reg = discrete_jacobian_det(reg_output_1, gt_1, maxlen, loss_weight)
                        losses = {**losses, **discrete_jacobian_reg}

                    train_loss = 0.
                    for loss_name, loss in losses.items():
                        single_loss = loss.mean()

                        if loss_schedules is not None and loss_name in loss_schedules:
                            # writer.add_scalar(loss_name + "_weight", loss_schedules[loss_name](total_steps), total_steps)
                            single_loss *= loss_schedules[loss_name](total_steps)

                        # writer.add_scalar(loss_name, single_loss, total_steps)
                        train_loss += single_loss

                train_losses.append(train_loss.item())
                # writer.add_scalar("total_train_loss", train_loss, total_steps)
                optim.zero_grad()
                # with torch.cuda.amp.autocast():
                scaler.scale(train_loss).backward()
                # Unscales gradients and calls
                scaler.step(optim)
                # Updates the scale for next iteration
                scaler.update()

                if steps_til_summary:
                    if not total_steps % steps_til_summary:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))
                else:
                    if not total_steps % 100:
                        tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (total_steps, train_loss, time.time() - start_time))

                # if steps_til_summary:
                #     if not total_steps % steps_til_summary:
                #         torch.save(deform_model.state_dict(),
                #                 os.path.join(checkpoints_dir, 'model_current.pth'))
                        # summary_fn(deform_model, moving_model, writer, total_steps)

                pbar.update(1)

                total_steps += 1

        torch.save(deform_model.state_dict(),
                   os.path.join(checkpoints_dir, 'model_final.pth'))
        np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
                   np.array(train_losses))


def train_pairwise_test_time(deform_model, moving_model, 
                            train_dataloader, epochs, lr, 
                            loss_types=['image_mse'], maxlen=[256, 256, 256]):

    optim = torch.optim.Adam(lr=lr, params=deform_model.deform_model.parameters())
    lr_scheduler = LinearDecaySchedule(lr, 0.1*lr, 2000)

    test_time = []
    start_time = time.time()
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        
        for epoch in range(epochs):

            adjust_learning_rate(lr_scheduler, optim, epoch)

            for step, (model_input, gt) in enumerate(train_dataloader):
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                reg_output = deform_model(model_input, moving_model)

                losses = {}
                if 'image_mse' in loss_types:
                    mse_loss = image_mse(None, reg_output, gt)
                    losses = {**losses, **mse_loss}
                if 'l1' in loss_types:
                    l1_loss = image_l1(None, reg_output, gt)
                    losses = {**losses, **l1_loss}
                if 'ncc' in loss_types:
                    ncc_loss = ncc(reg_output, gt)
                    losses = {**losses, **ncc_loss}
                if 'ssim' in loss_types:
                    ssim_loss = ssim(reg_output, gt)
                    losses = {**losses, **ssim_loss}
                if 'jacobian' in loss_types:
                    jacobian_reg = jacobian_det(reg_output)
                    losses = {**losses, **jacobian_reg}
                if 'discrete_jacobian' in loss_types:
                    discrete_jacobian_reg = discrete_jacobian_det(reg_output, gt, maxlen)
                    losses = {**losses, **discrete_jacobian_reg}
                if 'pointwise_reg' in loss_types:
                    pw_loss = pointwise_reg(reg_output)
                    losses = {**losses, **pw_loss}

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    train_loss += single_loss

                optim.zero_grad()
                train_loss.backward()

                optim.step()
                # test_time.append(end_time - start_time)

                pbar.update(1)

    
    end_time = time.time()
    print(end_time-start_time)

    # print(np.mean(test_time))


class LinearDecaySchedule():
    def __init__(self, start_val, final_val, num_steps):
        self.start_val = start_val
        self.final_val = final_val
        self.num_steps = num_steps

    def __call__(self, iter):
        return self.start_val + (self.final_val - self.start_val) * min(iter / self.num_steps, 1.)

def adjust_learning_rate(lr_schedules, optimizer, epoch):
    for i, param_group in enumerate(optimizer.param_groups):
        param_group["lr"] = lr_schedules(epoch)