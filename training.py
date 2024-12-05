import dataio, meta_modules, utils, loss_functions, modules
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from eval_utils import get_snr, plot_train_loss_psnr_vs_epoch
from functools import partial
from copy import deepcopy
import numpy as np
import torch
import time
import os

def train_image(data_path, config):
    data_name = data_path.split('/')[-1].split('.')[0]
    img_dataset = dataio.ImageFile(data_path)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=config.image.resolution)
    image_resolution = (config.image.resolution, config.image.resolution)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.image.batch_size, pin_memory=True, num_workers=0)
    
    if config.model_type == 'sine' or config.model_type == 'relu' or config.model_type == 'tanh' or config.model_type == 'selu' or config.model_type == 'elu'\
            or config.model_type == 'softplus':
        model = modules.SingleBVPNet(type=config.model_type, mode='mlp', 
                                    sidelength=image_resolution, out_features=3, hidden_features=config.image.hidden_features, 
                                    num_hidden_layers=config.image.num_hidden_layers)
    elif config.model_type == 'rbf' or config.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode=config.model_type, sidelength=image_resolution)
    else:
        raise NotImplementedError
    model.cuda()
    root_path = os.path.join(config.logging_root, config.experiment_name)
    loss_fn = partial(loss_functions.image_mse, None)
    start = time.time()
    inr = train_INR(model=model, train_dataloader=dataloader, epochs=config.image.num_epochs, lr=config.image.lr,
                    epochs_til_print=config.image.epochs_til_print,
                    model_dir=root_path, loss_fn=loss_fn, type='image', data_name=data_name)
    end = time.time()
    print(f"{end - start: .4f} sec")
    return inr

def train_audio(data_path, config):
    data_name = data_path.split('/')[-1].split('.')[0]
    audio_dataset = dataio.AudioFile(filename=data_path)
    coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.audio.batch_size, pin_memory=True, num_workers=0)
    
    if config.model_type == 'sine' or config.model_type == 'relu' or config.model_type == 'tanh':
        model = modules.SingleBVPNet(type=config.model_type, mode='mlp', in_features=1, hidden_features=config.audio.hidden_features, num_hidden_layers=config.audio.num_hidden_layers)
    elif config.model_type == 'rbf' or config.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode=config.model_type, fn_samples=len(audio_dataset.data), in_features=1)
    else:
        raise NotImplementedError
    model.cuda()
    root_path = os.path.join(config.logging_root, config.experiment_name)
    loss_fn = loss_functions.function_mse
    start = time.time()
    inr = train_INR(model=model, train_dataloader=dataloader, epochs=config.audio.num_epochs, lr=config.audio.lr,
                    epochs_til_print=config.audio.epochs_til_print,
                    model_dir=root_path, loss_fn=loss_fn, type='audio', data_name=data_name)
    end = time.time()
    print(f"{end - start: .4f} sec")
    return inr

def train_video(data_path, config):
    data_name = data_path.split('/')[-1].split('.')[0]
    vid_dataset = dataio.Video(data_path)
    coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=config.video.sample_frac)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.video.batch_size, pin_memory=True, num_workers=0)
    
    if config.model_type == 'sine' or config.model_type == 'relu' or config.model_type == 'tanh':
        model = modules.SingleBVPNet(type=config.model_type, in_features=3, out_features=vid_dataset.channels,
                                     mode='mlp', hidden_features=config.video.hidden_features, num_hidden_layers=config.video.num_hidden_layers)
    elif config.model_type == 'rbf' or config.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', in_features=3, out_features=vid_dataset.channels, mode=config.model_type)
    else:
        raise NotImplementedError
    model.cuda()
    root_path = os.path.join(config.logging_root, config.experiment_name)
    loss_fn = partial(loss_functions.image_mse, None)
    start = time.time()
    inr = train_INR(model=model, train_dataloader=dataloader, epochs=config.video.num_epochs, lr=config.video.lr,
                    epochs_til_print=config.video.epochs_til_print,
                    model_dir=root_path, loss_fn=loss_fn, type='video', data_name=data_name)
    end = time.time()
    print(f"{end - start: .4f} sec")
    return inr

def train_sdf(data_path, config):
    data_name = data_path.split('/')[-1].split('.')[0]
    sdf_dataset = dataio.PointCloud(data_path, on_surface_points=config.sdf.batch_size)
    dataloader = DataLoader(sdf_dataset, shuffle=True, batch_size=1, pin_memory=True, num_workers=0)
    
    if config.model_type == 'nerf':
        model = modules.SingleBVPNet(type='relu', mode='nerf', in_features=3, hidden_features=config.sdf.hidden_features, num_hidden_layers=config.sdf.num_hidden_layers)
    else:
        model = modules.SingleBVPNet(type=config.model_type, in_features=3, hidden_features=config.sdf.hidden_features, num_hidden_layers=config.sdf.num_hidden_layers)
    model.cuda()
    root_path = os.path.join(config.logging_root, config.experiment_name)
    loss_fn = loss_functions.sdf
    start = time.time()
    inr = train_INR(model=model, train_dataloader=dataloader, epochs=config.sdf.num_epochs, lr=config.sdf.lr,
                    epochs_til_print=config.sdf.epochs_til_print,
                    model_dir=root_path, loss_fn=loss_fn, type='sdf', data_name=data_name)
    end = time.time()
    print(f"{end - start: .4f} sec")
    return inr

def train_INR(model, train_dataloader, epochs, lr, steps_til_print, 
              model_dir, loss_fn, val_dataloader=None, double_precision=False, clip_grad=False, use_lbfgs=False, loss_schedules=None, plot=False, type=None, data_name=None):
    utils.cond_mkdir(model_dir)
    optim = torch.optim.Adam(lr=lr, params=model.parameters())
    total_steps = 0
    with tqdm(total=len(train_dataloader) * epochs) as pbar:
        best_psnr = -10.0
        stop = 0
        epochs_list = []
        total_psnr = []
        total_train_loss = []
        for epoch in range(epochs):
            for step, (model_input, gt) in enumerate(train_dataloader):
                start_time = time.time()
            
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}

                if double_precision:
                    model_input = {key: value.double() for key, value in model_input.items()}
                    gt = {key: value.double() for key, value in gt.items()}

                if use_lbfgs:
                    def closure():
                        optim.zero_grad()
                        model_output = model(model_input)
                        losses = loss_fn(model_output, gt)
                        train_loss = 0.
                        for loss_name, loss in losses.items():
                            train_loss += loss.mean() 
                        train_loss.backward()
                        return train_loss
                    optim.step(closure)

                model_output = model(model_input)
                losses = loss_fn(model_output, gt)

                train_loss = 0.
                for loss_name, loss in losses.items():
                    single_loss = loss.mean()

                    if loss_schedules is not None and loss_name in loss_schedules:
                        single_loss *= loss_schedules[loss_name](total_steps)
                    train_loss += single_loss

                if not use_lbfgs:
                    optim.zero_grad()
                    train_loss.backward()
                    optim.step()

                pbar.update(1)

                if not total_steps % steps_til_print:
                    tqdm.write("Epoch %d, Total loss %0.6f, iteration time %0.6f" % (epoch, train_loss, time.time() - start_time))

                total_steps += 1
            if type == 'audio':
                waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
                gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
                psnr = get_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))

                if (psnr - best_psnr) > 0.00001:
                    best_psnr = psnr
                    best_model = deepcopy(model.state_dict())
                    ae = np.abs(gt_wf - waveform)
                    second_metric = round(np.mean(ae), 4)
                    stop = 0
                else:
                    stop += 1
                if stop > 1000:
                    break
            elif type == 'video':
                # psnr, apd = calculate_video_psnr(model)
                psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
                if (psnr - best_psnr) > 0.00001:
                    best_psnr = psnr
                    best_model = deepcopy(model.state_dict())
                    # try:
                    #     second_metric = round(apd.item(), 4)
                    # except:
                    #     second_metric = round(apd, 4)
                    stop = 0
                else:
                    stop += 1
                if stop > 5000:
                    break
            elif type == 'image':
                psnr = 10*torch.log10(1 / torch.mean((gt['img'] - model_output['model_out'])**2))
                if (psnr - best_psnr) > 0.00001:
                    best_psnr = psnr
                    best_model = deepcopy(model.state_dict())
                    # best_optim = deepcopy(optim.state_dict())
                    # second_metric = calculate_rmse(gt['img'], model_output['model_out'])
                    # try:
                    #     second_metric = round(second_metric, 4)
                    # except:
                    #     second_metric = round(second_metric.item(), 4)
                    stop = 0
                else:
                    stop += 1
                if stop > 10000:
                    break 
            elif type == 'sdf':
                if epoch == 15000:
                    best_model = deepcopy(model.state_dict())
                    break
            if not epoch % steps_til_print:
                epochs_list.append(epoch)
                total_psnr.append(psnr)
                total_train_loss.append(train_loss.item())
        if plot:
            plot_path = os.path.join(model_dir, f'{data_name}.png')
            plot_train_loss_psnr_vs_epoch(epochs_list, total_train_loss, total_psnr, plot_path)
        torch.save(best_model, os.path.join(model_dir, f'{data_name}.pth'))
        return best_model
        # torch.save(best_model,
        #            os.path.join(checkpoints_dir, 'model_final.pth'))
        # np.savetxt(os.path.join(checkpoints_dir, 'train_losses_final.txt'),
        #            np.array(train_losses))