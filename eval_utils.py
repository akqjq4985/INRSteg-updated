from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR
from torchvision.utils import make_grid
from torch.utils.data import DataLoader
import scipy.io.wavfile as wavfile
import matplotlib.pyplot as plt
from einops import rearrange
import dataio, sdf_meshing
import skvideo.datasets
import numpy as np
import torch
import math
import os
import pdb

def calculate_psnr(data1, data2):
    # data1 and data2 have range [0, 255]
    data1 = data1.astype(np.float64)
    data2 = data2.astype(np.float64)
    mse = np.mean((data1 - data2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_rmse(data1, data2):
    mse = torch.mean((data1 - data2)**2)
    rmse = torch.sqrt(mse)
    return rmse*255

def calculate_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

def calculate_mae(data1, data2):
    data1 = data1.detach().cpu().numpy()
    data2 = data2.detach().cpu().numpy()
    ae = np.abs(data1 - data2)
    mae = round(np.mean(ae), 4)
    return mae*255

def calculate_apd(data1, data2):
    data1 = np.array(data1)
    data2 = np.array(data2)
    data1 = data1.astype(np.float64)
    data2 = data2.astype(np.float64)
    apd = np.mean(np.abs(data1 - data2))
    if apd == 0:
        return float('inf')
    return np.mean(apd)

def plot_train_loss_psnr_vs_epoch(epochs_list, total_train_loss, total_psnr, plot_path):
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.plot(epochs_list, total_train_loss)
    plt.subplot(1,2,2)
    plt.xlabel("Epoch")
    plt.ylabel("PSNR")
    plt.plot(epochs_list, total_psnr)
    plt.savefig(plot_path)

def inr2img(inr, resolution=256):
    mgrid = dataio.get_mgrid(resolution)
    coord_dataset = {'idx': 0, 'coords': mgrid}
    with torch.no_grad():
        model_output = inr.forward_with_activations(coord_dataset)
    out = model_output['model_out'][1]
    out = rearrange(out, '(h w) c -> c h w', h=resolution, w=resolution)
    return out

def recon_image(data_path, inr, config, save_gt=False):
    data_name = "image_" + os.path.splitext(data_path.split('/')[-1])[0]
    log_dir = os.path.join(config.logging_root, config.experiment_name)
    if not os.path.exists(f'{log_dir}/recon/'):
        os.makedirs(f'{log_dir}/recon/')
    img_dataset = dataio.ImageFile(data_path)
    coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=config.image.resolution)
    _, gt_dict = coord_dataset[0]
    gt_img = gt_dict['img']
    gt_img = rearrange(gt_img, '(h w) c -> c h w', h=256, w=256)
    if save_gt:
        plt.figure()
        plt.imsave(f'{log_dir}/recon/GT_{data_name}.png', gt_img.permute(1,2,0))
        plt.close()
    img = inr2img(inr, resolution=config.image.resolution)
    plt.figure()
    plt.imsave(f'{log_dir}/recon/{data_name}.png', img.permute(1,2,0))
    plt.close()
    psnr = 10*torch.log10(1 / torch.mean((gt_img - img)**2))
    rmse = calculate_rmse(gt_img, img)
    print("secret image PSNR: ", psnr)
    print("secret image RMSE: ", rmse)
    with open(f'{log_dir}/recon/{data_name}_eval_results.txt', 'w') as f:
        f.write(f"PSNR: {psnr.item()}\n")
        f.write(f"RMSE: {rmse.item()}\n")

def recon_audio(data_path, inr, config, save_gt=False):
    data_name = "audio_" + os.path.splitext(data_path.split('/')[-1])[0]
    log_dir = os.path.join(config.logging_root, config.experiment_name)
    if not os.path.exists(f'{log_dir}/recon/'):
        os.makedirs(f'{log_dir}/recon/')
    audio_dataset = dataio.AudioFile(data_path)
    coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
    dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.audio.batch_size, pin_memory=True, num_workers=0)
    model_input, gt = next(iter(dataloader))
    model_input = {key: value.cuda() for key, value in model_input.items()}
    gt = {key: value.cuda() for key, value in gt.items()}
    with torch.no_grad():
        model_output = inr(model_input)
    waveform = torch.squeeze(model_output['model_out']).detach().cpu().numpy()
    rate = torch.squeeze(gt['rate']).detach().cpu().numpy()
    gt_wf = torch.squeeze(gt['func']).detach().cpu().numpy()
    if save_gt:
        wavfile.write(f'{log_dir}/recon/GT_{data_name}.wav', rate, gt_wf)
    wavfile.write(f'{log_dir}/recon/{data_name}.wav', rate, waveform)
    mae = calculate_mae(torch.Tensor(gt_wf), torch.Tensor(waveform))
    snr = calculate_snr(torch.Tensor(gt_wf), torch.Tensor(waveform))
    print("secret audio mae ", mae)
    print("secret audio snr", snr)
    with open(f'{log_dir}/recon/{data_name}_eval_results.txt', 'w') as f:
        f.write(f"MAE: {mae}\n")
        f.write(f"SNR: {snr.item()}\n")

def recon_video(data_path, inr, config, save_gt=False):
    data_name = "video_" + os.path.splitext(data_path.split('/')[-1])[0]
    log_dir = os.path.join(config.logging_root, config.experiment_name)
    if not os.path.exists(f'{log_dir}/recon/'):
        os.makedirs(f'{log_dir}/recon/')
    vid_dataset = dataio.Video(data_path)
    resolution = vid_dataset.shape
    frames = [0, 4, 8, 12]
    Nslice = 10
    with torch.no_grad():
        coords = [dataio.get_mgrid((1, resolution[1], resolution[2]), dim=3)[None,...].cuda() for f in frames]
        for idx, f in enumerate(frames):
            coords[idx][..., 0] = (f / (resolution[0] - 1) - 0.5) * 2
        coords = torch.cat(coords, dim=0)
        output = torch.zeros(coords.shape)
        split = int(coords.shape[1] / Nslice)
        for i in range(Nslice):
            inr = inr.cuda()
            coords = coords.cuda()
            pred = inr({'coords':coords[:, i*split:(i+1)*split, :]})['model_out']
            output[:, i*split:(i+1)*split, :] =  pred.cpu()
    pred_vid = output.view(len(frames), resolution[1], resolution[2], 3) / 2 + 0.5
    pred_vid = torch.clamp(pred_vid, 0, 1)
    gt_vid = torch.from_numpy(vid_dataset.vid[frames, :, :, :])
    pred_vid = pred_vid.permute(0, 3, 1, 2)
    gt_vid = gt_vid.permute(0, 3, 1, 2)
    if save_gt:
        output_vs_gt = torch.cat((gt_vid, pred_vid), dim=-2)
        out = make_grid(output_vs_gt, scale_each=False, normalize=True)
        plt.figure()
        plt.imsave(f'{log_dir}/recon/{data_name}.png', out.permute(1,2,0).cpu().numpy())
        plt.close()
    else:
        out = make_grid(pred_vid, scale_each=False, normalize=True)
        plt.figure()
        plt.imsave(f'{log_dir}/recon/{data_name}.png', out.permute(1,2,0).cpu().numpy())
        plt.close()
    psnr = 10*torch.log10(1 / torch.mean((gt_vid - pred_vid)**2))
    apd = calculate_apd(gt_vid, pred_vid)
    print("secret video psnr", psnr)
    print("secret video apd", apd)
    with open(f'{log_dir}/recon/{data_name}_eval_results.txt', 'w') as f:
        f.write(f"PSNR: {psnr.item()}\n")
        f.write(f"APD: {apd}\n")

def recon_sdf(data_path, inr, config, save_gt=False):
    log_dir = os.path.join(config.logging_root, config.experiment_name)
    if not os.path.exists(f'{log_dir}/recon/'):
        os.makedirs(f'{log_dir}/recon/')
    class SDFDecoder:
        def __init__(self, inr):
            super().__init__()
            self.inr = inr
            self.inr.cuda()
        def forward(self, coords):
            model_in = {'coords': coords}
            return self.inr(model_in)['model_out']
    sdf_decoder = SDFDecoder(inr)
    sdf_meshing.create_mesh(sdf_decoder, f'{log_dir}/recon/{config.experiment_name}', N=512)