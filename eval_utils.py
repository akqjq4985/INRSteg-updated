from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR
import matplotlib.pyplot as plt
import numpy as np
import torch
import math

def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def calculate_rmse(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    return torch.sqrt(mse)

def calculate_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

def calculate_mae(img1, img2):
    img1 = img1.detach().cpu().numpy()
    img2 = img2.detach().cpu().numpy()
    ae = np.abs(img1 - img2)
    mae = round(np.mean(ae), 4)
    return mae

def calculate_rmse(img1, img2):
    mse = torch.mean((img1 - img2)**2)
    return torch.sqrt(mse)

def calculate_apd(img1, img2):
    img1 = np.array(img1)
    img2 = np.array(img2)
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    apd = np.mean(np.abs(img1 - img2))
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

def recon_image(data_path, inr, config):
    pass
    