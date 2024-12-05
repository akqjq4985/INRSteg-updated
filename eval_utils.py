from torchmetrics.audio import ScaleInvariantSignalNoiseRatio as SNR
import matplotlib.pyplot as plt
import os

def get_snr(gt, pred):
    snr = SNR()
    return snr(gt, pred)

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