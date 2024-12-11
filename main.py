import os
from utils import load_config, get_type_config, get_files_in_folder
from training import train_image, train_audio, train_video, train_sdf
from steganography import assign_secret_inr_size, hide, reveal, reconstruct
import pdb

if __name__ == "__main__":
    config = load_config("./config.yaml")

    # Load secret and cover data
    secret_data_folder = "./data/secret"
    cover_data_folder = "./data/cover"
    secret_data_list = get_files_in_folder(secret_data_folder)
    cover_data_list = get_files_in_folder(cover_data_folder)

    # Select INR size for cover
    cover_inr_size = {"hidden_features": 256, "num_hidden_layers": 6}
    secret_data_list = assign_secret_inr_size(secret_data_list, cover_data_list, cover_inr_size)

    # Train secret data
    secret_inrs = []
    for secret_data in secret_data_list:
        data_type = secret_data["type"]
        data_path = secret_data["path"]
        inr_size = secret_data["inr_size"]
        print(f"Transforming data, {data_path}")
        if data_type == "image":
            inr = train_image(data_path, inr_size, config)
        elif data_type == "audio":
            inr = train_audio(data_path, inr_size, config)
        elif data_type == "video":
            inr = train_video(data_path, inr_size, config)
        elif data_type == "sdf":
            inr = train_sdf(data_path, inr_size, config)
        else:
            raise ValueError(f"Data type {data_type} not supported")
        secret_inrs.append(inr)

    # Insert secret data and train cover INR
    cover_inr, mask_per_data = hide(secret_inrs, secret_data_list, cover_data_list, cover_inr_size, config)
    
    # Evaluate secret and cover
        # Extract secret inrs
    secret_inrs = reveal(cover_inr, mask_per_data, secret_data_list, config)
        # Reconstruct secret data
    reconstruct(secret_inrs, secret_data_list, config)