import os
from utils import load_config, get_type_config, get_files_in_folder
from training import train_image, train_audio, train_video, train_sdf
import pdb

if __name__ == "__main__":
    config = load_config("./config.yaml")
    secret_data_folder = "./data/secret"
    cover_data_folder = "./data/cover"
    
    # Train secret data
    secret_data_list = get_files_in_folder(secret_data_folder)
    cover_data_list = get_files_in_folder(cover_data_folder)
    secret_INRs = []

    for secret_data in secret_data_list:
        data_type = secret_data["type"]
        data_path = secret_data["path"]
        print(f"Transforming data, {data_path}")
        if data_type == "image":
            inr = train_image(data_path, config)
        elif data_type == "audio":
            inr = train_audio(data_path, config)
        elif data_type == "video":
            inr = train_video(data_path, config)
        elif data_type == "sdf":
            inr = train_sdf(data_path, config)
        else:
            raise ValueError(f"Data type {data_type} not supported")
        secret_INRs.append(inr)

    # Insert secret data into cover data
    