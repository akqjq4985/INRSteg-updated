from training import train_image, train_audio, train_video, train_sdf
from torch.utils.data import DataLoader
from eval_utils import recon_image, recon_audio, recon_video, recon_sdf
import modules
import dataio
import torch
import copy
import pdb
import os

type_dict = {'image': (2,3), 'audio': (1,1), 'sdf': (3,1), 'video': (3,3)}

def assign_secret_inr_size(secret_data_list, cover_data_list, cover_inr_size):
    num_secrets = len(secret_data_list)
    cover_input_dim, cover_output_dim = type_dict[cover_data_list[0]["type"]]
    size_per_secret = cover_inr_size["hidden_features"] // num_secrets
    remaining_features = cover_inr_size["hidden_features"] % num_secrets
    
    for i, secret_data in enumerate(secret_data_list):
        secret_type = secret_data["type"]
        input_dim, output_dim = type_dict[secret_type]
        secret_inr_size = {"hidden_features": size_per_secret, "num_hidden_layers": cover_inr_size["num_hidden_layers"]}
        if input_dim > cover_input_dim:
            secret_inr_size["num_hidden_layers"] -= 1
        if output_dim > cover_output_dim:
            secret_inr_size["num_hidden_layers"] -= 1
        if num_secrets == 1:
            secret_inr_size["hidden_features"] = cover_inr_size["hidden_features"] // 2
        else:
            if i == (num_secrets - 1):
                secret_inr_size["hidden_features"] += remaining_features
        secret_data["inr_size"] = secret_inr_size
    return secret_data_list

def insert_single_inr(secret_inr, cover_inr, pad):
    secret_inr_hidden_features = secret_inr["net.net.0.0.weight"].shape[0]
    secret_inr_num_hidden_layers = (len(secret_inr) // 2) - 2
    secret_input_dim = secret_inr["net.net.0.0.weight"].shape[1]
    secret_output_dim = secret_inr[f"net.net.{secret_inr_num_hidden_layers+1}.0.weight"].shape[0]
    hidden_features = cover_inr.net.net[0][0].weight.shape[0]
    num_hidden_layers = len(cover_inr.net.net) - 2
    input_dim = cover_inr.net.net[0][0].weight.shape[1]
    output_dim = cover_inr.net.net[num_hidden_layers+1][0].weight.shape[0]
    mask = {}
    for i in range(num_hidden_layers+2):
        layer_name = f"net.net.{i}.0.weight"
        org_weight = cover_inr.net.net[i][0].weight.data
        mask[layer_name] = torch.ones_like(org_weight).cuda()
        if i == 0:
            if input_dim < secret_input_dim:
                layer_name = f"net.net.{i}.0.bias"
                org_bias = cover_inr.net.net[i][0].bias.data
                mask[layer_name] = torch.ones_like(org_bias).cuda()
                continue
        if input_dim < secret_input_dim:
            secret_weight = secret_inr[f"net.net.{i-1}.0.weight"]
        else:
            secret_weight = secret_inr[f"net.net.{i}.0.weight"]
        if i == 0:
            org_weight[pad:pad+secret_inr_hidden_features, :secret_input_dim] = 0
            mask[layer_name][pad:pad+secret_inr_hidden_features, :secret_input_dim] = 0
            m = torch.nn.ZeroPad2d((0,(input_dim-secret_input_dim), pad, hidden_features-secret_inr_hidden_features-pad))
        elif i == 1:
            if input_dim < secret_input_dim:
                org_weight[pad:pad+secret_inr_hidden_features, :secret_input_dim] = 0
                mask[layer_name][pad:pad+secret_inr_hidden_features, :secret_input_dim] = 0
                m = torch.nn.ZeroPad2d((0,(hidden_features-secret_input_dim), pad, hidden_features-secret_inr_hidden_features-pad))
            else:
                org_weight[pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ZeroPad2d((pad, hidden_features-secret_inr_hidden_features-pad, pad, hidden_features-secret_inr_hidden_features-pad))
        elif i == num_hidden_layers:
            if output_dim < secret_output_dim:
                org_weight[:secret_output_dim, pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][:secret_output_dim, pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ZeroPad2d((pad, hidden_features-secret_inr_hidden_features-pad, 0, (output_dim-secret_output_dim)))
            else:
                org_weight[pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ZeroPad2d((pad, hidden_features-secret_inr_hidden_features-pad, pad, hidden_features-secret_inr_hidden_features-pad))
        elif i == (num_hidden_layers+1):
            if output_dim < secret_output_dim:
                continue
            else:
                org_weight[:secret_output_dim, pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][:secret_output_dim, pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ZeroPad2d((pad, hidden_features-secret_inr_hidden_features-pad, 0, (output_dim-secret_output_dim)))
        else:
            org_weight[pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
            mask[layer_name][pad:pad+secret_inr_hidden_features, pad:pad+secret_inr_hidden_features] = 0
            m = torch.nn.ZeroPad2d((pad, hidden_features-secret_inr_hidden_features-pad, pad, hidden_features-secret_inr_hidden_features-pad))
        secret_weight = m(secret_weight)
        new_weight = org_weight.cuda() + secret_weight.cuda()
        cover_inr.net.net[i][0].weight.data = new_weight
        
        layer_name = f"net.net.{i}.0.bias"
        org_bias = cover_inr.net.net[i][0].bias.data
        mask[layer_name] = torch.ones_like(org_bias).cuda()
        if input_dim < secret_input_dim:
            secret_bias = secret_inr[f"net.net.{i-1}.0.bias"]
        else:
            secret_bias = secret_inr[f"net.net.{i}.0.bias"]
        if output_dim < secret_output_dim:
            if i != (num_hidden_layers):
                org_bias[pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ConstantPad1d((pad, hidden_features-secret_inr_hidden_features-pad),0)
            elif i == (num_hidden_layers+1):
                continue
            else:
                m = torch.nn.ConstantPad1d((0,(output_dim-secret_output_dim)), 0.0)
                org_bias[:secret_output_dim] = 0
                mask[layer_name][:secret_output_dim] = 0
        else:
            if i != (num_hidden_layers+1):
                org_bias[pad:pad+secret_inr_hidden_features] = 0
                mask[layer_name][pad:pad+secret_inr_hidden_features] = 0
                m = torch.nn.ConstantPad1d((pad, hidden_features-secret_inr_hidden_features-pad),0)
            else:
                m = torch.nn.ConstantPad1d((0,(output_dim-secret_output_dim)), 0.0)
                org_bias[:secret_output_dim] = 0
                mask[layer_name][:secret_output_dim] = 0
        secret_bias = m(secret_bias)
        new_bias = org_bias.cuda() + secret_bias.cuda()    
        cover_inr.net.net[i][0].bias.data = new_bias
    return cover_inr, mask

def insert_inr(secret_inrs, cover_data, cover_inr_size, config):
    hidden_features = cover_inr_size["hidden_features"]
    num_hidden_layers = cover_inr_size["num_hidden_layers"]
    if cover_data["type"] == 'image':
        cover_inr = modules.SingleBVPNet(type=config.model_type, mode='mlp', 
                                   sidelength=config.image.resolution, out_features=3, hidden_features=hidden_features,
                                   num_hidden_layers=num_hidden_layers)
    elif cover_data["type"] == 'audio':
        cover_inr = modules.SingleBVPNet(type=config.model_type, mode='mlp', in_features=1, hidden_features=hidden_features, 
                                   num_hidden_layers=num_hidden_layers)
    elif cover_data["type"] == 'video':
        cover_inr = modules.SingleBVPNet(type=config.model_type, in_features=3, out_features=3, 
                                   mode='mlp', hidden_features=hidden_features, 
                                   num_hidden_layers=num_hidden_layers)
    elif cover_data["type"] == 'sdf':
        cover_inr = modules.SingleBVPNet(type=config.model_type, in_features=3, hidden_features=hidden_features, 
                                   num_hidden_layers=num_hidden_layers)
    else:
        raise ValueError(f"Data type {cover_data['type']} not supported")
    
    if len(secret_inrs) == 1:
        secret_inr_hidden_features = secret_inrs[0]["net.net.0.0.weight"].shape[0]
        pad = (hidden_features - secret_inr_hidden_features) // 2
        cover_inr, mask = insert_single_inr(secret_inrs[0], cover_inr, pad)
        return cover_inr, mask, [mask]
    else:
        pad = 0
        accumulated_mask = {}
        mask_per_data = []
        for secret_inr in secret_inrs:
            cover_inr, mask = insert_single_inr(secret_inr, cover_inr, pad)
            pad += secret_inr["net.net.0.0.weight"].shape[0]
            if len(accumulated_mask) == 0:
                accumulated_mask = copy.deepcopy(mask)
            else:
                for key in mask.keys():
                    accumulated_mask[key] = accumulated_mask[key].cuda() * mask[key].cuda()
            mask_per_data.append(mask)
        return cover_inr, accumulated_mask, mask_per_data

def permute(cover_inr, mask, mask_per_data, key):
    torch.manual_seed(key)
    torch.cuda.manual_seed(key)
    torch.cuda.manual_seed_all(key)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    permute_dict = {}
    num_hidden_layers = len(cover_inr.net.net) - 2
    for i in range(1, num_hidden_layers+1):
        vec = cover_inr.net.net[i][0].weight
        perm_vec = torch.randperm(vec.shape[0])
        perm_matrix = torch.eye(vec.shape[0])[perm_vec].T
        permute_dict[i] = perm_matrix.cuda()

    for name, param in cover_inr.named_parameters():
        layer_index = int(name.split('.')[2])
        if layer_index == (num_hidden_layers):
            break
        P = permute_dict[layer_index+1]
        if 'weight' in name:
            cover_inr.net.net[layer_index][0].weight.data = P.mm(param)
            cover_inr.net.net[layer_index+1][0].weight.data = cover_inr.net.net[layer_index+1][0].weight.data.mm(P.T)
            mask[name] = P.mm(mask[name])
            next_name = f'net.net.{layer_index+1}.0.weight'
            mask[next_name] = mask[next_name].mm(P.T)
            for i in range(len(mask_per_data)):
                mask_per_data[i][name] = P.mm(mask_per_data[i][name])
                mask_per_data[i][next_name] = mask_per_data[i][next_name].mm(P.T)
        elif 'bias' in name:
            cover_inr.net.net[layer_index][0].bias.data = P.matmul(param)
            mask[name] = P.matmul(mask[name])
            for i in range(len(mask_per_data)):
                mask_per_data[i][name] = P.matmul(mask_per_data[i][name])
    return cover_inr, mask, mask_per_data

def checked_upchanged_permute(cover_inr, cover_data_list, config, mask, mask_per_data, key):
    data_path = cover_data_list[0]["path"]
    data_type = cover_data_list[0]["type"]
    if data_type == 'image':
        img_dataset = dataio.ImageFile(data_path)
        coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=config.image.resolution)
        image_resolution = (config.image.resolution, config.image.resolution)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.image.batch_size, pin_memory=True, num_workers=0)
    elif data_type == 'audio':
        audio_dataset = dataio.AudioFile(data_path)
        coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.audio.batch_size, pin_memory=True, num_workers=0)
    elif data_type == 'video':
        vid_dataset = dataio.Video(data_path)
        coord_dataset = dataio.Implicit3DWrapper(vid_dataset)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.video.batch_size, pin_memory=True, num_workers=0)
    elif data_type == 'sdf':
        sdf_dataset = dataio.SDFFile(data_path)
        coord_dataset = dataio.Implicit3DWrapper(sdf_dataset)
        dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.sdf.batch_size, pin_memory=True, num_workers=0)
    with torch.no_grad():
        model_input, gt = next(iter(dataloader))
        model_input = {key: value.cuda() for key, value in model_input.items()}
        gt = {key: value.cuda() for key, value in gt.items()}
        model_output = cover_inr(model_input)
    cover_inr, mask, mask_per_data = permute(cover_inr, mask, mask_per_data, key)
    with torch.no_grad():
        model_output_after_permute = cover_inr(model_input)
    if not torch.all(model_output['model_out'] == model_output_after_permute['model_out']):
        ValueError("Model output has changed after permutation")

def hide(secret_inrs, secret_data_list, cover_data_list, cover_inr_size, config, key=None):
    cover_inr, mask, mask_per_data = insert_inr(secret_inrs, cover_data_list[0], cover_inr_size, config)
    if key is not None:
        checked_upchanged_permute(cover_inr, cover_data_list, config, mask, mask_per_data, key)
        permuted_cover_inr, mask, mask_per_data = permute(cover_inr, mask, mask_per_data, key)
    data_path = cover_data_list[0]["path"]
    inr_size = cover_inr_size
    if cover_data_list[0]["type"] == 'image':
        cover_inr = train_image(data_path, inr_size, config, mask=mask, model=cover_inr)
    elif cover_data_list[0]["type"] == 'audio':
        cover_inr = train_audio(data_path, inr_size, config, mask=mask, model=cover_inr)
    elif cover_data_list[0]["type"] == 'video':
        cover_inr = train_video(data_path, inr_size, config, mask=mask, model=cover_inr)
    elif cover_data_list[0]["type"] == 'sdf':
        cover_inr = train_sdf(data_path, inr_size, config, mask=mask, model=cover_inr)
    else:
        raise ValueError(f"Data type {cover_data_list[0]['type']} not supported")
    return cover_inr, mask_per_data

def reveal(cover_inr, mask_per_data, secret_data_list, config):
    secret_inrs = []
    for i, mask in enumerate(mask_per_data):
        reverse_mask = {key: 1 - mask[key] for key in mask}
        data_type = secret_data_list[i]["type"]
        hidden_features = secret_data_list[i]["inr_size"]["hidden_features"]
        num_hidden_layers = secret_data_list[i]["inr_size"]["num_hidden_layers"]
        if data_type == 'image':
            secret_inr = modules.SingleBVPNet(
                type=config.model_type, mode='mlp', out_features=3, hidden_features=hidden_features, in_features=2, 
                sidelength=config.image.resolution, num_hidden_layers=num_hidden_layers
            )
        elif data_type == 'audio':
            secret_inr = modules.SingleBVPNet(
                type=config.model_type, mode='mlp', out_features=1, hidden_features=hidden_features, in_features=1, 
                num_hidden_layers=num_hidden_layers
            )
        elif data_type == 'video':
            secret_inr = modules.SingleBVPNet(
                type=config.model_type, mode='mlp', out_features=3, hidden_features=hidden_features, in_features=3, 
                num_hidden_layers=num_hidden_layers
            )
        elif data_type == 'sdf':
            secret_inr = modules.SingleBVPNet(
                type=config.model_type, mode='mlp', out_features=1, hidden_features=hidden_features, in_features=3, 
                num_hidden_layers=num_hidden_layers
            )
        else:
            raise ValueError(f"Data type {data_type} not supported")
        secret_layer_index = 0
        for name, param in cover_inr.items():
            layer_index = int(name.split('.')[2])
            if name in reverse_mask:
                if reverse_mask[name].sum() == 0:
                    continue
                if 'weight' in name:
                    mask_indices_x = (reverse_mask[f'net.net.{layer_index}.0.weight'] != 0).any(dim=1).nonzero(as_tuple=True)[0]
                    mask_indices_y = (reverse_mask[f'net.net.{layer_index}.0.weight'] != 0).any(dim=0).nonzero(as_tuple=True)[0]
                    secret_weight = param[mask_indices_x]
                    secret_weight = secret_weight[:,mask_indices_y]
                    secret_inr.net.net[secret_layer_index][0].weight.data = secret_weight.clone()
                elif 'bias' in name:
                    mask_indices = (reverse_mask[f'net.net.{layer_index}.0.bias'] != 0).nonzero(as_tuple=True)[0]
                    secret_bias = param[mask_indices]
                    secret_inr.net.net[secret_layer_index][0].bias.data = secret_bias.clone()
                    secret_layer_index += 1
        secret_inrs.append(secret_inr)
        del secret_inr
    return secret_inrs

def check_unchanged_secret_data(secret_inrs, secret_data_list, config, encode=False):
    log_dir = os.path.join(config.logging_root, config.experiment_name)
    if encode:
        for i, secret_data in enumerate(secret_data_list):
            data_path = secret_data["path"]
            data_type = secret_data["type"]
            inr_size = secret_data["inr_size"]
            if data_type == 'image':
                data_name = "image_" + os.path.splitext(data_path.split('/')[-1])[0]
                img_dataset = dataio.ImageFile(data_path)
                coord_dataset = dataio.Implicit2DWrapper(img_dataset, sidelength=config.image.resolution)
                image_resolution = (config.image.resolution, config.image.resolution)
                dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.image.batch_size, pin_memory=True, num_workers=0)
            elif data_type == 'audio':
                data_name = "audio_" + os.path.splitext(data_path.split('/')[-1])[0]
                audio_dataset = dataio.AudioFile(data_path)
                coord_dataset = dataio.ImplicitAudioWrapper(audio_dataset)
                dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.audio.batch_size, pin_memory=True, num_workers=0)
            elif data_type == 'video':
                data_name = "video_" + os.path.splitext(data_path.split('/')[-1])[0]
                vid_dataset = dataio.Video(data_path)
                coord_dataset = dataio.Implicit3DWrapper(vid_dataset, sidelength=vid_dataset.shape, sample_fraction=config.video.sample_frac)
                dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.video.batch_size, pin_memory=True, num_workers=0)
            elif data_type == 'sdf':
                data_name = "sdf_" + os.path.splitext(data_path.split('/')[-1])[0]
                sdf_dataset = dataio.SDFFile(data_path)
                coord_dataset = dataio.Implicit3DWrapper(sdf_dataset)
                dataloader = DataLoader(coord_dataset, shuffle=True, batch_size=config.sdf.batch_size, pin_memory=True, num_workers=0)
            checkpoint = torch.load(f'{log_dir}/{data_name}.pth')
            if data_type == 'image':
                secret_inr_before_encoding = modules.SingleBVPNet(type=config.model_type, mode='mlp', 
                                            sidelength=image_resolution, out_features=3, hidden_features=inr_size["hidden_features"], 
                                            num_hidden_layers=inr_size["num_hidden_layers"])
            elif data_type == 'audio':
                secret_inr_before_encoding = modules.SingleBVPNet(type=config.model_type, mode='mlp', in_features=1, 
                                            hidden_features=inr_size["hidden_features"], num_hidden_layers=inr_size["num_hidden_layers"])
            elif data_type == 'video':
                secret_inr_before_encoding = modules.SingleBVPNet(type=config.model_type, mode='mlp', in_features=3, 
                                            out_features=3, hidden_features=inr_size["hidden_features"], num_hidden_layers=inr_size["num_hidden_layers"])
            elif data_type == 'sdf':
                secret_inr_before_encoding = modules.SingleBVPNet(type=config.model_type, mode='mlp', in_features=3, 
                                            hidden_features=inr_size["hidden_features"], num_hidden_layers=inr_size["num_hidden_layers"])
            secret_inr_before_encoding.load_state_dict(checkpoint)
            secret_inr_before_encoding = secret_inr_before_encoding.cuda()
            with torch.no_grad():
                model_input, gt = next(iter(dataloader))
                model_input = {key: value.cuda() for key, value in model_input.items()}
                gt = {key: value.cuda() for key, value in gt.items()}
                model_output = secret_inr_before_encoding(model_input)
            secret_inr_after_encoding = secret_inrs[i]
            with torch.no_grad():
                model_output_after_encoding = secret_inr_after_encoding(model_input)
            if not torch.all(model_output['model_out'] == model_output_after_encoding['model_out']):
                ValueError("Model output has changed after permutation")
    else:
        for i, secret_data in enumerate(secret_data_list):
            data_path = secret_data["path"]
            data_type = secret_data["type"]
            if data_type == 'image':
                data_name = "image_" + os.path.splitext(data_path.split('/')[-1])[0]
            elif data_type == 'audio':
                data_name = "audio_" + os.path.splitext(data_path.split('/')[-1])[0]
            elif data_type == 'video':
                data_name = "video_" + os.path.splitext(data_path.split('/')[-1])[0]
            elif data_type == 'sdf':
                data_name = "sdf_" + os.path.splitext(data_path.split('/')[-1])[0]
            checkpoint = torch.load(f'{log_dir}/{data_name}.pth')
            secret_inr = secret_inrs[i]
            changed = 0
            for name, param in checkpoint.items():
                layer_index = int(name.split('.')[2])
                if 'weight' in name:
                    if not torch.all(param == secret_inr.net.net[layer_index][0].weight.data):
                        changed += 1
                elif 'bias' in name:
                    if not torch.all(param == secret_inr.net.net[layer_index][0].bias.data):
                        changed += 1
            if changed > 0:
                raise ValueError(f"Data {data_name} has changed")

def reconstruct(secret_inrs, secret_data_list, config):
    for i, secret_data in enumerate(secret_data_list):
        data_type = secret_data["type"]
        data_path = secret_data["path"]
        if data_type == 'image':
            recon_image(data_path, secret_inrs[i], config)
        elif data_type == 'audio':
            recon_audio(data_path, secret_inrs[i], config)
        elif data_type == 'video':
            recon_video(data_path, secret_inrs[i], config)
        elif data_type == 'sdf':
            recon_sdf(data_path, secret_inrs[i], config)
        else:
            raise ValueError(f"Data type {data_type} not supported")
    