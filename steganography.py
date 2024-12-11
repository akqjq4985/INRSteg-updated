from training import train_image, train_audio, train_video, train_sdf
from eval_utils import recon_image, recon_audio, recon_video, recon_sdf
import modules
import torch
import copy
import pdb

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

# def pad_inr(secret_inr, cover_data, cover_inr_size, config): ##size1 is the number of hidden features for the hidden data, size2 is the padded size/2
#     hidden_features = cover_inr_size["hidden_features"]
#     num_hidden_layers = cover_inr_size["num_hidden_layers"]
#     secret_inr_hidden_features = secret_inr["net.net.0.0.weight"].shape[0]
#     secret_inr_num_hidden_layers = (len(secret_inr) // 2) - 2
#     pad = (hidden_features - secret_inr_hidden_features) // 2
#     mask = {}
    
#     input_dim, output_dim = type_dict[cover_data["type"]]
#     secret_input_dim = secret_inr["net.net.0.0.weight"].shape[1]
#     secret_output_dim = secret_inr[f"net.net.{secret_inr_num_hidden_layers+1}.0.weight"].shape[0]
#     for i in range(num_hidden_layers+2):
#         layer_name = f"net.net.{i}.0.weight"
#         org_weight = inr.net.net[i][0].weight.data
#         mask[layer_name] = torch.ones_like(org_weight)
#         if i == 0:
#             if input_dim < secret_input_dim:
#                 continue
#         if input_dim < secret_input_dim:
#             secret_weight = secret_inr[f"net.net.{i-1}.0.weight"]
#         else:
#             secret_weight = secret_inr[f"net.net.{i}.0.weight"]
#         if i == 0:
#             org_weight[pad:-pad, :secret_input_dim] = 0
#             mask[layer_name][pad:-pad, :secret_input_dim] = 0
#             m = torch.nn.ZeroPad2d((0,(input_dim-secret_input_dim),pad,pad))
#         elif i == 1:
#             if input_dim < secret_input_dim:
#                 org_weight[pad:-pad, :secret_input_dim] = 0
#                 mask[layer_name][pad:-pad, :secret_input_dim] = 0
#                 m = torch.nn.ZeroPad2d((0,(hidden_features-secret_input_dim),pad,pad))
#             else:
#                 org_weight[pad:-pad, pad:-pad] = 0
#                 mask[layer_name][pad:-pad, pad:-pad] = 0
#                 m = torch.nn.ZeroPad2d((pad,pad,pad,pad))
#         elif i == num_hidden_layers:
#             if output_dim < secret_output_dim:
#                 org_weight[:secret_output_dim, pad:-pad] = 0
#                 mask[layer_name][:secret_output_dim, pad:-pad] = 0
#                 m = torch.nn.ZeroPad2d((pad,pad,0,(output_dim-secret_output_dim)))
#             else:
#                 org_weight[pad:-pad, pad:-pad] = 0
#                 mask[layer_name][pad:-pad, pad:-pad] = 0
#                 m = torch.nn.ZeroPad2d((pad,pad,pad,pad))
#         elif i == (num_hidden_layers+1):
#             if output_dim < secret_output_dim:
#                 continue
#             else:
#                 org_weight[:secret_output_dim, pad:-pad] = 0
#                 mask[layer_name][:secret_output_dim, pad:-pad] = 0
#                 m = torch.nn.ZeroPad2d((pad,pad,0,(output_dim-secret_output_dim)))
#         else:
#             org_weight[pad:-pad, pad:-pad] = 0
#             mask[layer_name][pad:-pad, pad:-pad] = 0
#             m = torch.nn.ZeroPad2d((pad,pad,pad,pad))
#         secret_weight = m(secret_weight)
#         new_weight = org_weight.cuda() + secret_weight.cuda()
#         inr.net.net[i][0].weight.data = new_weight
        
#         layer_name = f"net.net.{i}.0.bias"
#         org_bias = inr.net.net[i][0].bias.data
#         mask[layer_name] = torch.ones_like(org_bias)
#         if input_dim < secret_input_dim:
#             secret_bias = secret_inr[f"net.net.{i-1}.0.bias"]
#         else:
#             secret_bias = secret_inr[f"net.net.{i}.0.bias"]
#         if output_dim < secret_output_dim:
#             if i != (num_hidden_layers):
#                 org_bias[pad:-pad] = 0
#                 mask[layer_name][pad:-pad] = 0
#                 m = torch.nn.ConstantPad1d((pad,pad),0)
#             elif i == (num_hidden_layers+1):
#                 continue
#             else:
#                 m = torch.nn.ConstantPad1d((0,(output_dim-secret_output_dim)), 0.0)
#                 org_bias[:secret_output_dim] = 0
#                 mask[layer_name][:secret_output_dim] = 0
#         else:
#             if i != (num_hidden_layers+1):
#                 org_bias[pad:-pad] = 0
#                 mask[layer_name][pad:-pad] = 0
#                 m = torch.nn.ConstantPad1d((pad,pad),0)
#             else:
#                 m = torch.nn.ConstantPad1d((0,(output_dim-secret_output_dim)), 0.0)
#                 org_bias[:secret_output_dim] = 0
#                 mask[layer_name][:secret_output_dim] = 0
#         secret_bias = m(secret_bias)
#         new_bias = org_bias.cuda() + secret_bias.cuda()    
#         inr.net.net[i][0].bias.data = new_bias
#     return inr, mask

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
        mask[layer_name] = torch.ones_like(org_weight)
        if i == 0:
            if input_dim < secret_input_dim:
                layer_name = f"net.net.{i}.0.bias"
                org_bias = cover_inr.net.net[i][0].bias.data
                mask[layer_name] = torch.ones_like(org_bias)
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
        mask[layer_name] = torch.ones_like(org_bias)
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

def hide(secret_inrs, secret_data_list, cover_data_list, cover_inr_size, config):
    cover_inr, mask, mask_per_data = insert_inr(secret_inrs, cover_data_list[0], cover_inr_size, config)

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
                    secret_weight = param.cuda() * reverse_mask[name].cuda()
                    secret_inr.net.net[secret_layer_index][0].weight.data = secret_weight.clone()
                elif 'bias' in name:
                    secret_bias = param.cuda() * reverse_mask[name].cuda()
                    secret_inr.net.net[secret_layer_index][0].bias.data = secret_bias.clone()
                    secret_layer_index += 1
        secret_inrs.append(secret_inr)
    return secret_inrs

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
    