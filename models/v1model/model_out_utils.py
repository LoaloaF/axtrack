import pandas as pd
import torch

def yolo_coo2tile_coo(target_tensor, Sx, Sy, tilesize, device, with_batch_dim=True):
    # nobatch dim
    # this creates a bool mask tensor that is false for all pos label entries
    box_mask = (target_tensor == 0).all(-1)

    # these rescale vectors scale depending on the gridbox the label is in
    rescale_x = torch.arange(Sx).reshape(Sx, 1, 1).to(device)
    rescale_y = torch.arange(Sy).reshape(1, Sy, 1).to(device)
    # add batch dim at position 0
    if with_batch_dim:
        rescale_x = rescale_x.unsqueeze(0)
        rescale_y = rescale_y.unsqueeze(0)
        
    # this is 200 IQ: we add the gridbox (0-S) to the within-gridbox
    # coordinate (0-1). The coordiante is now represented in terms of
    # gridboxsize (here 64)
    target_tensor[..., 1::3] = (target_tensor[..., 1::3] + rescale_x)
    target_tensor[..., 2::3] = (target_tensor[..., 2::3] + rescale_y)
    # now we just muplitply by gridboxsize to scale to image coo
    target_tensor[..., 1::3] = (target_tensor[..., 1::3] * tilesize/Sx).round()
    target_tensor[..., 2::3] = (target_tensor[..., 2::3] * tilesize/Sy).round()
    # the rescale vector also adds to the 0-elements (unwanted). Mask them here
    target_tensor[box_mask] = 0
    return target_tensor

def filter_anchors(target_tensor, conf_threshold, ceil_conf=True):
    # convert the target tensor to shape (batchsize, Sx*Sy, 3)
    batch_size, labelsize = target_tensor.shape[0], target_tensor.shape[-1]
    target_tensor = target_tensor.reshape((batch_size, -1, labelsize))
    if ceil_conf:
        target_tensor[..., 0][target_tensor[..., 0]>1] = 1
    
    # split batches into list because of variable sizes 
    frame_targets = list(torch.tensor_split(target_tensor, batch_size))
    # filter each example
    frame_targets_fltd = [tar[tar[:,:,0] >= conf_threshold] for tar in frame_targets]
    return frame_targets_fltd

def fltd_tensor2pandas(target_tensor, sort=True):
    axons_ids = target_tensor[:,3].to(int) if target_tensor.shape[1]>3 else range(target_tensor.shape[0])
    axon_lbls = [f'Axon_{i:0>3}' for i in axons_ids]
    # axon_lbls = pd.MultiIndex.from_product([[t], axon_lbls])
    cols = 'conf', 'anchor_x', 'anchor_y'
    
    target_df = pd.DataFrame(target_tensor[:,:3].to('cpu').numpy(), index=axon_lbls, 
                             columns=cols).convert_dtypes()
    if sort:
        target_df.sort_values('conf', inplace=True)
    return target_df

def Y2pandas_anchors(Y, Sx, Sy, tilesize, device, conf_thr):
    # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize)
    Y_imgcoo = yolo_coo2tile_coo(Y.clone(), Sx, Sy, tilesize, device)

    # filter the (batch, Sx, Sy, 3) output into a list of length batchsize,
    # where the list elements are tensors of shape (#_boxes_above_thr x 3)
    # each element in the list represents a frame/tile
    Ys_list_fltd = filter_anchors(Y_imgcoo, conf_threshold=conf_thr)
    # convert each frame from tensor to pandas
    for i in range(len(Ys_list_fltd)):
        Ys_list_fltd[i] = fltd_tensor2pandas(Ys_list_fltd[i])
    return Ys_list_fltd