# import pandas as pd
# import numpy as np
# import torch

# def yolo_coo2tile_coo(target_tensor, Sx, Sy, tilesize, device, with_batch_dim=True):
#     # nobatch dim
#     # this creates a bool mask tensor that is false for all pos label entries
#     box_mask = (target_tensor == 0).all(-1)

#     # these rescale vectors scale depending on the gridbox the label is in
#     rescale_x = torch.arange(Sx).reshape(Sx, 1, 1).to(device)
#     rescale_y = torch.arange(Sy).reshape(1, Sy, 1).to(device)
#     # add batch dim at position 0
#     if with_batch_dim:
#         rescale_x = rescale_x.unsqueeze(0)
#         rescale_y = rescale_y.unsqueeze(0)
        
#     # this is 200 IQ: we add the gridbox (0-S) to the within-gridbox
#     # coordinate (0-1). The coordiante is now represented in terms of
#     # gridboxsize (here 64)
#     target_tensor[..., 1::3] = (target_tensor[..., 1::3] + rescale_x)
#     target_tensor[..., 2::3] = (target_tensor[..., 2::3] + rescale_y)
#     # now we just muplitply by gridboxsize to scale to image coo
#     target_tensor[..., 1::3] = (target_tensor[..., 1::3] * tilesize/Sx).round()
#     target_tensor[..., 2::3] = (target_tensor[..., 2::3] * tilesize/Sy).round()
#     # the rescale vector also adds to the 0-elements (unwanted). Mask them here
#     target_tensor[box_mask] = 0
#     return target_tensor

# def filter_anchors(target_tensor, conf_threshold):
#     # convert the target tensor to shape (batchsize, Sx*Sy, 3)
#     batch_size, labelsize = target_tensor.shape[0], target_tensor.shape[-1]
#     target_tensor = target_tensor.reshape((batch_size, -1, labelsize))
    
#     # split batches into list because of variable sizes 
#     frame_targets = list(torch.tensor_split(target_tensor, batch_size))
#     # filter each example
#     frame_targets_fltd = [tar[tar[:,:,0] >= conf_threshold] for tar in frame_targets]
#     return frame_targets_fltd

# def fltd_tensor2pandas(target_tensor, sort=True):
#     axons_ids = target_tensor[:,3].to(int) if target_tensor.shape[1]>3 else range(target_tensor.shape[0])
#     axon_lbls = [f'Axon_{i:0>3}' for i in axons_ids]
#     # axon_lbls = pd.MultiIndex.from_product([[t], axon_lbls])
#     cols = 'conf', 'anchor_x', 'anchor_y'
    
#     target_df = pd.DataFrame(target_tensor[:,:3].to('cpu').numpy(), index=axon_lbls, 
#                              columns=cols).convert_dtypes()
#     if sort:
#         target_df.sort_values('conf', inplace=True)
#     return target_df

# def Y2pandas_anchors(Y, Sx, Sy, tilesize, device, conf_thr):
#     # convert the YOLO coordinates (0-1) to image coordinates (tilesize x tilesize)
#     Y_imgcoo = yolo_coo2tile_coo(Y.clone(), Sx, Sy, tilesize, device)

#     # filter the (batch, Sx, Sy, 3) output into a list of length batchsize,
#     # where the list elements are tensors of shape (#_boxes_above_thr x 3)
#     # each element in the list represents a frame/tile
#     Ys_list_fltd = filter_anchors(Y_imgcoo, conf_threshold=conf_thr)
#     # convert each frame from tensor to pandas
#     for i in range(len(Ys_list_fltd)):
#         Ys_list_fltd[i] = fltd_tensor2pandas(Ys_list_fltd[i])
#     return Ys_list_fltd

# def compute_TP_FP_FN(pred_target, target):
#     obj_exists = target[..., 0].to(bool)
#     pred_obj_exists = pred_target[..., 0]
    
#     TP, FP, FN = [], [], []
#     conf_thresholds = np.arange(0.5, 1.01, 0.01)
#     for thr in conf_thresholds:
#         pos_pred = torch.where(pred_obj_exists>thr, 1, 0).to(bool)
#         TP.append((pos_pred & obj_exists).sum().item())
#         FP.append((pos_pred & ~obj_exists).sum().item())
#         FN.append((~pos_pred & obj_exists).sum().item())
#     return np.array([TP, FP, FN])

# def compute_prc_rcl_F1(confus_mtrx, epoch=None, which_data=None):
#     prc = confus_mtrx[0] / (confus_mtrx[0]+confus_mtrx[1]+0.000001)
#     rcl = confus_mtrx[0] / (confus_mtrx[0]+confus_mtrx[2]+0.000001)
#     f1 =  2*(prc*rcl)/(prc+rcl)

#     if epoch is not None and which_data is not None:
#         conf_thresholds = np.arange(0.5, 1.01, 0.01)
#         index = pd.MultiIndex.from_product([[epoch],[which_data],conf_thresholds])
#         return pd.DataFrame([prc, rcl, f1], columns=index, index=('precision', 'recall', 'F1'))
#     else:
#         return np.array([prc, rcl, f1])

# def non_max_supression_pandas(pred, min_dist):
#     # order by confidence
#     pred = pred.sort_values('conf', ascending=False)
#     i = 0
#     # get most confident anchor first
#     while i < pred.shape[0]:
#         xs, ys = pred[['anchor_x', 'anchor_y']].values.T
#         x,y = pred.iloc[i, 1:]

#         # compute distances of that anchor to all others
#         dists = np.sqrt(((xs-x)**2 + (ys-y)**2).astype(int))
#         # check where its < thr, skip the first one (dist. with itself = 0)
#         non_max_indices = np.where(dists<min_dist)[0][1:]
#         # drop those anchors
#         pred.drop(pred.index[non_max_indices], inplace=True)
#         i += 1
#     return pred
    
# def non_max_supress_pred(pred, Sx, Sy, tilesize, device, min_dist):
#     batchs = pred.shape[0]
#     # take the raw flat model output and reshape to standard yolo target shape
#     pred = pred.reshape((batchs, Sx, Sy, 3))
#     # convert coordinates to tile coordinates 
#     pred_tilecoo = yolo_coo2tile_coo(pred.clone(), Sx, Sy, tilesize, device)

#     # iterate the tiles/ batch
#     for tile_i in range(batchs):
#         # index to the tile and flatten the SxSy dim (12x120) -> 144 anchors
#         tile_pred = pred_tilecoo[tile_i]
#         tile_pred_flat = tile_pred.flatten(0,1)

#         # get the indices where the anchor confidence is at least above 0.5
#         pos_pred_idx = torch.where(tile_pred_flat[:,0] > 0.5)[0]
#         # sort these by their associated confidences 
#         pos_pred_idx = pos_pred_idx[tile_pred_flat[pos_pred_idx, 0].argsort(descending=True)]
        
#         i = 0
#         # iterate the conf-sorted anchor indices that were above 0.5
#         while i < len(pos_pred_idx):
#             # get the coordianates of all the anchors above 0.5
#             xs, ys = tile_pred_flat[pos_pred_idx, 1:].T
#             # and the specifc anchor being tested in this iteration
#             x,y = tile_pred_flat[pos_pred_idx[i], 1:]

#             # compute euclidian distances of that anchor to all others
#             dists = torch.sqrt(((xs-x)**2 + (ys-y)**2))
                
#             # check where it is < thr, skip the first one (dist. with itself = 0)
#             non_max_indices = torch.where(dists<min_dist)[0][1:]
#             # if any of the conf-above-0.5 anchors was smaller than distance:
#             if len(non_max_indices):
#                 # set the confidence at those indices to 0
#                 tile_pred_flat[pos_pred_idx[non_max_indices], 0] = 0
#                 # remove those anchor indices, they shouldn't be checked after being dropped
#                 pos_pred_idx = torch.tensor([pos_pred_idx[idx] for idx in range(len(pos_pred_idx)) if idx not in non_max_indices])
#             i += 1

#         # undo the flattening and set the tile-coo confidences to the yolo box confs.
#         tile_pred = tile_pred.reshape(Sx, Sy, 3)
#         pred[tile_i, :, :, 0] = tile_pred.reshape(Sx, Sy, 3)[:, :, 0]
#     return pred