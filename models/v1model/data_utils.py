import numpy as np
import pandas as pd

import torch
import torchvision.transforms.functional as TF

def gen_denseTensorChunk(X, tchunksize):
    X_coal = X.coalesce()
    (t_idx, c_idx, y_idx, x_idx), vals = X_coal.indices(), X_coal.values()
    
    tstart, tstop = 0, tchunksize
    last_chunk = False
    while not last_chunk:
        if tstop >= X.shape[0]:
            tstop = X.shape[0]
            last_chunk = True

        t_mask = (t_idx >=tstart) & (t_idx <tstop)
        indices = torch.stack((t_idx[t_mask]-tstart, c_idx[t_mask], y_idx[t_mask], x_idx[t_mask]))
        outshape = (tstop-tstart, *X.shape[1:])

        yield torch.sparse_coo_tensor(indices, vals[t_mask], size=outshape).to_dense()
        tstart, tstop = tstop, tstop+tchunksize

def transform_X(X, tchunksize, angle, flip_dims, dy, dx, sizey, sizex):
    # translating
    if dx or dy:
        X_coal = X.coalesce()
        indices, vals = X_coal.indices(), X_coal.values()
        oof = torch.ones(vals.shape, dtype=bool) # ~out of frame

        if dy:
            indices[2] = (indices[2] + dy).to(torch.long)
            oof[(indices[2]>=sizey) | (indices[2]<0)] = False
        if dx:
            indices[3] = (indices[3] + dx).to(torch.long)
            oof[(indices[3]>=sizex) | (indices[3]<0)] = False
        
        # torch doens't broadcast like numpy?
        oof_brdcst = torch.stack([oof]*4)
        indices = indices[oof_brdcst].reshape((4, oof.sum()))
        X = torch.sparse_coo_tensor(indices, vals[oof], X.shape)
    
    if not flip_dims and not angle:
        return X.to('cpu').to_dense()
    else:
        X_dense = []
        for X_chunk in gen_denseTensorChunk(X, tchunksize):
            
            # flipping
            if flip_dims:
                X_chunk = torch.flip(X_chunk, flip_dims)
            # rotating
            if angle:
                X_chunk = TF.rotate(X_chunk, angle)
            X_dense.append(X_chunk.to('cpu'))

        return torch.cat(X_dense)

def transform_Y(target, angle, flip_dims,  dy, dx, sizey, sizex):
    def rotate_indices(bboxes, imshape, angle):
        angle = torch.tensor([angle * np.pi / 180.])
        y_mid = (imshape[0] + 1) / 2.
        x_mid = (imshape[1] + 1) / 2.

        rotated_bboxes = bboxes.copy()
        rotated_bboxes.loc[:] = np.nan
        
        for tpoint in bboxes.index:
            pixelwise_indices_df = bboxes.loc[tpoint].unstack().dropna()
            pixelwise_indices = pixelwise_indices_df.values

            for row in range(pixelwise_indices.shape[0]):
                x, y = pixelwise_indices[row]
                x_rot = (x- x_mid) * torch.cos(angle) + (y - y_mid) * torch.sin(angle)
                y_rot = -1.0 * (x - x_mid) * torch.sin(angle) + (y - y_mid) * torch.cos(angle)
                x_rot = torch.round(x_rot + x_mid)
                y_rot = torch.round(y_rot + y_mid)

                x_rot = x_rot if (x_rot > 0  and x_rot < imshape[1]) else 0
                y_rot = y_rot if (y_rot > 0  and y_rot < imshape[0]) else 0
                pixelwise_indices[row] = (x_rot, y_rot) if x_rot and y_rot else (np.nan, np.nan)

            rotated_bboxes.loc[tpoint,pixelwise_indices_df.index] = pixelwise_indices.flatten()
        return rotated_bboxes

    def flip_indices(bboxes, imshape, which):
        y_mid = (imshape[0] + 1) / 2.
        x_mid = (imshape[1] + 1) / 2.

        if 2 in which:
            bboxes.loc[:, (slice(None),'anchor_y')] = y_mid + (y_mid - bboxes.loc[:, (slice(None),'anchor_y')])
        if 3 in which:
            bboxes.loc[:, (slice(None),'anchor_x')] = x_mid + (x_mid - bboxes.loc[:, (slice(None),'anchor_x')])
        return bboxes

    # make a new dataframe with reset index (y-axis from 0 - last timepoint)
    target_transf = target.copy()
    # get the anchor points of all the bboxes (center point)
    anchors = target.loc[:,(slice(None),['anchor_y', 'anchor_x'])].sort_index(1).copy()

    # translating
    if dy or dx:
        y_anchor = anchors.loc[:, (slice(None), 'anchor_y')]
        x_anchor = anchors.loc[:, (slice(None), 'anchor_x')]
        if dy:
            y_anchor += dy
            oof = ((1 >= y_anchor) | (y_anchor >= sizey-1)).values
            y_anchor = y_anchor.where(~oof)
        if dx:
            x_anchor += dx
            oof = ((1 >= x_anchor) | (x_anchor >= sizex-1)).values
            x_anchor = x_anchor.where(~oof)
        # ----- check which bbox is lost after translation? -----
        anchors.loc[:, (slice(None), 'anchor_y')] = y_anchor
        anchors.loc[:, (slice(None), 'anchor_x')] = x_anchor

    # flipping
    if flip_dims:
        anchors = flip_indices(anchors, (sizey, sizex), flip_dims)
    
    # rotating
    # if transform_configs.get('rot',0) > .5:
    if angle:
        # ----- check which bbox is lost after rotation? -----
        anchors = rotate_indices(anchors, (sizey, sizex), angle)
    
    # get the new x,y anchors and mask extend in case bboxes got lost in rotation
    y_anchor = anchors.loc[:, (slice(None), 'anchor_y')].round()
    x_anchor = anchors.loc[:, (slice(None), 'anchor_x')].round()
    target_transf.loc[:, (slice(None),'anchor_y')] = y_anchor
    target_transf.loc[:, (slice(None),'anchor_x')] = x_anchor
    return target_transf

def apply_transformations(transform_configs, X, target, sizey, sizex, device):
    # make a transformation config: rotate, hflip, vflip translate - or not
    transform_configs = {key: round(torch.rand(1).item(), 2) for key in transform_configs}
    print(f'New transform config set: {transform_configs}\n'
            'Transforming data...', end='', flush=True)

    # translating parameter
    dy, dx = 0, 0
    # dy and dx are in [-.25, +.25]
    if transform_configs.get('translateY', 0) >.5:
        dy = round(sizey *(transform_configs.get('translateY', 0)-.75))
    if transform_configs.get('translateX', 0) >.5:
        dx = round(sizex *(transform_configs.get('translateY', 0)-.75))

    # flipping parameter
    flip_dims = []
    if transform_configs.get('hflip',0) > .5:
        flip_dims.append(2)
    if transform_configs.get('vflip',0) > .5:
        flip_dims.append(3)

    # rotating parameter
    angle = None
    if transform_configs.get('rot',0) > .5:
        angle = ((transform_configs['rot'] * 40) -20)
    
    # apply the transformation using paramters above, do on GPU, return on CPU
    tchunksize = 2
    X = transform_X(X.to(device), tchunksize, angle, flip_dims, dy, dx, sizey, sizex)
    # now transform the bboxes with the same paramters
    target_transf = transform_Y(target, angle, flip_dims, dy, dx, sizey, sizex)
    print('Done.')
    torch.cuda.empty_cache()
    return X, target_transf

def sprse_scipy2torch(scipy_coo):
    indices = np.vstack((scipy_coo.row, scipy_coo.col))
    return torch.sparse_coo_tensor(indices, scipy_coo.data, scipy_coo.shape, dtype=torch.float32)