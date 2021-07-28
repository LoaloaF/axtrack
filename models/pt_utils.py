import numpy as np
import pandas as pd
import torch

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

def coo2tensor(scipy_coo):
    indices = np.vstack((scipy_coo.row, scipy_coo.col))
    return torch.sparse_coo_tensor(indices, scipy_coo.data, scipy_coo.shape, dtype=torch.float32)

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