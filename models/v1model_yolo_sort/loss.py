"""
Implementation of Yolo Loss Function from the original yolo paper

"""
import pandas as pd
import numpy as np

import torch
import torch.nn as nn

class YOLO_AXTrack_loss(nn.Module):
    """
    Calculate the loss for yolo (v1) model
    """

    def __init__(self, Sy, Sx, lambda_obj, lambda_noobj, lambda_coord_anchor):
        super(YOLO_AXTrack_loss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")

        """
        S is split size of image (in paper 7),
        B is number of boxes (in paper 2),
        """

        self.Sy = Sy
        self.Sx = Sx
        self.B = 1

        self.lambda_obj = lambda_obj
        self.lambda_noobj = lambda_noobj
        self.lambda_coord_anchor = lambda_coord_anchor

    def forward(self, predictions, target):
        # predictions are shaped (BATCH_SIZE, Sx*Sy(B*3) when inputted
        bs = target.shape[0]
        total_pos_labels_rate = target[..., 0].sum() /(bs*self.Sx*self.Sy)
        n = bs * self.Sy * self.Sx
        predictions = predictions.reshape(bs, self.Sy, self.Sx, self.B*3)

        # Calculate IoU for the two predicted bounding boxes with target bbox
        # size of predictions is Sx x Sy x 5, if a box exists, 1,0>x>1, 0>y>1, w>0, h>0 
        obj_exists, true_xy = target[..., 0:1], target[..., 1:3]
        no_obj_exists = 1 - obj_exists
        pred_conf, pred_xy = predictions[..., 0:1], predictions[..., 1:3]

        pred_xy_positive_label = pred_xy * obj_exists
        pred_conf_positive_label = pred_conf * obj_exists
        pred_conf_negative_label = pred_conf * no_obj_exists

        # ======================== #
        #   FOR BOX COORDINATES    #
        # ======================== #
        box_loss_anchors = self.mse(
            torch.flatten(pred_xy_positive_label, end_dim=-2),
            torch.flatten(true_xy, end_dim=-2),
        )
        # print(box_loss_anchors.item(), end='...')

        # ==================== #
        #   FOR OBJECT LOSS    #
        # ==================== #
        object_loss = self.mse(
            torch.flatten(pred_conf_positive_label),
            torch.flatten(obj_exists),
        )
        # print('Object loss: ', object_loss.item(), end='...')

        # ======================= #
        #   FOR NO OBJECT LOSS    #
        # ======================= #
        no_object_loss = self.mse(
            torch.flatten(pred_conf_negative_label, start_dim=1),
            torch.flatten(torch.zeros_like(pred_conf_negative_label), start_dim=1),
        )
        # print('no Object loss: ', no_object_loss.item(), end='...')

        loss_components = {
                    'total_xy_anchors_loss': (self.lambda_coord_anchor * box_loss_anchors) /bs,
                    'total_object_loss': (self.lambda_obj * object_loss) /bs,
                    'total_no_object_loss': (self.lambda_noobj * no_object_loss) /bs}
        loss = sum([l for l in loss_components.values()])
        loss_components['total_summed_loss'] = loss
        loss_components['total_pos_labels_rate'] = total_pos_labels_rate
        loss_components = pd.Series({idx: v.item() for idx, v in loss_components.items()})
        return loss, loss_components




# def get_iou(loader, model, iou_threshold, threshold, device, 
#                pred_format="cells", box_format="midpoint"):
#     Sx = model.Sx
#     Sy = model.Sy
#     all_pred_boxes = []
#     all_true_boxes = []

#     # make sure model is in eval before get bboxes
#     model.eval()
#     train_idx = 0
#     for batch_idx, (x, labels) in enumerate(loader):
#         batch_size = x.shape[0]
#         x = x.to(device)
#         labels = labels.to(device)

#         with torch.no_grad():
#             predictions = model(x)

#         bboxes = convert_cellboxes(out, Sy, Sx, device)[...,2:6]
#         bboxes[..., 0] = bboxes[..., 0].long()
#         pred_bboxes = convert_cellboxes(out, Sy, Sx, device)[...,2:6]
#         pred_bboxes[..., 0] = pred_bboxes[..., 0].long()

#         iou = intersection_over_union(pred_bboxes, bboxes)
#         print(iou)

#         converted_pred = convert_cellboxes(predictions, Sy, Sx).reshape(batch_size, Sy, Sx, -1)
#         # converted_pred[..., 0] = converted_pred[..., 0].long()
#         converted_target = convert_cellboxes(labels, Sy, Sx).reshape(batch_size, Sy, Sx, -1)
#         # converted_pred[..., 0] = converted_pred[..., 0].long()
        
#         # Calculate IoU for the two predicted bounding boxes with target bbox
#         # size of predictions is Sx x Sy x 5, if a box exists, 1,0>x>1, 0>y>1, w>0, h>0 
#         iou = intersection_over_union(converted_pred[..., 2:6], converted_target[..., 2:6])
#         # ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
#         print(iou.mean())
#         continue