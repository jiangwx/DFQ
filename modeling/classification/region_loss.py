import time
import torch
import math
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

def bbox_anchor_iou(bbox, anchor):
    # bbox[0] ground truth width, bbox[1] ground truth hight, anchor[0] anchor width, anchor[1], anchor hight
    inter_area = torch.min(bbox[0], anchor[0]) * torch.min(bbox[1], anchor[1])
    union_area = (bbox[0] * bbox[1] + 1e-16) + anchor[0] * anchor[1] - inter_area
    return inter_area / union_area

def box_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def boxes_iou(box1, box2, x1y1x2y2=True):
    """
    Returns the IoU of two bounding boxes
    """
    if not x1y1x2y2:
        # Transform from center and width to exact coordinates
        b1_x1, b1_x2 = box1[..., 0] - box1[..., 2] / 2, box1[..., 0] + box1[..., 2] / 2
        b1_y1, b1_y2 = box1[..., 1] - box1[..., 3] / 2, box1[..., 1] + box1[..., 3] / 2
        b2_x1, b2_x2 = box2[..., 0] - box2[..., 2] / 2, box2[..., 0] + box2[..., 2] / 2
        b2_y1, b2_y2 = box2[..., 1] - box2[..., 3] / 2, box2[..., 1] + box2[..., 3] / 2
    else:
        # Get the coordinates of bounding boxes
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[..., 0], box1[..., 1], box1[..., 2], box1[..., 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[..., 0], box2[..., 1], box2[..., 2], box2[..., 3]

    # get the corrdinates of the intersection rectangle
    inter_rect_x1 = torch.max(b1_x1, b2_x1)
    inter_rect_y1 = torch.max(b1_y1, b2_y1)
    inter_rect_x2 = torch.min(b1_x2, b2_x2)
    inter_rect_y2 = torch.min(b1_y2, b2_y2)
    # Intersection area
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(
        inter_rect_y2 - inter_rect_y1 + 1, min=0
    )
    # Union Area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = inter_area / (b1_area + b2_area - inter_area + 1e-16)

    return iou

def build_targets(pred_boxes, targets, anchors, ignore_thres):
    # target.shape [nB,4],(center x, center y, w, h)
    nB = pred_boxes.size(0)
    nA = pred_boxes.size(1)
    nH = pred_boxes.size(2)
    nW = pred_boxes.size(3)
    obj_mask   = torch.cuda.BoolTensor(nB, nA, nH, nW).fill_(False)
    noobj_mask = torch.cuda.BoolTensor(nB, nA, nH, nW).fill_(True)
    tx         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    ty         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    tw         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    th         = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)
    tconf      = torch.cuda.FloatTensor(nB, nA, nH, nW).fill_(0)

    gt_x = targets[:,0]*nW # ground truth x
    gt_y = targets[:,1]*nH # ground truth y
    gt_w = targets[:,2]*nW # ground truth w
    gt_h = targets[:,3]*nH # ground truth h

    gt_box = torch.cuda.FloatTensor(targets.shape)
    gt_box[:,0] = targets[:,0]*nW # ground truth x
    gt_box[:,1] = targets[:,1]*nH # ground truth y 
    gt_box[:,2] = targets[:,2]*nW # ground truth w
    gt_box[:,3] = targets[:,3]*nH # ground truth h
    grid_x = gt_x.long()  # grid x
    grid_y = gt_y.long()  # grid y

    recall50, recall75, avg_iou = 0.0, 0.0, 0.0
    for b in range(nB):
        anchor_ious = torch.stack([bbox_anchor_iou((gt_w[b],gt_h[b]), anchor) for anchor in anchors])
        best_ious, best_n = anchor_ious.max(0)
        obj_mask[b, best_n, grid_y[b], grid_x[b]] = True
        noobj_mask[b, best_n, grid_y[b], grid_x[b]] = False
        
        # Set noobj mask to zero where iou exceeds ignore threshold
        gt_boxes = gt_box[b].repeat(nA*nH*nW,1).view(nA,nH,nW,-1)
        ious = boxes_iou(pred_boxes[b], gt_boxes, x1y1x2y2=False)
        noobj_mask[b][ious>ignore_thres] = False

        # Coordinates
        tx[b, best_n, grid_y[b], grid_x[b]] = gt_x[b] - gt_x[b].floor()
        ty[b, best_n, grid_y[b], grid_x[b]] = gt_y[b] - gt_y[b].floor()
        # Width and height
        tw[b, best_n, grid_y[b], grid_x[b]] = torch.log(gt_w[b] / anchors[best_n][0] + 1e-16)
        th[b, best_n, grid_y[b], grid_x[b]] = torch.log(gt_h[b] / anchors[best_n][1] + 1e-16)
        tconf[b, best_n, grid_y[b], grid_x[b]] = 1
        iou = box_iou(pred_boxes[b, best_n, grid_y[b], grid_x[b]], gt_box[b], x1y1x2y2=False)
        if(iou > 0.5):
            recall50 = recall50 + 1
        if(iou > 0.75):
            recall75 = recall75 + 1
        avg_iou += iou.item()

    scale = 2 - targets[:,2]*targets[:,3]

    return obj_mask, noobj_mask, scale, tx, ty, tw, th, tconf, recall50/nB, recall75/nB, avg_iou/nB


class RegionLoss(nn.Module):
    def __init__(self, anchors=[[1.4940052559648322,2.3598481287086823],[4.0113013115312155,5.760873975661669]]):
        super(RegionLoss, self).__init__()
        self.anchors = torch.cuda.FloatTensor(anchors)
        self.num_anchors = len(anchors)
        self.noobject_scale = 1
        self.object_scale = 5
        self.thresh = 0.6
        self.seen = 0

    def forward(self, output, targets):
       
        nB = output.data.size(0)
        nA = self.num_anchors
        nH = output.data.size(2)
        nW = output.data.size(3)

        output  = output.view(nB, nA, 5, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
        x    = torch.sigmoid(output[...,0])
        y    = torch.sigmoid(output[...,1])
        w    = output[...,2]
        h    = output[...,3]
        conf = torch.sigmoid(output[...,4])
        
        pred_boxes = torch.cuda.FloatTensor(4,nB*nA*nH*nW)
        grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
        anchor_w = self.anchors[:,0]
        anchor_h = self.anchors[:,1]
        anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
        pred_boxes[0] = x.view(nB*nA*nH*nW) + grid_x
        pred_boxes[1] = y.view(nB*nA*nH*nW) + grid_y
        pred_boxes[2] = torch.exp(w).view(nB*nA*nH*nW) * anchor_w
        pred_boxes[3] = torch.exp(h).view(nB*nA*nH*nW) * anchor_h
        pred_boxes = pred_boxes.transpose(0,1).contiguous().view(nB,nA,nH,nW,4)
        #pred_boxes = convert2cpu(pred_boxes.transpose(0,1).contiguous().view(nB,nA,nH,nW,4))
        obj_mask, noobj_mask, scale, tx, ty, tw, th, tconf, recall50, recall75, avg_iou = build_targets(pred_boxes, targets.data, self.anchors, self.thresh)


        tx    = Variable(tx.cuda())
        ty    = Variable(ty.cuda())
        tw    = Variable(tw.cuda())
        th    = Variable(th.cuda())
        tconf = Variable(tconf.cuda())
        obj_mask = Variable(obj_mask.cuda())
        noobj_mask  = Variable(noobj_mask.cuda())
        
        loss_x = nn.MSELoss()(x[obj_mask]*scale, tx[obj_mask]*scale)
        loss_y = nn.MSELoss()(y[obj_mask]*scale, ty[obj_mask]*scale)
        loss_w = nn.MSELoss()(w[obj_mask]*scale, tw[obj_mask]*scale)
        loss_h = nn.MSELoss()(h[obj_mask]*scale, th[obj_mask]*scale)
        loss_conf = self.object_scale*nn.MSELoss()(conf[obj_mask], tconf[obj_mask]) + self.noobject_scale * nn.MSELoss()(conf[noobj_mask], tconf[noobj_mask])

        loss = loss_x + loss_y + loss_w + loss_h + loss_conf

        print('loss: x %f, y %f, w %f, h %f, conf %f, total loss %f, recall50 %f, recall75 %f, avg_iou %f' % (loss_x.data, loss_y.data, loss_w.data, loss_h.data, loss_conf.data, loss.data, recall50, recall75, avg_iou))

        return loss, recall50, recall75, avg_iou

def evaluate(output, targets, anchors = [[1.4940052559648322,2.3598481287086823],[4.0113013115312155,5.760873975661669]]):
    
    nB = output.data.size(0)
    nA = len(anchors)
    nH = output.data.size(2)
    nW = output.data.size(3)

    grid_x = torch.linspace(0, nW-1, nW).repeat(nH,1).repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
    grid_y = torch.linspace(0, nH-1, nH).repeat(nW,1).t().repeat(nB*nA, 1, 1).view(nB*nA*nH*nW).cuda()
    anchor_w = torch.cuda.FloatTensor(anchors)[:,0]
    anchor_h = torch.cuda.FloatTensor(anchors)[:,1]
    anchor_w = anchor_w.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)
    anchor_h = anchor_h.repeat(nB, 1).repeat(1, 1, nH*nW).view(nB*nA*nH*nW)

    output  = output.view(nB, nA, 5, nH, nW).permute(0, 1, 3, 4, 2).contiguous()
    conf    = torch.sigmoid(output[...,4]).view(nB*nA*nH*nW)

    gt_box = torch.cuda.FloatTensor(targets.shape)
    gt_box[:,0] = targets[:,0]*nW # ground truth x
    gt_box[:,1] = targets[:,1]*nH # ground truth y
    gt_box[:,2] = targets[:,2]*nW # ground truth w
    gt_box[:,3] = targets[:,3]*nH # ground truth h

    x = torch.sigmoid(output[..., 0]).view(nB*nA*nH*nW) + grid_x
    y = torch.sigmoid(output[..., 1]).view(nB*nA*nH*nW) + grid_y
    w = torch.exp(output[..., 2]).view(nB*nA*nH*nW) * anchor_w
    h = torch.exp(output[..., 3]).view(nB*nA*nH*nW) * anchor_h
    ious = np.zeros(nB)
    for b in range(nB):
        confidence = torch.FloatTensor(nA*nH*nW).copy_(conf[b*nA*nH*nW:(b+1)*nA*nH*nW]).detach().numpy()
        index = np.argmax(confidence)
        px = x[b*nA*nH*nW + index]
        py = y[b*nA*nH*nW + index]
        pw = w[b*nA*nH*nW + index]
        ph = h[b*nA*nH*nW + index]

        ious[b] = box_iou((px,py,pw,ph), gt_box[b], x1y1x2y2=False).item()
        
    return np.mean(ious)
