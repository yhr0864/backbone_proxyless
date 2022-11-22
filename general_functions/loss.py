import math

import torch
import torch.nn as nn

from .utils import to_cpu

# This new loss function is based on https://github.com/ultralytics/yolov3/blob/master/utils/loss.py


def bbox_iou(box1, box2, x1y1x2y2=True, GIoU=False, DIoU=False, CIoU=False, eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4xn, box2 is nx4
    box2 = box2.T

    # Get the coordinates of bounding boxes
    if x1y1x2y2:  # x1, y1, x2, y2 = box1
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[2], box1[3] # [1,n]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[2], box2[3] # [1,n]
    else:  # transform from xywh to xyxy
        b1_x1, b1_x2 = box1[0] - box1[2] / 2, box1[0] + box1[2] / 2
        b1_y1, b1_y2 = box1[1] - box1[3] / 2, box1[1] + box1[3] / 2
        b2_x1, b2_x2 = box2[0] - box2[2] / 2, box2[0] + box2[2] / 2
        b2_y1, b2_y2 = box2[1] - box2[3] / 2, box2[1] + box2[3] / 2

    # Intersection area
    inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0) # [1,n]

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps # [1,n]

    iou = inter / union # [1,n]
    if GIoU or DIoU or CIoU:
        # convex (smallest enclosing box) width 外接矩形
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height
        if CIoU or DIoU:  # Distance or Complete IoU https://arxiv.org/abs/1911.08287v1
            c2 = cw ** 2 + ch ** 2 + eps  # convex diagonal squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 +
                    (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4  # center distance squared
            if DIoU:
                return iou - rho2 / c2  # DIoU
            elif CIoU:  # https://github.com/Zzh-tju/DIoU-SSD-pytorch/blob/master/utils/box/box_utils.py#L47
                v = (4 / math.pi ** 2) * \
                    torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    alpha = v / ((1 + eps) - iou + v)
                return iou - (rho2 / c2 + v * alpha)  # CIoU
        else:  # GIoU https://arxiv.org/pdf/1902.09630.pdf
            c_area = cw * ch + eps  # convex area
            return iou - (c_area - union) / c_area  # GIoU
    else:
        return iou  # IoU [1,n]


def compute_loss(predictions, targets, model):
    '''
    input: 
        prediction: [(bs,3,26,26,25),(bs,3,13,13,25)]
        targets: [num_targets,6]
        model: yolov3 or yolov3-tiny  
    return: 
        loss:
    '''

    # Check which device was used
    device = targets.device

    # Add placeholder varables for the different losses
    lcls, lbox, lobj = torch.zeros(1, device=device), torch.zeros(1, device=device), torch.zeros(1, device=device)

    # Build yolo targets
    tcls, tbox, indices, anchors = build_targets(predictions, targets, model)  # targets

    # Define different loss functions classification
    BCEcls = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))
    BCEobj = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([1.0], device=device))

    # Calculate losses for each yolo layer
    for layer_index, layer_predictions in enumerate(predictions):
        # Get image ids, anchors, grid index i and j for each target in the current yolo layer
        b, anchor, grid_j, grid_i = indices[layer_index] # b.shape=(k,1) anchor.shape=(k,1) grid_j,grid_i.shape=(k,1) 共k组固定搭配参数
        # Build empty object target tensor with the same shape as the object prediction
        tobj = torch.zeros_like(layer_predictions[..., 0], device=device)  # target obj tobj.shape=(bs,3,26,26) or (bs,3,13,13) 
        # Get the number of targets for this layer.
        # Each target is a label box with some scaling and the association of an anchor box.
        # Label boxes may be associated to 0 or multiple anchors. So they are multiple times or not at all in the targets.
        num_targets = b.shape[0] # num_targets=?
        # Check if there are targets for this batch
        if num_targets:
            # Load the corresponding values from the predictions for each of the targets
            ps = layer_predictions[b, anchor, grid_j, grid_i] #从所有的predbox中筛选出与gtbox对应的 ps.shape=(k,25)

            # Regression of the box
            # Apply sigmoid to xy offset predictions in each cell that has a target
            pxy = ps[:, :2].sigmoid()
            # Apply exponent to wh predictions and multiply with the anchor box that matched best with the label for each cell that has a target
            pwh = torch.exp(ps[:, 2:4]) * anchors[layer_index]
            # Build box out of xy and wh
            pbox = torch.cat((pxy, pwh), 1) # pbox.shape=(k,4)
            # Calculate CIoU or GIoU for each target with the predicted box for its cell + anchor
            iou = bbox_iou(pbox.T, tbox[layer_index], x1y1x2y2=False, CIoU=True) # iou.shape=(1,k)
            # We want to minimize our loss so we and the best possible IoU is 1 so we take 1 - IoU and reduce it with a mean
            lbox += (1.0 - iou).mean()  # iou loss lbox.shape=(1,k) 定位损失

            # Classification of the objectness
            # Fill our empty object target tensor with the IoU we just calculated for each target at the targets position
            #tobj.shape=(bs,3,26,26) or (bs,3,13,13) 将iou>0的部分填入tobj,为正样本部分
            #tobj[b, anchor, grid_j, grid_i].shape=(k,)
            tobj[b, anchor, grid_j, grid_i] = iou.detach().clamp(0).type(tobj.dtype)  # Use cells with iou > 0 as object targets 

            # Classification of the class
            # Check if we need to do a classification (number of classes > 1)
            if ps.size(1) - 5 > 1:
                # Hot one class encoding
                t = torch.zeros_like(ps[:, 5:], device=device)  # targets t.shape=(?,20)
                t[range(num_targets), tcls[layer_index]] = 1
                # Use the tensor to calculate the BCE loss
                lcls += BCEcls(ps[:, 5:], t)  # BCE

        # Classification of the objectness the sequel
        # Calculate the BCE loss between the on the fly generated target and the network prediction
        lobj += BCEobj(layer_predictions[..., 4], tobj) # obj loss

    lbox *= 0.05
    lobj *= 1.0
    lcls *= 0.5

    # Merge losses
    loss = lbox + lobj + lcls

    return loss, to_cpu(torch.cat((lbox, lobj, lcls, loss)))


def build_targets(p, targets, model):
    '''
    input: 
        p: [(bs,3,26,26,25),(bs,3,13,13,25)]
        targets: [num_targets,6]所有的gtbox
        model: yolov3 or yolov3-tiny  
    return: 
        tcls.shape=(2*k,1), tcls[i]=(class_num) 2表示2layers
        tbox.shape=(2*k,4), tbox[i]=(gx-gi,gy-gj,gw,gh)
        indices.shape=(2*k,4), indices[i]=((img_id,anchor_id,cell_idx,cell_idy))
        anch.shape=(2*k,2), anch[i]=(anchor_w,anchor_h)
    '''
    # Build targets for compute_loss(), input targets(image,class,x,y,w,h)
    na, nt = 3, targets.shape[0]  # number of anchors, nt是当前batch图片中所有目标框的数量
    tcls, tbox, indices, anch = [], [], [], []
    gain = torch.ones(7, device=targets.device)  # normalized to gridspace gain
    # Make a tensor that iterates 0-2 for 3 anchors and repeat that as many times as we have target boxes
    ai = torch.arange(na, device=targets.device).float().view(na, 1).repeat(1, nt) #[3,num_targets]
    # Copy target boxes anchor size times and append an anchor index to each copy the anchor index is also expressed by the new first dimension
    # targets.repeat(na, 1, 1)->[3,nt,6], ai[3,nt,None]
    targets = torch.cat((targets.repeat(na, 1, 1), ai[:, :, None]), 2) #[3,nt,7] 多出的第7个数为对应anchor的index：0,1,2
    # targets的值[ [[image,class,x,y,w,h,0],
    #             [image,class,x,y,w,h,0],   ->anchor1
    #               	...		共nt个   ]

	# 			  [[image,class,x,y,w,h,1]，
	#              [image,class,x,y,w,h,1],  ->anchor2
	#                   ...		共nt个    ]

	# 			  [[image,class,x,y,w,h,2]，
	#              [image,class,x,y,w,h,2],  ->anchor3
	#                   ...		共nt个    ]
	#          ]

    for i, yolo_layer in enumerate(model.yolo_layers): # 每一层layer(共三层）单独计算, for tiny only 2 layers
        # Scale anchors by the yolo grid cell size so that an anchor with the size of the cell would result in 1
        anchors = yolo_layer.anchors / yolo_layer.stride #anchor放缩至featuremap上 [3,2]
        # Add the number of yolo cells in this layer the gain tensor
        # The gain tensor matches the collums of our targets (img id, class, x, y, w, h, anchor id)
        gain[2:6] = torch.tensor(p[i].shape)[[3, 2, 3, 2]]  # p[i].shape: [bs,3,26,26,25], gain[2:6]=tensor[26,26,26,26] xyxy gain
        # Scale targets by the number of yolo layer cells, they are now in the yolo cell coordinate system
        # targets是归一化的值，要放缩到featuremap下
        t = targets * gain # [3,nt,7]
        # Check if we have targets
        if nt:
            # Calculate ration between anchor and target box for both width and height
            # Matches, t[:, :, 4:6].shape=(3, nt, 2), anchors[:, None].shape=(3, 1, 2)
            r = t[:, :, 4:6] / anchors[:, None] #获得所有gtbox与anchor边长比(每个gtbox都对应3个anchor)
            # Select the ratios that have the highest divergence in any axis and check if the ratio is less than 4
            j = torch.max(r, 1. / r).max(2)[0] < 4  # compare #TODO j.shape = (3, nt) 
            # Only use targets that have the correct ratios for their anchors
            # That means we only keep ones that have a matching anchor and we loose the anchor dimension
            # The anchor id is still saved in the 7th value of each target
            
            # before filter, t.shape = (3, nt, 7), j.shape = (3, nt)
            # after filter, t.shape = (k, 7) 
            # t经过[j]索引后，就是匹配上这3个anchor的targets，匹配的数量视情况而定，这里用k表示。
            # t经过[j]索引后的值有重复，因为每个target可能匹配到多个anchor,也可能有的gtbox没有anchor对应
            # 到这一步其实指示根据anchor的长宽比来计算匹配的targets。

            t = t[j] # t.shape = (k, 7) 也可理解为对每个anchor筛选出满足条件的gtbox并全部拼起来得到t,存在同gtbox不同anchor,同anchor不同gtbox
        else:
            t = targets[0] # t.shape = (nt,7)
        
        # t = [image,class,x,y,w,h,0
        #             ... ...
        #     image,class,x,y,w,h,0]  共k行
        # Extract image id in batch and class id
        b, c = t[:, :2].long().T # b, c.shape=(k,1)
        # We isolate the target cell associations.
        # x, y, w, h are allready in the featuremap meaning an x = 1.2 would be 1.2 times cellwidth=1
        gxy = t[:, 2:4] # gxy.shape = (k,2)
        gwh = t[:, 4:6]  # grid wh
        # Cast to int to get an cell index e.g. 1.2 gets associated to cell 1
        gij = gxy.long() # 向下取整
        # Isolate x and y index dimensions
        gi, gj = gij.T  # grid xy indices

        # Convert anchor indexes to int
        a = t[:, 6].long() # a.shape = (k,1) 即每行gtbox对应的满足条件的anchor(可能会重复)
        # Add target tensors for this yolo layer to the output lists
        # Add to index list and limit index range to prevent out of bounds
        indices.append((b, a, gj.clamp_(0, gain[3] - 1), gi.clamp_(0, gain[2] - 1))) # cell_index:gj.clamp_(0, gain[3] - 1):gj取0-25,>25则取25,<0则取0
        # Add to target box list and convert box coordinates from global grid coordinates to local offsets in the grid cell
        tbox.append(torch.cat((gxy - gij, gwh), 1))  # box只含x,y,w,h且是相对girdcell左上角偏移坐标 tbox.shape = (2*k,4) 2表示2layers
        # Add correct anchor for each target to the list
        anch.append(anchors[a]) # anch.shape = (2*k,2) 每3个选一个 2表示2layers
        # Add class for each target to the list
        tcls.append(c) # tcls.shape = (2*k,1) 2表示2layers

    return tcls, tbox, indices, anch
