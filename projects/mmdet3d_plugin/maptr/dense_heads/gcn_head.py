import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet.models import HEADS, build_loss
from mmdet.models.dense_heads import DETRHead, AnchorFreeHead
from .detr_head import DETRMapHead
from mmdet3d.core.bbox.coders import build_bbox_coder
from mmcv.runner import force_fp32, auto_fp16
from mmcv.cnn import Linear, bias_init_with_prob, xavier_init, constant_init
from mmdet.models.utils.transformer import inverse_sigmoid
from mmdet.core.bbox.transforms import bbox_xyxy_to_cxcywh, bbox_cxcywh_to_xyxy
from mmdet.core import (multi_apply, multi_apply, reduce_mean, build_assigner)
from mmcv.utils import TORCH_VERSION, digit_version

from .EdgeConv.blocks import FeatureExtraction, DownsampleAdjust, Upsampling

from .GCN.net import GCNFeatureExtractor, Downsample


def normalize_2d_bbox(bboxes, pc_range):

    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    cxcywh_bboxes = bbox_xyxy_to_cxcywh(bboxes)
    cxcywh_bboxes[...,0:1] = cxcywh_bboxes[..., 0:1] - pc_range[0]
    cxcywh_bboxes[...,1:2] = cxcywh_bboxes[...,1:2] - pc_range[1]
    factor = bboxes.new_tensor([patch_w, patch_h,patch_w,patch_h])

    normalized_bboxes = cxcywh_bboxes / factor
    return normalized_bboxes

def normalize_2d_pts(pts, pc_range):
    patch_h = pc_range[4]-pc_range[1]
    patch_w = pc_range[3]-pc_range[0]
    new_pts = pts.clone()
    new_pts[...,0:1] = pts[..., 0:1] - pc_range[0]
    new_pts[...,1:2] = pts[...,1:2] - pc_range[1]
    factor = pts.new_tensor([patch_w, patch_h])
    normalized_pts = new_pts / factor
    return normalized_pts

def denormalize_2d_bbox(bboxes, pc_range):

    bboxes = bbox_cxcywh_to_xyxy(bboxes)
    bboxes[..., 0::2] = (bboxes[..., 0::2]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    bboxes[..., 1::2] = (bboxes[..., 1::2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])

    return bboxes
def denormalize_2d_pts(pts, pc_range):
    new_pts = pts.clone()
    new_pts[...,0:1] = (pts[..., 0:1]*(pc_range[3] -
                            pc_range[0]) + pc_range[0])
    new_pts[...,1:2] = (pts[...,1:2]*(pc_range[4] -
                            pc_range[1]) + pc_range[1])
    return new_pts

@HEADS.register_module()
class GCNHead(nn.Module):
    """Head of Detr3D.
    Args:
        with_box_refine (bool): Whether to refine the reference points
            in the decoder. Defaults to False.
        as_two_stage (bool) : Whether to generate the proposal from
            the outputs of encoder.
        transformer (obj:`ConfigDict`): ConfigDict is used for building
            the Encoder and Decoder.
        bev_h, bev_w (int): spatial shape of BEV queries.
    """

    def __init__(self,  
                    input_dim=2, 
                    output_dim=64,
                    bbox_coder=None,
                    dir_interval=1,
                    pc_range=None,
                    num_vec=50,
                    num_pts_per_vec=20,
                    num_pts_per_gt_vec = 20,
                    transform_method='minmax',
                    code_weights=[1.0, 1.0, 1.0, 1.0],
                    loss_bbox=dict(type='L1Loss', loss_weight=0.0),
                    loss_iou=dict(type='GIoULoss', loss_weight=0.0),
                    loss_pts=dict(type='PtsL1Loss', loss_weight=5.0),
                    #  loss_cos=dict(type='CosineLoss', loss_weight=5.0),
                    loss_dir=dict(type='PtsDirCosLoss', loss_weight=0.005),
                    loss_angle=dict(type='PtsAngleLoss', loss_weight=2.0),
                    **idgcn_kwargs):
        super(GCNHead, self).__init__()
        self.dir_interval = dir_interval
        self.pc_range = pc_range
        self.num_vec = num_vec
        self.num_pts_per_vec = num_pts_per_vec
        self.num_pts_per_gt_vec = num_pts_per_gt_vec
        self.transform_method = transform_method
        self.code_weights = nn.Parameter(torch.tensor(
            code_weights, requires_grad=False), requires_grad=False)
        self.feats = nn.ModuleList()
        self.feat_dim = 0
        
        # Layers
        self.feature_extractor = GCNFeatureExtractor(input_dim, output_dim)
        self.reconstructor = Downsample(output_dim*3+256)
        
        self.bbox_coder = build_bbox_coder(bbox_coder)

        self.loss_bbox = build_loss(loss_bbox)
        self.loss_iou = build_loss(loss_iou)
        self.loss_pts = build_loss(loss_pts)
        # self.loss_cos = build_loss(loss_cos)
        self.loss_dir = build_loss(loss_dir)
        self.loss_angle = build_loss(loss_angle)
        self.loss_pts_type = loss_pts['type']
    
    # @auto_fp16(apply_to=('mlvl_feats'))
    @force_fp32(apply_to=('mlvl_feats', 'prev_bev'))
    def forward(self, preds_dicts):
        """Forward function.
        Args:
            points_pred: [6, bs, 50, 20, 2]
            mlvl_feats (tuple[Tensor]): Features from the upstream
                network, each is a 5D-tensor with shape
                (B, N, C, H, W).
            prev_bev: previous bev featues
            only_bev: only compute BEV features with encoder. 
        Returns:
            all_cls_scores (Tensor): Outputs from the classification head, \
                shape [nb_dec, bs, num_query, cls_out_channels]. Note \
                cls_out_channels should includes background.
            all_bbox_preds (Tensor): Sigmoid outputs from the regression \
                head with normalized coordinate format (cx, cy, w, l, cz, h, theta, vx, vy). \
                Shape [nb_dec, bs, num_query, 9].
        """
        decoder_embed = preds_dicts['decoder_embed']
        all_cls_scores = preds_dicts['all_cls_scores']#6*4*50*3
        all_bbox_preds = preds_dicts['all_bbox_preds']
        all_pts_preds  = preds_dicts['all_pts_preds']
        enc_cls_scores = preds_dicts['enc_cls_scores']
        enc_bbox_preds = preds_dicts['enc_bbox_preds']
        enc_pts_preds  = preds_dicts['enc_pts_preds']

        layers_num, bs, samples_num, pts_num, coords_num = all_pts_preds.shape
        points_input = all_pts_preds.view(layers_num*bs*samples_num, pts_num, coords_num)
        decoder_embed = decoder_embed.contiguous().view(layers_num*bs*samples_num, pts_num, 256)

        feat = self.feature_extractor(points_input)
        # print('feat: ', feat.shape)
        # print('decoder_embed: ', decoder_embed.shape)
        feat = torch.cat((feat, decoder_embed), dim=-1)
        # x = self.upsampler(x, edge_index=edge_index, pos=pos, batch=batch)
        pos = self.reconstructor(points_input, feat)

        # polygon_idx = all_cls_scores.argmax(-1) == 0
        # # all_cls_scores = all_cls_scores.view(layers_num*bs*samples_num, -1).unsqueeze(-2).repeat(1, pts_num, 1)

        # feats = []
        # for feat_unit in self.feats:
        #     feats.append(feat_unit(points_input))
        # feat = torch.cat(feats, dim=-1)
        # feat = torch.cat((feat, decoder_embed), dim=-1)
        # # feat = torch.cat((feat, all_cls_scores), dim=-1)

        # idx, pos, feat = self.downsample(points_input, feat)
        # self.adjusted = pos
        # pos = self.upsample(pos, feat)
        # out = pos.view(layers_num, bs, samples_num, -1, coords_num)

        # pos = pos.permute(0,2,1)
        # pos = F.interpolate(pos, size=(self.num_pts_per_gt_vec), mode='linear',
        #                         align_corners=True)
        # out = pos.permute(0,2,1).contiguous().view(layers_num, bs, samples_num, -1, coords_num)
        out = pos.contiguous().view(layers_num, bs, samples_num, -1, coords_num)
        # out[polygon_idx] = all_pts_preds[polygon_idx]

        outputs_coords = []
        outputs_pts_coords = []
        for lvl in range(layers_num):
            outputs_coord, outputs_pts_coord = self.transform_box(out[lvl])
            outputs_coords.append(outputs_coord)
            outputs_pts_coords.append(outputs_pts_coord)
        outputs_coords = torch.stack(outputs_coords)
        outputs_pts_coords = torch.stack(outputs_pts_coords)
        preds_dicts.update({'all_bbox_preds': outputs_coords, 'all_pts_preds': outputs_pts_coords})

        return preds_dicts

        
    def transform_box(self, pts, y_first=False):
        """
        Converting the points set into bounding box.

        Args:
            pts: the input points sets (fields), each points
                set (fields) is represented as 2n scalar.
            y_first: if y_fisrt=True, the point set is represented as
                [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
                represented as [x1, y1, x2, y2 ... xn, yn].
        Returns:
            The bbox [cx, cy, w, h] transformed from points.
        """

        pts_reshape = pts.view(pts.shape[0], self.num_vec,
                                self.num_pts_per_vec,2)
        pts_y = pts_reshape[:, :, :, 0] if y_first else pts_reshape[:, :, :, 1]
        pts_x = pts_reshape[:, :, :, 1] if y_first else pts_reshape[:, :, :, 0]
        if self.transform_method == 'minmax':
            # import pdb;pdb.set_trace()

            xmin = pts_x.min(dim=2, keepdim=True)[0]
            xmax = pts_x.max(dim=2, keepdim=True)[0]
            ymin = pts_y.min(dim=2, keepdim=True)[0]
            ymax = pts_y.max(dim=2, keepdim=True)[0]
            bbox = torch.cat([xmin, ymin, xmax, ymax], dim=2)
            bbox = bbox_xyxy_to_cxcywh(bbox)
        else:
            raise NotImplementedError
        return bbox, pts_reshape
    def _get_target_single(self,
                           cls_score,
                           bbox_pred,
                           pts_pred,
                           gt_labels,
                           gt_bboxes,
                           gt_shifts_pts,
                           gt_bboxes_ignore=None):
        """"Compute regression and classification targets for one image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_score (Tensor): Box score logits from a single decoder layer
                for one image. Shape [num_query, cls_out_channels].
            bbox_pred (Tensor): Sigmoid outputs from a single decoder layer
                for one image, with normalized coordinate (cx, cy, w, h) and
                shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth bboxes for one image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (Tensor): Ground truth class indices for one image
                with shape (num_gts, ).
            gt_bboxes_ignore (Tensor, optional): Bounding boxes
                which can be ignored. Default None.
        Returns:
            tuple[Tensor]: a tuple containing the following for one image.
                - labels (Tensor): Labels of each image.
                - label_weights (Tensor]): Label weights of each image.
                - bbox_targets (Tensor): BBox targets of each image.
                - bbox_weights (Tensor): BBox weights of each image.
                - pos_inds (Tensor): Sampled positive indices for each image.
                - neg_inds (Tensor): Sampled negative indices for each image.
        """
        # print('gt_shifts_pts: ', gt_shifts_pts)
        # assert False, 'test'
        # import pdb;pdb.set_trace()
        num_bboxes = bbox_pred.size(0)
        # assigner and sampler
        gt_c = gt_bboxes.shape[-1]
        # import pdb;pdb.set_trace()
        assign_result, order_index = self.assigner.assign(bbox_pred, cls_score, pts_pred,
                                             gt_bboxes, gt_labels, gt_shifts_pts,
                                             gt_bboxes_ignore)

        sampling_result = self.sampler.sample(assign_result, bbox_pred,
                                              gt_bboxes)
        # pts_sampling_result = self.sampler.sample(assign_result, pts_pred,
        #                                       gt_pts)

        
        # import pdb;pdb.set_trace()
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # label targets
        labels = gt_bboxes.new_full((num_bboxes,),
                                    self.num_classes,
                                    dtype=torch.long)
        labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
        label_weights = gt_bboxes.new_ones(num_bboxes)

        # bbox targets
        bbox_targets = torch.zeros_like(bbox_pred)[..., :gt_c]
        bbox_weights = torch.zeros_like(bbox_pred)
        bbox_weights[pos_inds] = 1.0

        # pts targets
        # import pdb;pdb.set_trace()
        # pts_targets = torch.zeros_like(pts_pred)
        # num_query, num_order, num_points, num_coords
        if order_index is None:
            # import pdb;pdb.set_trace()
            assigned_shift = gt_labels[sampling_result.pos_assigned_gt_inds]
        else:
            assigned_shift = order_index[sampling_result.pos_inds, sampling_result.pos_assigned_gt_inds]
        pts_targets = pts_pred.new_zeros((pts_pred.size(0),
                        pts_pred.size(1), pts_pred.size(2)))
        pts_weights = torch.zeros_like(pts_targets)
        pts_weights[pos_inds] = 1.0

        # DETR
        bbox_targets[pos_inds] = sampling_result.pos_gt_bboxes
        pts_targets[pos_inds] = gt_shifts_pts[sampling_result.pos_assigned_gt_inds,assigned_shift,:,:]
        return (labels, label_weights, bbox_targets, bbox_weights,
                pts_targets, pts_weights,
                pos_inds, neg_inds)

    def get_targets(self,
                    cls_scores_list,
                    bbox_preds_list,
                    pts_preds_list,
                    gt_bboxes_list,
                    gt_labels_list,
                    gt_shifts_pts_list,
                    gt_bboxes_ignore_list=None):
        """"Compute regression and classification targets for a batch image.
        Outputs from a single decoder layer of a single feature level are used.
        Args:
            cls_scores_list (list[Tensor]): Box score logits from a single
                decoder layer for each image with shape [num_query,
                cls_out_channels].
            bbox_preds_list (list[Tensor]): Sigmoid outputs from a single
                decoder layer for each image, with normalized coordinate
                (cx, cy, w, h) and shape [num_query, 4].
            gt_bboxes_list (list[Tensor]): Ground truth bboxes for each image
                with shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels_list (list[Tensor]): Ground truth class indices for each
                image with shape (num_gts, ).
            gt_bboxes_ignore_list (list[Tensor], optional): Bounding
                boxes which can be ignored for each image. Default None.
        Returns:
            tuple: a tuple containing the following targets.
                - labels_list (list[Tensor]): Labels for all images.
                - label_weights_list (list[Tensor]): Label weights for all \
                    images.
                - bbox_targets_list (list[Tensor]): BBox targets for all \
                    images.
                - bbox_weights_list (list[Tensor]): BBox weights for all \
                    images.
                - num_total_pos (int): Number of positive samples in all \
                    images.
                - num_total_neg (int): Number of negative samples in all \
                    images.
        """
        assert gt_bboxes_ignore_list is None, \
            'Only supports for gt_bboxes_ignore setting to None.'
        num_imgs = len(cls_scores_list)
        gt_bboxes_ignore_list = [
            gt_bboxes_ignore_list for _ in range(num_imgs)
        ]

        (labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, pts_targets_list, pts_weights_list,
         pos_inds_list, neg_inds_list) = multi_apply(
            self._get_target_single, cls_scores_list, bbox_preds_list,pts_preds_list,
            gt_labels_list, gt_bboxes_list, gt_shifts_pts_list, gt_bboxes_ignore_list)
        num_total_pos = sum((inds.numel() for inds in pos_inds_list))
        num_total_neg = sum((inds.numel() for inds in neg_inds_list))
        return (labels_list, label_weights_list, bbox_targets_list,
                bbox_weights_list, pts_targets_list, pts_weights_list,
                num_total_pos, num_total_neg)

    def loss_single(self,
                    cls_reg_targets, 
                    pts_preds, 
                    bbox_preds):
        """"Loss function for outputs from a single decoder layer of a single
        feature level.
        Args:
        Returns:
            dict[str, Tensor]: A dictionary of loss components for outputs from
                a single decoder layer.
        """
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         pts_targets_list, pts_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets

        labels = torch.cat(labels_list, 0)
        label_weights = torch.cat(label_weights_list, 0)
        bbox_targets = torch.cat(bbox_targets_list, 0)
        bbox_weights = torch.cat(bbox_weights_list, 0)
        pts_targets = torch.cat(pts_targets_list, 0)
        pts_weights = torch.cat(pts_weights_list, 0)

        bbox_preds = bbox_preds.reshape(-1, bbox_preds.size(-1))
        normalized_bbox_targets = normalize_2d_bbox(bbox_targets, self.pc_range)
        # normalized_bbox_targets = bbox_targets
        isnotnan = torch.isfinite(normalized_bbox_targets).all(dim=-1)
        bbox_weights = bbox_weights * self.code_weights

        loss_bbox = self.loss_bbox(
            bbox_preds[isnotnan, :4], normalized_bbox_targets[isnotnan,
                                                               :4], bbox_weights[isnotnan, :4],
            avg_factor=num_total_pos)

        # regression pts CD loss
        # pts_preds = pts_preds
        # import pdb;pdb.set_trace()
        
        # num_samples, num_order, num_pts, num_coords
        normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # num_samples, num_pts, num_coords
        pts_preds = pts_preds.reshape(-1, pts_preds.size(-2),pts_preds.size(-1))
        if self.num_pts_per_vec != self.num_pts_per_gt_vec:
            pts_preds = pts_preds.permute(0,2,1)
            pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
                                    align_corners=True)
            pts_preds = pts_preds.permute(0,2,1).contiguous()

        # import pdb;pdb.set_trace()
        if self.loss_pts_type == 'ChamferDistance':
            loss_source, loss_target = self.loss_pts(pts_preds[isnotnan,:,:], 
                                                    normalized_pts_targets[isnotnan,:,:])
            loss_pts = (loss_source + loss_target)/2
        else:
            loss_pts = self.loss_pts(
                pts_preds[isnotnan,:,:], normalized_pts_targets[isnotnan,
                                                                :,:], 
                pts_weights[isnotnan,:,:],
                avg_factor=num_total_pos)
        # loss_cos = self.loss_cos(pts_preds[isnotnan,:,:], 
        #                             normalized_pts_targets[isnotnan,:,:], 
        #                             pts_weights[isnotnan,:,0],
        #                             avg_factor=num_total_pos)
        # loss_pts += loss_cos
        dir_weights = pts_weights[:, :-self.dir_interval,0]
        denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        
        denormed_pts_preds_dedir = denormed_pts_preds[:,:-self.dir_interval,:] - denormed_pts_preds[:,self.dir_interval:,:]
        pts_targets_dedir = pts_targets[:, :-self.dir_interval,:] - pts_targets[:,self.dir_interval:,:]
        # dir_weights = pts_weights[:, indice,:-1,0]
        # import pdb;pdb.set_trace()
        loss_dir = self.loss_dir(
            denormed_pts_preds_dir[isnotnan,:,:], pts_targets_dir[isnotnan,
                                                                          :,:],
            dir_weights[isnotnan,:],
            avg_factor=num_total_pos)

        loss_angle = self.loss_angle(denormed_pts_preds_dir[isnotnan,:,:],
                                        denormed_pts_preds_dedir[isnotnan,:,:],
                                        pts_targets_dir[isnotnan,:,:],
                                        pts_targets_dedir[isnotnan,:,:],
                                        dir_weights[isnotnan,:],
                                        avg_factor=num_total_pos)

        bboxes = denormalize_2d_bbox(bbox_preds, self.pc_range)
        # regression IoU loss, defaultly GIoU loss
        loss_iou = self.loss_iou(
            bboxes[isnotnan, :4], bbox_targets[isnotnan, :4], bbox_weights[isnotnan, :4], 
            avg_factor=num_total_pos)

        if digit_version(TORCH_VERSION) >= digit_version('1.8'):
            # loss_cls = torch.nan_to_num(loss_cls)
            loss_bbox = torch.nan_to_num(loss_bbox)
            loss_iou = torch.nan_to_num(loss_iou)
            loss_pts = torch.nan_to_num(loss_pts)
            loss_dir = torch.nan_to_num(loss_dir)
            loss_angle = torch.nan_to_num(loss_angle)
        return loss_bbox, loss_iou, loss_pts, loss_dir, loss_angle

        # # num_samples, num_order, num_pts, num_coords
        # bbox_coord, outputs_pts_coord = self.transform_box(pts_preds)
        # normalized_pts_targets = normalize_2d_pts(pts_targets, self.pc_range)

        # # num_samples, num_pts, num_coords
        # pts_preds = pts_preds.reshape(-1, pts_preds.size(-2), pts_preds.size(-1))
        # # if self.num_pts_per_vec != self.num_pts_per_gt_vec:
        # #     pts_preds = pts_preds.permute(0,2,1)
        # #     pts_preds = F.interpolate(pts_preds, size=(self.num_pts_per_gt_vec), mode='linear',
        # #                             align_corners=True)
        # #     pts_preds = pts_preds.permute(0,2,1).contiguous()

        # # import pdb;pdb.set_trace()
        # loss_pts = self.loss_pts(pts_preds, normalized_pts_targets, pts_weights, avg_factor=num_total_pos)
        # # loss_cos = self.loss_cos(pts_preds[isnotnan,:,:], 
        # #                             normalized_pts_targets[isnotnan,:,:], 
        # #                             pts_weights[isnotnan,:,0],
        # #                             avg_factor=num_total_pos)
        # # loss_pts += loss_cos
        # dir_weights = pts_weights[:, :-self.dir_interval,0]
        # denormed_pts_preds = denormalize_2d_pts(pts_preds, self.pc_range)
        # denormed_pts_preds_dir = denormed_pts_preds[:,self.dir_interval:,:] - denormed_pts_preds[:,:-self.dir_interval,:]
        # pts_targets_dir = pts_targets[:, self.dir_interval:,:] - pts_targets[:,:-self.dir_interval,:]
        # # dir_weights = pts_weights[:, indice,:-1,0]
        # # import pdb;pdb.set_trace()
        # loss_dir = self.loss_dir(denormed_pts_preds_dir, pts_targets_dir,
        #                             dir_weights, avg_factor=num_total_pos)

        # if digit_version(TORCH_VERSION) >= digit_version('1.8'):
        #     loss_pts = torch.nan_to_num(loss_pts)
        #     loss_dir = torch.nan_to_num(loss_dir)
        # return loss_pts, loss_dir

    @force_fp32(apply_to=('outs_manifold'))
    def loss(self,
             cls_reg_targets, 
             outs_manifold):
        """"Loss function.
        Args:
        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        all_cls_scores = outs_manifold['all_cls_scores']
        all_bbox_preds = outs_manifold['all_bbox_preds']
        all_pts_preds  = outs_manifold['all_pts_preds']
        enc_cls_scores = outs_manifold['enc_cls_scores']
        enc_bbox_preds = outs_manifold['enc_bbox_preds']
        enc_pts_preds  = outs_manifold['enc_pts_preds']

        losses_bbox, losses_iou, losses_pts, losses_dir, losses_angle = multi_apply(self.loss_single, cls_reg_targets, all_pts_preds, all_bbox_preds)

        # loss_dict = dict()
        # loss of proposal generated from encode feature map.
        # if enc_cls_scores is not None:
        #     binary_labels_list = [
        #         torch.zeros_like(gt_labels_list[i])
        #         for i in range(len(all_gt_labels_list))
        #     ]
        #     # TODO bug here
        #     enc_losses_pts, enc_losses_dir = self.loss_single(pts_targets,
        #                                                         pts_preds,
        #                                                         num_total_pos,
        #                                                         pts_weights,)
        #     loss_dict['enc_losses_pts'] = enc_losses_pts
        #     loss_dict['enc_losses_dir'] = enc_losses_dir

        # loss from the last decoder layer
        
        loss_dict = dict()
        loss_dict['manifold_loss_bbox'] = losses_bbox[-1]
        loss_dict['manifold_loss_iou'] = losses_iou[-1]
        loss_dict['manifold_loss_pts'] = losses_pts[-1]
        loss_dict['manifold_loss_dir'] = losses_dir[-1]
        loss_dict['manifold_loss_angle'] = losses_angle[-1]
        # loss from other decoder layers
        num_dec_layer = 0
        for loss_bbox_i, loss_iou_i, loss_pts_i, loss_dir_i, loss_angle_i in zip(losses_bbox[:-1],
                                           losses_iou[:-1],
                                           losses_pts[:-1],
                                           losses_dir[:-1],
                                           losses_angle[:-1]):
            loss_dict[f'd{num_dec_layer}.manifold_loss_bbox'] = loss_bbox_i
            loss_dict[f'd{num_dec_layer}.manifold_loss_iou'] = loss_iou_i
            loss_dict[f'd{num_dec_layer}.manifold_loss_pts'] = loss_pts_i
            loss_dict[f'd{num_dec_layer}.manifold_loss_dir'] = loss_dir_i
            loss_dict[f'd{num_dec_layer}.manifold_loss_angle'] = loss_angle_i
            num_dec_layer += 1
        return loss_dict

    @force_fp32(apply_to=('preds_dicts'))
    def get_bboxes(self, preds_dicts, img_metas, rescale=False):
        """Generate bboxes from bbox head predictions.
        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
            img_metas (list[dict]): Point cloud and image's meta info.
        Returns:
            list[dict]: Decoded bbox, scores and labels after nms.
        """
        # bboxes: xmin, ymin, xmax, ymax
        preds_dicts = self.bbox_coder.decode(preds_dicts)

        num_samples = len(preds_dicts)
        ret_list = []
        for i in range(num_samples):
            preds = preds_dicts[i]
            bboxes = preds['bboxes']
            # bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 5] * 0.5

            # code_size = bboxes.shape[-1]
            # bboxes = img_metas[i]['box_type_3d'](bboxes, code_size)
            scores = preds['scores']
            labels = preds['labels']
            pts = preds['pts']

            ret_list.append([bboxes, scores, labels, pts])

        return ret_list