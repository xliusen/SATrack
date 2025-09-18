from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate

##################################################################
# from lib.utils.nt_xent import NTXentLoss
from lib.utils.bdirectional_attention_contrastive_loss import BidirectionalAttentionContrastiveLoss


class OSTrackActor(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

        self.device = self._get_device()
        # self.temperature = 0.5
        # self.use_cosine_similarity = True
        # self.nt_xent_criterion = NTXentLoss(self.device, self.bs, self.temperature, self.use_cosine_similarity)
        # self.margin = 1.0
        self.bdirectional_attention_contrastive_loss = BidirectionalAttentionContrastiveLoss(self.device, dim=768, top_k=self.bs-1)

    def _get_device(self):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("\nRunning on:", device)

        if device == 'cuda':
            device_name = torch.cuda.get_device_name()
            print("The device name is:", device_name)
            cap = torch.cuda.get_device_capability(device=None)
            print("The capability of this device is:", cap, '\n')
        return device

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        template_list = []
        search_list = []
        for i in range(len(data['template_images'])):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        for i in range(self.settings.num_search):
            search_img_i = data['search_images'][i].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
            # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)
            search_list.append(search_img_i)


        box_mask_z = []
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            # for i in range(len(template_list)):
            #     box_mask_z.append(generate_mask_cond(self.cfg, template_list[i].shape[0], template_list[i].device,
            #                                          data['template_anno'][i]))
            # box_mask_z = torch.cat(box_mask_z, dim=1)
            box_mask_z = None

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                            total_epochs=ce_start_epoch + ce_warm_epoch,
                                            ITERS_PER_EPOCH=1,
                                            base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        # if len(template_list) == 1:
        #     template_list = template_list[0]

        ###################################################################
        # phrase_ids = data['phrase_ids'].permute(1, 0)  # 40*32-->32*40
        # phrase_attnmask = data['phrase_attnmask'].permute(1, 0)  # 40*32-->32*40


        out_dict = self.net(template=template_list,  # 32*3*128*128
                            search=search_list,
                            language=data['language'],
                            # phrase_ids=phrase_ids,  # 32*40
                            # phrase_attnmask=phrase_attnmask,  # 32*40
                            ce_template_mask=box_mask_z,  # 32*64
                            ce_keep_rate=ce_keep_rate,  # 1
                            return_last_attn=False)

        return out_dict

    # def triplet_loss(self, anchor, positive, negative, margin):
    #     pos_dist = F.pairwise_distance(anchor, positive)
    #     neg_dist = F.pairwise_distance(anchor, negative)
    #     loss = F.relu(pos_dist - neg_dist + margin)
    #     return loss.mean()

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # currently only support the type of pred_dict is list
        assert isinstance(pred_dict, list)
        loss_dict = {}
        total_status = {}
        total_loss = torch.tensor(0., dtype=torch.float).cuda()  # 定义 0 tensor，并指定GPU设备

        # generate gt gaussian map
        gt_gaussian_maps_list = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE,
                                                 self.cfg.MODEL.BACKBONE.STRIDE)

        for i in range(len(pred_dict)):
            # get GT
            gt_bbox = gt_dict['search_anno'][i]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
            gt_gaussian_maps = gt_gaussian_maps_list[i].unsqueeze(1)

            # Get boxes
            pred_boxes = pred_dict[i]['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                               max=1.0)
            # (B,4) --> (B,1,4) --> (B,N,4)

            # compute giou and iou
            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            except:
                giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
            loss_dict['giou'] = giou_loss

            # compute l1 loss
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
            loss_dict['l1'] = l1_loss

            # compute location loss
            if 'score_map' in pred_dict[i]:
                location_loss = self.objective['focal'](pred_dict[i]['score_map'], gt_gaussian_maps)
            else:
                location_loss = torch.tensor(0.0, device=l1_loss.device)
            loss_dict['focal'] = location_loss

            # weighted sum
            loss = sum(loss_dict[k] * self.loss_weight[k] for k in loss_dict.keys() if k in self.loss_weight)
            total_loss += loss

            ###########################################################################################
            ## Multi-Modal Alignment
            # 语义感知的困难负样本对比对齐方法（Semantically-Aware Hard Negative Contrastive Alignment, SA-HNCA）
            # sema_loss = 1/2(search_loss+Temp_loss)
            search_language_loss = self.bdirectional_attention_contrastive_loss(pred_dict[i]['vision_x_vectors'],pred_dict[i]['language_vectors'])
            template_language_loss = 0
            for j in range(len(pred_dict[i]['vision_z_vectors'])):
                template_language_loss += self.bdirectional_attention_contrastive_loss(pred_dict[i]['vision_z_vectors'][j], pred_dict[i]['language_vectors'])
            template_language_loss = template_language_loss / len(pred_dict[i]['vision_z_vectors'])
            sema_loss = (search_language_loss + template_language_loss) * 0.01
            total_loss += sema_loss
            ##########################################################################################

            if return_status:
                # status for log
                status = {}

                mean_iou = iou.detach().mean()
                status = {f"{i}frame_Loss/total": total_loss.item(),
                          f"{i}frame_Loss/sema_loss": sema_loss.item(),
                          # f"{i}frame_Loss/ima": loss_ima.item(),
                          f"{i}frame_Loss/giou": giou_loss.item(),
                          f"{i}frame_Loss/l1": l1_loss.item(),
                          f"{i}frame_Loss/location": location_loss.item(),
                          f"{i}frame_IoU": mean_iou.item()}

                total_status.update(status)

        if return_status:
            return total_loss, total_status
        else:
            return total_loss
