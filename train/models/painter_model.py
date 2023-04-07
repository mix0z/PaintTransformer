import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image


def renderer(curve_points, locations, colors, widths, H, W, K, canvas_color='gray', dtype=torch.float32):
    N, S, _ = curve_points.shape
    t_H = torch.linspace(0., float(H), int(H // 5))
    t_W = torch.linspace(0., float(W), int(W // 5))
    t_H = t_H.to(dtype)
    t_W = t_W.to(dtype)
    P_y, P_x = torch.meshgrid(t_H, t_W)
    P = torch.stack([P_x, P_y], dim=-1)
    D_to_all_B_centers = torch.sum((P.unsqueeze(-2) - locations).square(), dim=-1)
    _, idcs = torch.topk(-D_to_all_B_centers, k=K)
    canvas_with_nearest_Bs = torch.index_select(curve_points, 0, idcs.flatten()).view(*idcs.shape, S, 2)
    canvas_with_nearest_Bs_colors = torch.index_select(colors, 0, idcs.flatten()).view(*idcs.shape, 3)
    canvas_with_nearest_Bs_bs = torch.index_select(widths, 0, idcs.flatten()).view(*idcs.shape, 1)
    H_, W_, r1, r2, r3 = canvas_with_nearest_Bs.shape
    canvas_with_nearest_Bs = canvas_with_nearest_Bs.repeat_interleave(H // H_, dim=0).repeat_interleave(W // W_,
                                                                                                        dim=1)
    H_, W_, r1, r2 = canvas_with_nearest_Bs_colors.shape
    canvas_with_nearest_Bs_colors = canvas_with_nearest_Bs_colors.repeat_interleave(H // H_,
                                                                                    dim=0).repeat_interleave(
        W // W_, dim=1)
    H_, W_, r1, r2 = canvas_with_nearest_Bs_bs.shape
    canvas_with_nearest_Bs_bs = canvas_with_nearest_Bs_bs.repeat_interleave(H // H_, dim=0).repeat_interleave(
        W // W_, dim=1)
    t_H = torch.linspace(0., float(H), H)
    t_W = torch.linspace(0., float(W), W)
    t_H = t_H.to(dtype)
    t_W = t_W.to(dtype)
    P_y, P_x = torch.meshgrid(t_H, t_W)
    P_full = torch.stack([P_x, P_y], dim=-1)
    canvas_with_nearest_Bs_a = canvas_with_nearest_Bs[:, :, :, :-1]
    canvas_with_nearest_Bs_b = canvas_with_nearest_Bs[:, :, :, 1:]
    canvas_with_nearest_Bs_b_a = canvas_with_nearest_Bs_b - canvas_with_nearest_Bs_a
    P_full_canvas_with_nearest_Bs_a = (P_full.unsqueeze(2).unsqueeze(2) - canvas_with_nearest_Bs_a)
    t = (canvas_with_nearest_Bs_b_a * P_full_canvas_with_nearest_Bs_a).sum(dim=-1) / (
            canvas_with_nearest_Bs_b_a.square().sum(dim=-1) + 1e-8)
    t = torch.clamp(t, 0.0, 1.0)
    closest_points_on_each_line_segment = canvas_with_nearest_Bs_a + t.unsqueeze(-1) * canvas_with_nearest_Bs_b_a
    dist_to_closest_point_on_line_segment = (
            P_full.unsqueeze(2).unsqueeze(2) - closest_points_on_each_line_segment).square().sum(dim=-1)
    min_values, _ = dist_to_closest_point_on_line_segment.min(dim=-1)
    min_values, _ = min_values.min(dim=-1)

    I_NNs_B_ranking = torch.softmax(100000. * (1.0 / (1e-8 + dist_to_closest_point_on_line_segment.min(dim=-1)[0])),
                                    dim=-1)  # [H, W, N]

    I_colors = torch.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_colors, I_NNs_B_ranking)  # [H, W, 3]

    # I_NNs_B_ranking = torch.softmax(100000. * (1.0 / (1e-8 + min_values)), dim=-1)
    # I_colors = (canvas_with_nearest_Bs_colors * I_NNs_B_ranking.unsqueeze(-1)).sum(dim=2)
    bs = torch.einsum('hwnf,hwn->hwf', canvas_with_nearest_Bs_bs, I_NNs_B_ranking)  # [H, W, 1]
    bs_mask = torch.sigmoid(bs - min_values.unsqueeze(-1))

    if canvas_color == 'gray':
        canvas = torch.ones_like(I_colors) * 0.5
    elif canvas_color == 'white':
        canvas = torch.ones_like(I_colors)
    elif canvas_color == 'black':
        canvas = torch.zeros_like(I_colors)
    elif canvas_color == 'noise':
        canvas = torch.normal(mean=0.0, std=0.1, size=I_colors.shape, dtype=dtype)

    I = I_colors * bs_mask + (1 - bs_mask) * canvas
    return I


class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=8,
                            help='actually generated strokes number')
        parser.add_argument('--num_blocks', type=int, default=3,
                            help='number of transformer blocks for stroke generator')
        parser.add_argument('--lambda_w', type=float, default=10.0, help='weight for w loss of stroke shape')
        parser.add_argument('--lambda_pixel', type=float, default=10.0, help='weight for pixel-level L1 loss')
        parser.add_argument('--lambda_gt', type=float, default=1.0, help='weight for ground-truth loss')
        parser.add_argument('--lambda_decision', type=float, default=10.0, help='weight for stroke decision loss')
        parser.add_argument('--lambda_recall', type=float, default=10.0, help='weight of recall for stroke decision loss')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['pixel', 'gt', 'w', 'decision']
        self.visual_names = ['old', 'render', 'rec']
        self.model_names = ['g']
        self.d = 12  # xc, yc, w, h, theta, R0, G0, B0, R2, G2, B2, A
        self.d_shape = 5

        def read_img(img_path, img_type='RGB'):
            img = Image.open(img_path).convert(img_type)
            img = np.array(img)
            if img.ndim == 2:
                img = np.expand_dims(img, axis=-1)
            img = img.transpose((2, 0, 1))
            img = torch.from_numpy(img).unsqueeze(0).float() / 255.
            return img

        brush_large_vertical = read_img('brush/brush_large_vertical.png', 'L').to(self.device)
        brush_large_horizontal = read_img('brush/brush_large_horizontal.png', 'L').to(self.device)
        self.meta_brushes = torch.cat(
            [brush_large_vertical, brush_large_horizontal], dim=0)
        net_g = networks.Painter(self.d_shape, opt.used_strokes, opt.ngf,
                                 n_enc_layers=opt.num_blocks, n_dec_layers=opt.num_blocks)
        self.net_g = networks.init_net(net_g, opt.init_type, opt.init_gain, self.gpu_ids)
        self.old = None
        self.render = None
        self.rec = None
        self.gt_param = None
        self.pred_param = None
        self.gt_decision = None
        self.pred_decision = None
        self.patch_size = 32
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def param2stroke(self, param, H, W):
        print('param shape', param.shape)
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        x0, y0, w, h, theta = [item.squeeze(-1) for item in param_list[:5]]
        R0, G0, B0, R2, G2, B2, _ = param_list[5:]
        print('R0', R0, 'G0', G0, 'B0', B0, 'R2', R2, 'G2', G2, 'B2', B2)
        sin_theta = torch.sin(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        cos_theta = torch.cos(torch.acos(torch.tensor(-1., device=param.device)) * theta)
        index = torch.full((b,), -1, device=param.device)
        index[h > w] = 0
        index[h <= w] = 1
        brush = self.meta_brushes[index.long()]
        alphas = torch.cat([brush, brush, brush], dim=1)
        alphas = (alphas > 0).float()
        t = torch.arange(0, brush.shape[2], device=param.device).unsqueeze(0) / brush.shape[2]
        color_map = torch.stack([R0 * (1 - t) + R2 * t, G0 * (1 - t) + G2 * t, B0 * (1 - t) + B2 * t], dim=1)
        color_map = color_map.unsqueeze(-1).repeat(1, 1, 1, brush.shape[3])
        brush = brush * color_map

        warp_00 = cos_theta / w
        warp_01 = sin_theta * H / (W * w)
        warp_02 = (1 - 2 * x0) * cos_theta / w + (1 - 2 * y0) * sin_theta * H / (W * w)
        warp_10 = -sin_theta * W / (H * h)
        warp_11 = cos_theta / h
        warp_12 = (1 - 2 * y0) * cos_theta / h - (1 - 2 * x0) * sin_theta * W / (H * h)
        warp_0 = torch.stack([warp_00, warp_01, warp_02], dim=1)
        warp_1 = torch.stack([warp_10, warp_11, warp_12], dim=1)
        warp = torch.stack([warp_0, warp_1], dim=1)
        grid = torch.nn.functional.affine_grid(warp, torch.Size((b, 3, H, W)), align_corners=False)
        brush = torch.nn.functional.grid_sample(brush, grid, align_corners=False)
        alphas = torch.nn.functional.grid_sample(alphas, grid, align_corners=False)
        print('brush shape', brush.shape)
        print('alphas shape', alphas.shape)
        return brush, alphas

    def set_input(self, input_dict):
        self.image_paths = input_dict['A_paths']
        with torch.no_grad():
            old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
            old_param[:, :, :4] = old_param[:, :, :4] * 0.5 + 0.2
            old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            old_param = old_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                           self.patch_size * 2).contiguous()
            alphas = alphas.view(self.opt.batch_size // 4, self.opt.used_strokes, 3, self.patch_size * 2,
                                 self.patch_size * 2).contiguous()
            old = torch.zeros(self.opt.batch_size // 4, 3, self.patch_size * 2, self.patch_size * 2, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                old = foreground * alpha + old * (1 - alpha)
            old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
            old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
            self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()

            gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
            gt_param[:, :, :4] = gt_param[:, :, :4] * 0.5 + 0.2
            gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size)
            foregrounds = morphology.Dilation2d(m=1)(foregrounds)
            alphas = morphology.Erosion2d(m=1)(alphas)
            foregrounds = foregrounds.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                           self.patch_size).contiguous()
            alphas = alphas.view(self.opt.batch_size, self.opt.used_strokes, 3, self.patch_size,
                                 self.patch_size).contiguous()
            self.render = self.old.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)
            for i in range(self.opt.used_strokes):
                foreground = foregrounds[:, i, :, :, :]
                alpha = alphas[:, i, :, :, :]
                for j in range(i):
                    iou = (torch.sum(alpha * alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5) / (
                            torch.sum(alphas[:, j, :, :, :], dim=(-3, -2, -1)) + 1e-5)
                    gt_decision[:, i] = ((iou < 0.75) | (~gt_decision[:, j].bool())).float() * gt_decision[:, i]
                decision = gt_decision[:, i].view(self.opt.batch_size, 1, 1, 1).contiguous()
                self.render = foreground * alpha * decision + self.render * (1 - alpha * decision)
            self.gt_decision = gt_decision

    def forward(self):
        param, decisions = self.net_g(self.render, self.old)
        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape]
        param = param.view(-1, self.d).contiguous()
        foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size)
        foregrounds = morphology.Dilation2d(m=1)(foregrounds)
        alphas = morphology.Erosion2d(m=1)(alphas)
        # foreground, alpha: b * stroke_per_patch, 3, output_size, output_size
        foregrounds = foregrounds.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        alphas = alphas.view(-1, self.opt.used_strokes, 3, self.patch_size, self.patch_size)
        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())
        self.rec = self.old.clone()
        for j in range(foregrounds.shape[1]):
            foreground = foregrounds[:, j, :, :, :]
            alpha = alphas[:, j, :, :, :]
            decision = decisions[:, j, :, :, :]
            self.rec = foreground * alpha * decision + self.rec * (1 - alpha * decision)

    @staticmethod
    def get_sigma_sqrt(w, h, theta):
        sigma_00 = w * (torch.cos(theta) ** 2) / 2 + h * (torch.sin(theta) ** 2) / 2
        sigma_01 = (w - h) * torch.cos(theta) * torch.sin(theta) / 2
        sigma_11 = h * (torch.cos(theta) ** 2) / 2 + w * (torch.sin(theta) ** 2) / 2
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    @staticmethod
    def get_sigma(w, h, theta):
        sigma_00 = w * w * (torch.cos(theta) ** 2) / 4 + h * h * (torch.sin(theta) ** 2) / 4
        sigma_01 = (w * w - h * h) * torch.cos(theta) * torch.sin(theta) / 4
        sigma_11 = h * h * (torch.cos(theta) ** 2) / 4 + w * w * (torch.sin(theta) ** 2) / 4
        sigma_0 = torch.stack([sigma_00, sigma_01], dim=-1)
        sigma_1 = torch.stack([sigma_01, sigma_11], dim=-1)
        sigma = torch.stack([sigma_0, sigma_1], dim=-2)
        return sigma

    def gaussian_w_distance(self, param_1, param_2):
        mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        h_1 = h_1.squeeze(-1)
        theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4
        mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        h_2 = h_2.squeeze(-1)
        theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        return torch.sum((mu_1 - mu_2) ** 2, dim=-1) + trace_1 + trace_2 - 2 * trace_12

    def optimize_parameters(self):
        self.forward()
        self.loss_pixel = self.criterion_pixel(self.rec, self.render) * self.opt.lambda_pixel
        cur_valid_gt_size = 0
        with torch.no_grad():
            r_idx = []
            c_idx = []
            for i in range(self.gt_param.shape[0]):
                is_valid_gt = self.gt_decision[i].bool()
                valid_gt_param = self.gt_param[i, is_valid_gt]
                cost_matrix_l1 = torch.cdist(self.pred_param[i], valid_gt_param, p=1)
                pred_param_broad = self.pred_param[i].unsqueeze(1).contiguous().repeat(
                    1, valid_gt_param.shape[0], 1)
                valid_gt_param_broad = valid_gt_param.unsqueeze(0).contiguous().repeat(
                    self.pred_param.shape[1], 1, 1)
                cost_matrix_w = self.gaussian_w_distance(pred_param_broad, valid_gt_param_broad)
                decision = self.pred_decision[i]
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0])
                r, c = linear_sum_assignment((cost_matrix_l1 + cost_matrix_w + cost_matrix_decision).cpu())
                r_idx.append(torch.tensor(r + self.pred_param.shape[1] * i, device=self.device))
                c_idx.append(torch.tensor(c + cur_valid_gt_size, device=self.device))
                cur_valid_gt_size += valid_gt_param.shape[0]
            r_idx = torch.cat(r_idx, dim=0)
            c_idx = torch.cat(c_idx, dim=0)
            paired_gt_decision = torch.zeros(self.gt_decision.shape[0] * self.gt_decision.shape[1], device=self.device)
            paired_gt_decision[r_idx] = 1.
        all_valid_gt_param = self.gt_param[self.gt_decision.bool(), :]
        all_pred_param = self.pred_param.view(-1, self.pred_param.shape[2]).contiguous()
        all_pred_decision = self.pred_decision.view(-1).contiguous()
        paired_gt_param = all_valid_gt_param[c_idx, :]
        paired_pred_param = all_pred_param[r_idx, :]
        self.loss_gt = self.criterion_pixel(paired_pred_param, paired_gt_param) * self.opt.lambda_gt
        self.loss_w = self.gaussian_w_distance(paired_pred_param, paired_gt_param).mean() * self.opt.lambda_w
        self.loss_decision = self.criterion_decision(all_pred_decision, paired_gt_decision) * self.opt.lambda_decision
        loss = self.loss_pixel + self.loss_gt + self.loss_w + self.loss_decision
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
