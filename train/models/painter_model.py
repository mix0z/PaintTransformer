import torch
import numpy as np
from .base_model import BaseModel
from . import networks
from util import morphology
from scipy.optimize import linear_sum_assignment
from PIL import Image


def renderer(curve_points, locations, colors, widths, H, W, K, canvas_color='gray', dtype=torch.float32):
    if K <= 0:
        return torch.zeros((H, W, 3), device=curve_points.device, dtype=dtype), torch.zeros((H, W, 1), device=curve_points.device, dtype=dtype)
    N, S, _ = curve_points.shape
    t_H = torch.linspace(0., float(H), int(H // 5), device=curve_points.device)
    t_W = torch.linspace(0., float(W), int(W // 5), device=curve_points.device)
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
    t_H = torch.linspace(0., float(H), H, device=curve_points.device)
    t_W = torch.linspace(0., float(W), W, device=curve_points.device)
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
    return I, bs_mask


def sample_quadratic_bezier_curve(s, c, e, num_points=20, dtype=torch.float32):
    """
    Samples points from the quadratic bezier curves defined by the control points.
    Number of points to sample is num.
    Args:
    s (tensor): Start point of each curve, shape [N, 2].
    c (tensor): Control point of each curve, shape [N, 2].
    e (tensor): End point of each curve, shape [N, 2].
    num_points (int): Number of points to sample on every curve.
    Return:
    (tensor): Coordinates of the points on the Bezier curves, shape [N, num_points, 2]
    """
    N, _ = s.shape
    t = torch.linspace(0., 1., num_points, dtype=dtype, device=s.device)
    t = t.expand(N, num_points)
    s_x = s[..., 0].unsqueeze(1)
    s_y = s[..., 1].unsqueeze(1)
    e_x = e[..., 0].unsqueeze(1)
    e_y = e[..., 1].unsqueeze(1)
    c_x = c[..., 0].unsqueeze(1)
    c_y = c[..., 1].unsqueeze(1)
    x = c_x + (1. - t) ** 2 * (s_x - c_x) + t ** 2 * (e_x - c_x)
    y = c_y + (1. - t) ** 2 * (s_y - c_y) + t ** 2 * (e_y - c_y)
    return torch.stack([x, y], dim=-1)


def render_all(s, c, e, locations, colors, widths, H, W, K, canvas_color='gray', num_points=20, dtype=torch.float32):
    colors = colors * 255.0
    s = s * H
    c = c * H
    e = e * H
    locations = locations * H
    widths = widths * H

    curve_points = sample_quadratic_bezier_curve(s + locations, c + locations, e + locations, num_points, dtype)

    return renderer(curve_points, locations, colors, widths, H, W, K, canvas_color, dtype)


class PainterModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(dataset_mode='null')
        parser.add_argument('--used_strokes', type=int, default=1,
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
        self.d = 16  # s_x, s_y, e_x, e_y, c_x, c_y, x, y, w, r0, g0, b0, r1, g1, b1, a
        self.d_shape = 9

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
        self.patch_size = 30
        self.loss_pixel = torch.tensor(0., device=self.device)
        self.loss_gt = torch.tensor(0., device=self.device)
        self.loss_w = torch.tensor(0., device=self.device)
        self.loss_decision = torch.tensor(0., device=self.device)
        self.criterion_pixel = torch.nn.L1Loss().to(self.device)
        self.criterion_decision = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(opt.lambda_recall)).to(self.device)
        if self.isTrain:
            self.optimizer = torch.optim.Adam(self.net_g.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer)

    def param2stroke(self, param, H, W, decision, batch_size):
        K = self.opt.used_strokes
        print("K", K)
        # param: b, 12
        b = param.shape[0]
        param_list = torch.split(param, 1, dim=1)
        R0, G0, B0, R2, G2, B2, _ = param_list[9:]
        s_X, s_y, c_x, c_y, e_x, e_y, locations_x, locations_y, widths = [item.squeeze(-1) for item in param_list[:9]]

        s = torch.stack([s_X, s_y], dim=-1)
        c = torch.stack([c_x, c_y], dim=-1)
        e = torch.stack([e_x, e_y], dim=-1)
        locations = torch.stack([locations_x, locations_y], dim=-1)
        colors = torch.cat([R0, G0, B0], dim=-1)
        widths = widths.unsqueeze(-1)

        # s: b, 2
        # c: b, 2
        # e: b, 2
        # locations: b, 2
        # colors: b, 3
        # widths: b, 1
        # decision: b, 1
        # H: 1
        # W: 1
        # K: 1
        ans = torch.zeros(batch_size, H, W, 3, device=self.device)
        bs_mask = torch.zeros(batch_size, H, W, 1, device=self.device)
        for i in range(0, b, batch_size):
            indexes = decision[i:i + batch_size] > 0
            ans[i:i + batch_size], bs_mask[i:i + batch_size] = render_all(
                s[i:i + batch_size][indexes],
                c[i:i + batch_size][indexes],
                e[i:i + batch_size][indexes],
                locations[i:i + batch_size][indexes],
                colors[i:i + batch_size][indexes],
                widths[i:i + batch_size][indexes], H, W, min(K, indexes.sum().item()))

        return ans.view(batch_size, 3, H, W), bs_mask.view(batch_size, 1, H, W)

    def set_input(self, input_dict):
        self.image_paths = input_dict['A_paths']
        with torch.no_grad():
            old_param = torch.rand(self.opt.batch_size // 4, self.opt.used_strokes, self.d, device=self.device)
            old_param[:, :, :4] = old_param[:, :, :4] - 0.5

            left = torch.maximum(torch.zeros(old_param[:, :, :2].shape, device=self.device), old_param[:, :, :2] - 0.2 + 0.5)
            right = torch.maximum(torch.zeros(old_param[:, :, :2].shape, device=self.device), 0.5 - old_param[:, :, :2] - 0.2)
            len_gamma = left + right
            old_param[:, :, 4:6] = old_param[:, :, 4:6] * len_gamma
            old_param[:, :, 4:6] = torch.where(old_param[:, :, 4:6] < left, old_param[:, :, 4:6] - 0.5, old_param[:, :, 4:6] - len_gamma + 0.5)

            old_param[:, :, 6:9] = old_param[:, :, 6:9] * 0.5 + 0.2
            old_param[:, :, -4:-1] = old_param[:, :, -7:-4]
            old_param = old_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(old_param, self.patch_size * 2, self.patch_size * 2, torch.ones_like(old_param[:, 0]), self.opt.batch_size // 4)

            old = torch.zeros(self.opt.batch_size // 4, 3, self.patch_size * 2, self.patch_size * 2, device=self.device)
            old = foregrounds * alphas + old * (1 - alphas)
            old = old.view(self.opt.batch_size // 4, 3, 2, self.patch_size, 2, self.patch_size).contiguous()
            old = old.permute(0, 2, 4, 1, 3, 5).contiguous()
            self.old = old.view(self.opt.batch_size, 3, self.patch_size, self.patch_size).contiguous()

            gt_param = torch.rand(self.opt.batch_size, self.opt.used_strokes, self.d, device=self.device)
            gt_param[:, :, :4] = gt_param[:, :, :4] - 0.5

            left = torch.maximum(torch.zeros(gt_param[:, :, :2].shape, device=self.device), gt_param[:, :, :2] - 0.2 + 0.5)
            right = torch.maximum(torch.zeros(gt_param[:, :, :2].shape, device=self.device), 0.5 - gt_param[:, :, :2] - 0.2)
            len_gamma = left + right
            gt_param[:, :, 4:6] = gt_param[:, :, 4:6] * len_gamma
            gt_param[:, :, 4:6] = torch.where(gt_param[:, :, 4:6] < left, gt_param[:, :, 4:6] - 0.5,
                                               gt_param[:, :, 4:6] - len_gamma + 0.5)

            gt_param[:, :, 6:9] = gt_param[:, :, 6:9] * 0.5 + 0.2
            gt_param[:, :, -4:-1] = gt_param[:, :, -7:-4]
            self.gt_param = gt_param[:, :, :self.d_shape]
            gt_param = gt_param.view(-1, self.d).contiguous()
            foregrounds, alphas = self.param2stroke(gt_param, self.patch_size, self.patch_size, torch.ones_like(gt_param[:, 0]), self.opt.batch_size)

            self.render = self.old.clone()
            gt_decision = torch.ones(self.opt.batch_size, self.opt.used_strokes, device=self.device)

            self.render = foregrounds * alphas + self.render * (1 - alphas)
            self.gt_decision = gt_decision

    def forward(self):

        param, decisions = self.net_g(self.render, self.old)

        # stroke_param: b, stroke_per_patch, param_per_stroke
        # decision: b, stroke_per_patch, 1
        self.pred_decision = decisions.view(-1, self.opt.used_strokes).contiguous()
        self.pred_param = param[:, :, :self.d_shape]
        param = param.view(-1, self.d).contiguous()
        foregrounds, alphas = self.param2stroke(param, self.patch_size, self.patch_size, decisions.view(-1).contiguous(), self.opt.batch_size)
        decisions = networks.SignWithSigmoidGrad.apply(decisions.view(-1, self.opt.used_strokes, 1, 1, 1).contiguous())

        # foreground, alpha: b, stroke_per_patch, 3, output_size, output_size
        self.rec = self.old.clone()
        self.rec = foregrounds * alphas + self.rec * (1 - alphas)

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

    @staticmethod
    def bezier_quadratic_curve(t, p0, p1, p2):
        return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * p1 + t ** 2 * p2

    @staticmethod
    def length_quadratic_bezier_curve(p0, p1, p2, n=1000):
        def derivative(t):
            return 2 * (1 - t) * (p1 - p0) + 2 * t * (p2 - p1)

        def simpsons_rule(a, b, n, f):
            h = (b - a) / n
            s = f(a) + f(b)
            for i in range(1, n, 2):
                s += 4 * f(a + i * h)
            for i in range(2, n - 1, 2):
                s += 2 * f(a + i * h)
            return s * h / 3

        return simpsons_rule(0, 1, n, lambda t: torch.linalg.norm(derivative(t), dim=-1))

    def gaussian_w_distance(self, param_1, param_2):

        s_1, c_1, e_1, mu_1, w_1 = torch.split(param_1, (2, 2, 2, 2, 1), dim=-1)
        h_1 = self.length_quadratic_bezier_curve(s_1.view(-1, 2), c_1.view(-1, 2), e_1.view(-1, 2)).view(s_1.shape[:-1])
        theta_1 = torch.atan2(e_1.view(-1, 2)[:, 0] - s_1.view(-1, 2)[:, 0], e_1.view(-1, 2)[:, 1] - s_1.view(-1, 2)[:, 1]).view(s_1.shape[:-1])
        theta_1[theta_1 < 0] += 2 * torch.acos(torch.tensor(-1., device=param_1.device))
        print("theta_1", theta_1)
        print("h_1", h_1)
        # mu_1, w_1, h_1, theta_1 = torch.split(param_1, (2, 1, 1, 1), dim=-1)
        w_1 = w_1.squeeze(-1)
        # h_1 = h_1.squeeze(-1)
        # theta_1 = torch.acos(torch.tensor(-1., device=param_1.device)) * theta_1.squeeze(-1)
        trace_1 = (w_1 ** 2 + h_1 ** 2) / 4

        s_2, c_2, e_2, mu_2, w_2 = torch.split(param_2, (2, 2, 2, 2, 1), dim=-1)
        h_2 = self.length_quadratic_bezier_curve(s_2.view(-1, 2), c_2.view(-1, 2), e_2.view(-1, 2)).view(s_2.shape[:-1])
        theta_2 = torch.atan2(e_2.view(-1, 2)[:, 0] - s_2.view(-1, 2)[:, 0], e_2.view(-1, 2)[:, 1] - s_2.view(-1, 2)[:, 1]).view(s_2.shape[:-1])
        theta_2[theta_2 < 0] += 2 * torch.acos(torch.tensor(-1., device=param_2.device))
        print("theta_2", theta_2)
        print("h_2", h_2)
        # mu_2, w_2, h_2, theta_2 = torch.split(param_2, (2, 1, 1, 1), dim=-1)
        w_2 = w_2.squeeze(-1)
        # h_2 = h_2.squeeze(-1)
        # theta_2 = torch.acos(torch.tensor(-1., device=param_2.device)) * theta_2.squeeze(-1)
        trace_2 = (w_2 ** 2 + h_2 ** 2) / 4
        sigma_1_sqrt = self.get_sigma_sqrt(w_1, h_1, theta_1)
        sigma_2 = self.get_sigma(w_2, h_2, theta_2)
        trace_12 = torch.matmul(torch.matmul(sigma_1_sqrt, sigma_2), sigma_1_sqrt)
        print("trace_12", trace_12)
        trace_12 = torch.sqrt(trace_12[..., 0, 0] + trace_12[..., 1, 1] + 2 * torch.sqrt(
            trace_12[..., 0, 0] * trace_12[..., 1, 1] - trace_12[..., 0, 1] * trace_12[..., 1, 0]))
        print("trace_12", trace_12)
        print("trace_1", trace_1)
        print("trace_2", trace_2)
        print("mu_1", mu_1)
        print("mu_2", mu_2)
        print("sigma_1_sqrt", sigma_1_sqrt)
        print("sigma_2", sigma_2)
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
                print("PARAM", pred_param_broad, valid_gt_param_broad)
                cost_matrix_w = self.gaussian_w_distance(pred_param_broad, valid_gt_param_broad)
                decision = self.pred_decision[i]
                cost_matrix_decision = (1 - decision).unsqueeze(-1).repeat(1, valid_gt_param.shape[0])
                print(cost_matrix_l1.shape, cost_matrix_w.shape, cost_matrix_decision.shape)
                print(cost_matrix_l1.max(), cost_matrix_w.max(), cost_matrix_decision.max())
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
