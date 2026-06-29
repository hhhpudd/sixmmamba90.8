# sixtransform.py
import torch
import math
import kornia


"""
sixTransform.Mask → 沙漏掩码生成和应用
SixTransform.Crop → 裁剪和旋转
"""


class SixTransform:

    _mask_cache = {}

    class Mask:
        """沙漏掩码生成与应用"""
        @staticmethod
        def _build_mask(H, W, device):
            """构建角度为0的基础沙漏掩码"""
            a = W / 2
            h_tri = (math.sqrt(3) / 2) * a
            center_x, center_y = W // 2, H // 2

            pts_up = torch.tensor([
                [center_x, center_y],
                [center_x + a/2, center_y - h_tri],
                [center_x - a/2, center_y - h_tri]
            ], dtype=torch.float32, device=device)

            pts_down = torch.tensor([
                [center_x, center_y],
                [center_x - a/2, center_y + h_tri],
                [center_x + a/2, center_y + h_tri]
            ], dtype=torch.float32, device=device)

            Y, X = torch.meshgrid(
                torch.arange(H, device=device),
                torch.arange(W, device=device),
                indexing='ij'
            )
            pts = torch.stack([X + 0.5, Y + 0.5], dim=-1)

            def point_in_triangle(pt, tri):
                A, B, C = tri
                v0 = C - A
                v1 = B - A
                v2 = pt - A
                dot00 = (v0 * v0).sum(-1)
                dot01 = (v0 * v1).sum(-1)
                dot02 = (v0 * v2).sum(-1)
                dot11 = (v1 * v1).sum(-1)
                dot12 = (v1 * v2).sum(-1)
                invDenom = 1 / (dot00 * dot11 - dot01 * dot01 + 1e-6)
                u = (dot11 * dot02 - dot01 * dot12) * invDenom
                v = (dot00 * dot12 - dot01 * dot02) * invDenom
                return (u >= 0) & (v >= 0) & (u + v <= 1)

            mask_up = point_in_triangle(pts, pts_up)
            mask_down = point_in_triangle(pts, pts_down)
            mask = mask_up | mask_down
            return mask.float()

        @staticmethod
        def create(H, W, angle=0, device='cpu'):
            """生成单个角度的沙漏掩码 tensor"""
            key = (H, W, angle)
            if key not in SixTransform._mask_cache:
                base_mask = SixTransform.Mask._build_mask(H, W, 'cpu')
                if angle != 0:
                    mask = kornia.geometry.transform.rotate(
                        base_mask.unsqueeze(0).unsqueeze(0),
                        torch.tensor([float(angle)]),
                        mode='nearest'
                    ).squeeze(0).squeeze(0)
                else:
                    mask = base_mask
                SixTransform._mask_cache[key] = mask
            mask = SixTransform._mask_cache[key]
            if mask.device.type != device:
                mask = mask.to(device)
            return mask

        @staticmethod
        def apply(img_tensor):
            """应用三个角度的沙漏mask，返回列表[B x C x H x W]"""
            B, C, H, W = img_tensor.shape
            device = img_tensor.device
            results = []
            for angle in [0, 60, 120]:
                mask = SixTransform.Mask.create(H, W, angle=angle, device=device)
                mask = mask.unsqueeze(0).unsqueeze(0).expand(B, C, H, W)
                results.append(img_tensor * mask)
            return results

    class Crop:
        """沙漏裁剪与旋转"""
        @staticmethod
        def get_corners(mask):
            H, W = mask.shape
            ys, xs = (mask > 0).nonzero(as_tuple=True)
            top_idx = ys.argmin()
            bottom_idx = ys.argmax()
            top = torch.tensor([xs[top_idx].item(), ys[top_idx].item()])
            bottom = torch.tensor([xs[bottom_idx].item(), ys[bottom_idx].item()])

            mid_y = (top[1] + bottom[1]) / 2
            mid_mask = ((ys > top[1]) & (ys < bottom[1]))
            mid_xs = xs[mid_mask]
            mid_ys = ys[mid_mask]

            if mid_xs.numel() == 0:
                left = torch.tensor([0, int(mid_y)])
                right = torch.tensor([W-1, int(mid_y)])
            else:
                left_idx = mid_xs.argmin()
                right_idx = mid_xs.argmax()
                left = torch.tensor([mid_xs[left_idx].item(), mid_ys[left_idx].item()])
                right = torch.tensor([mid_xs[right_idx].item(), mid_ys[right_idx].item()])
            return torch.stack([top, right, bottom, left], dim=0)

        @staticmethod
        def _rotate_expand(tensor, angle_deg, mode='bilinear'):
            """使用 kornia 旋转并自动扩展画布（等价 expand=True）"""
            if tensor.ndim == 2:
                tensor_4d = tensor.unsqueeze(0).unsqueeze(0)  # [1,1,H,W]
                was_2d = True
            elif tensor.ndim == 3:
                tensor_4d = tensor.unsqueeze(0)  # [1,C,H,W]
                was_2d = False
            else:
                tensor_4d = tensor
                was_2d = False

            B, C, H, W = tensor_4d.shape
            device = tensor_4d.device

            angle_rad = math.radians(angle_deg)
            cos_a = abs(math.cos(angle_rad))
            sin_a = abs(math.sin(angle_rad))
            new_H = int(H * cos_a + W * sin_a)
            new_W = int(H * sin_a + W * cos_a)

            alpha = math.cos(angle_rad)
            beta = math.sin(angle_rad)

            M = torch.tensor([[
                [alpha, -beta, W/2 - alpha * W/2 + beta * H/2],
                [beta, alpha, H/2 - beta * W/2 - alpha * H/2]
            ]], dtype=torch.float32, device=device)

            # 平移修正以适配扩展后的画布
            M[:, 0, 2] += new_W / 2 - W / 2
            M[:, 1, 2] += new_H / 2 - H / 2

            result = kornia.geometry.transform.warp_affine(
                tensor_4d, M, dsize=(new_H, new_W), mode=mode
            )

            if was_2d:
                result = result.squeeze(0).squeeze(0)
            else:
                result = result.squeeze(0)
            return result

        @staticmethod
        def crop(img_tensor, mask):
            """裁剪旋转后的沙漏区域，短边在下"""
            corners = SixTransform.Crop.get_corners(mask)
            tl, tr, br, bl = corners
            vec_diag = br - tl
            angle = math.degrees(-torch.atan2(vec_diag[1], vec_diag[0]))

            img_rot = SixTransform.Crop._rotate_expand(img_tensor, angle, mode='bilinear')
            mask_rot = SixTransform.Crop._rotate_expand(mask, angle, mode='nearest')

            ys, xs = (mask_rot > 0).nonzero(as_tuple=True)
            rmin, rmax = ys.min(), ys.max() + 1
            cmin, cmax = xs.min(), xs.max() + 1

            cropped = img_rot[:, rmin:rmax, cmin:cmax]
            C, h, w = cropped.shape
            if w > h:
                cropped = cropped.permute(0, 2, 1).flip(2)
            return cropped
