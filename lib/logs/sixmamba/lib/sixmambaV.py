import torch
from torchvision.transforms.functional import rotate, resize
from torchvision.transforms import InterpolationMode
from .sixtransform import SixTransform
from .models_mamba import VisionMamba
import math
import torch.nn.functional as F
import torch.nn as nn


class HexMambaProcessorV3(nn.Module):

    def __init__(self, square_size=14, ch=1024):
        super().__init__()
        self.orig_h = square_size
        self.orig_w = square_size
        self.square_size = square_size  # token 高宽

        # Vision Mamba
        self.vim = VisionMamba(
             img_size=self.square_size,
             patch_size=7,
             embed_dim=384,
             depth=8,
             rms_norm=True,
             channels=ch,  # 支持任意输入通道
             stride=7
         ).to(device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu"))

    # ===============================
    # resize tensor [C,H,W] -> [C,H_out,W_out]
    # ===============================
    @staticmethod
    def resize_tensor(x, size):
        """
        x: [C,H,W] 或 [1,C,H,W]
        size: [H_out,W_out]
        """
        if x.ndim == 3:
            x = x.unsqueeze(0)  # [1,C,H,W]
        x_resized = F.interpolate(x, size=size, mode='bilinear', align_corners=False)
        return x_resized.squeeze(0)

    # ===============================
    # rotate tensor [C,H,W] 任意通道
    # ===============================
    @staticmethod
    def rotate_tensor(x, angle_deg):
        """
        x: [C,H,W]
        angle_deg: 顺时针旋转角度
        """
        C,H,W = x.shape
        theta = torch.tensor([
            [math.cos(math.radians(angle_deg)), -math.sin(math.radians(angle_deg)), 0.0],
            [math.sin(math.radians(angle_deg)),  math.cos(math.radians(angle_deg)), 0.0]
        ], dtype=torch.float32, device=x.device).unsqueeze(0)  # [1,2,3]

        x_batch = x.unsqueeze(0)  # [1,C,H,W]
        grid = F.affine_grid(theta, x_batch.size(), align_corners=False)
        rotated = F.grid_sample(x_batch, grid, mode='bilinear', padding_mode='zeros', align_corners=False)
        return rotated.squeeze(0)  # [C,H,W]

    # ===============================
    # forward
    # ===============================
    def forward(self, x):
        """
        x: [B,C,H,W] 任意通道
        output: [B,C,orig_h,orig_w]
        """
        B,C,H,W = x.shape
        angles = [0,60,120]
        outputs = []

        for b in range(B):
            img = x[b]
            restored_imgs = []

            for angle in angles:
                # --------------------------
                # 沙漏 mask + crop
                # --------------------------
                mask = SixTransform.Mask.create(H,W,angle=angle, device=img.device)
                masked = img * mask.unsqueeze(0)
                cropped = SixTransform.Crop.crop(masked, mask)
                crop_h, crop_w = cropped.shape[-2:]

                # --------------------------
                # resize -> square
                # --------------------------
                square = self.resize_tensor(cropped, [self.square_size, self.square_size])

                # --------------------------
                # run Vision Mamba
                # --------------------------
                #colored_square = self.vim(square.unsqueeze(0))
                colored_square = square.unsqueeze(0)
                if colored_square.ndim == 4:
                    colored_square = colored_square.squeeze(0)

                # --------------------------
                # square -> rect
                # --------------------------
                compressed = self.resize_tensor(colored_square, [crop_h, crop_w])
                if angle==0:
                    compressed = compressed.flip(1).flip(2)

                # --------------------------
                # rotate back
                # --------------------------
                rotated_back = self.rotate_tensor(compressed, angle)

                # --------------------------
                # tight crop
                # --------------------------
                valid_mask = rotated_back.sum(0) > 0
                if valid_mask.any():
                    ys,xs = valid_mask.nonzero(as_tuple=True)
                    rmin,rmax = ys.min(), ys.max()+1
                    cmin,cmax = xs.min(), xs.max()+1
                    rotated_back = rotated_back[:, rmin:rmax, cmin:cmax]

                restored_imgs.append(rotated_back)

            # --------------------------
            # merge hex
            # --------------------------
            max_h = max(im.shape[1] for im in restored_imgs)
            max_w = max(im.shape[2] for im in restored_imgs)
            merged_imgs = []
            for im in restored_imgs:
                C,H_im,W_im = im.shape
                canvas = torch.zeros(C,max_h,max_w,device=im.device)
                y0 = (max_h - H_im)//2
                x0 = (max_w - W_im)//2
                canvas[:,y0:y0+H_im,x0:x0+W_im] = im
                merged_imgs.append(canvas)
            merged = torch.stack(merged_imgs,0).max(0)[0]

            # --------------------------
            # center pad to original size
            # --------------------------
            C,H_m,W_m = merged.shape
            canvas = torch.zeros(C,self.orig_h,self.orig_w,device=merged.device)
            y0 = (self.orig_h - H_m)//2
            x0 = (self.orig_w - W_m)//2
            canvas[:,y0:y0+H_m,x0:x0+W_m] = merged

            outputs.append(canvas)

        return torch.stack(outputs,0)  # [B,C,orig_h,orig_w]