import torch
import cv2
import kornia
import numpy as np
import math

# Torch实现的极坐标算子与逆算子
class CartToPolarTensor(object):
    def __init__(self,  radius, img_size=24):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # center, radius
        self.mapx, self.mapy = self.build_map((img_size//2, img_size//2), max_radius=radius)
        self.mapx_tensor = torch.tensor(self.mapx).unsqueeze(0).to(device)
        self.mapy_tensor = torch.tensor(self.mapy).unsqueeze(0).to(device)

    def build_map(self, center=(192, 192), max_radius=192):
        w = max_radius
        h = np.round(max_radius * np.pi).astype(int)
        dsize = (h, w)

        mapx = np.zeros(dsize, dtype=np.float32)
        mapy = np.zeros(dsize, dtype=np.float32)

        Kangle = (2 * np.pi) / h

        rhos = np.zeros((w,))
        Kmag = max_radius / w
        for rho in range(0, w):
            rhos[rho] = rho * Kmag

        for phi in range(0, h):
            KKy = Kangle * phi
            cp = np.cos(KKy)
            sp = np.sin(KKy)
            for rho in range(0, w):
                x = rhos[rho] * cp + center[1]
                y = rhos[rho] * sp + center[0]
                mapx[phi, rho] = x
                mapy[phi, rho] = y

        return mapx, mapy

    def __call__(self, img_tensor):
        polar = kornia.geometry.transform.remap(img_tensor, self.mapx_tensor, self.mapy_tensor)
        return polar

class PolarToCartTensor(object):
    def __init__(self, radius, img_size=24):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        # dsize还原的原始尺寸
        # maxax_radius=半径
        # center=圆心
        # src_size=现有的特征图尺寸
        # 圆不满时会补0（黑色的）
        self.mapx, self.mapy = self.build_map(dsize=(img_size, img_size),
                                              max_radius=radius,
                                              center=(img_size//2, img_size//2),
                                              src_size=(round(radius*3.14), radius))
        self.mapx_tensor = torch.tensor(self.mapx).unsqueeze(0).to(device)
        self.mapy_tensor = torch.tensor(self.mapy).unsqueeze(0).to(device)

    def build_map(self, dsize=(384, 384), max_radius=192, center=(192, 192), src_size=(603, 192)):
        w = dsize[1]
        h = dsize[0]

        angle_border = 1

        ssize_w = src_size[1]
        ssize_h = src_size[0] - 2 * angle_border

        mapx = np.zeros(dsize, dtype=np.float32)
        mapy = np.zeros(dsize, dtype=np.float32)

        Kangle = 2 * np.pi / ssize_h
        Kmag = max_radius / ssize_w

        bufx = np.zeros(w, dtype=np.float32)
        bufy = np.zeros(w, dtype=np.float32)

        for x in range(0, w):
            bufx[x] = x - center[1]

        for y in range(0, h):
            for x in range(0, w):
                bufy[x] = y - center[0]
            bufp, bufa = cv2.cartToPolar(bufx, bufy, angleInDegrees=False)
            for x in range(0, w):
                rho = bufp[x] / Kmag
                phi = bufa[x] / Kangle
                mapx[y, x] = rho
                mapy[y, x] = phi + angle_border

        return mapx, mapy

    def __call__(self, img_tensor):
        cart = kornia.geometry.transform.remap(img_tensor, self.mapx_tensor, self.mapy_tensor)
        return cart