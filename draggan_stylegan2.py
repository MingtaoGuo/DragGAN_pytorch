import torch 
import torch.nn.functional as functional
from training.networks_stylegan2 import Generator
import numpy as np 
from PIL import Image 
import os 
import shutil 


def points2mask(src_points, tar_points):
    points = src_points + tar_points
    x_min = 512
    x_max = 0 
    y_min = 512
    y_max = 0
    for p in points:
        if p[0] > x_max:
            x_max = p[0]
        if p[0] < x_min:
            x_min = p[0]
        if p[1] > y_max:
            y_max = p[1]
        if p[1] < y_min:
            y_min = p[1]
    c_x, c_y = int((x_min + x_max) / 2), int((y_min + y_max) / 2)
    r = int(np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2) / 2)

    mask = np.zeros([512, 512])
    mask[c_y-r:c_y+r, c_x-r:c_x+r] = np.ones([2*r, 2*r])

    return mask 

def omega_p_r1(point, r1=3):
    x0, y0 = point[0], point[1]
    qs = []
    for x in range(max(int(x0-r1), 0), min(int(x0+r1), 512)):
        y_max = y0 + np.sqrt(r1 ** 2 - (x - x0) ** 2)
        y_min = y0 - np.sqrt(r1 ** 2 - (x - x0) ** 2)
        for y in range(int(y_min), int(y_max)):
            qs.append(np.array([x, y]))
    return qs

def omega_p_r2(point, r2=12):
    x0, y0 = point[0], point[1]
    qs = []
    for x in range(max(int(x0-r2), 0), min(int(x0+r2), 512)):
        for y in range(max(int(y0-r2), 0), min(int(y0+r2), 512)):
            qs.append(np.array([x, y]))
    return qs

def bilinear(point, feature):
    x, y = point[0], point[1]
    x1, x2 = int(x - 1), int(x + 1)
    y2, y1 = int(y - 1), int(y + 1)
    
    f_q11 = feature[..., y1, x1]
    f_q12 = feature[..., y2, x1]
    f_q21 = feature[..., y1, x2]
    f_q22 = feature[..., y2, x2]
    f_R1 = (x2 - x) / (x2 - x1) * f_q11 + (x - x1) / (x2 - x1) * f_q21
    f_R2 = (x2 - x) / (x2 - x1) * f_q12 + (x - x1) / (x2 - x1) * f_q22
    f_P = (y2 - y) / (y2 - y1) * f_R1 + (y - y1) / (y2 - y1) * f_R2
    return f_P

def motion_supervision(src_points, tar_points, F, M, F0, r1=3, lambd=20):
    F = functional.interpolate(F, [512, 512], mode="bilinear")
    F0 = functional.interpolate(F0, [512, 512], mode="bilinear")
    L_motion = 0
    for src_p, tar_p in zip(src_points, tar_points):
        if np.sqrt(np.sum(np.square(src_p - tar_p))) != 0:
            d = (tar_p - src_p) / np.sqrt(np.sum(np.square(src_p - tar_p)))
            qs = omega_p_r1(src_p, r1=r1)
            for q in qs:
                F_q = F[..., int(q[1]), int(q[0])]
                F_q_d = bilinear(q + d, F)
                L_motion += torch.mean(torch.abs(F_q.detach() - F_q_d))
    L_motion += torch.mean(torch.abs(F - F0) * (1 - M)) * lambd
    return L_motion
    
def point_tracking(src_points, F, F0, src_points_0, r2=12):
    F = functional.interpolate(F, [512, 512], mode="bilinear")
    F0 = functional.interpolate(F0, [512, 512], mode="bilinear")
    best_q = []
    for src_p, src_p_0 in zip(src_points, src_points_0):
        f_i = F0[..., src_p_0[1], src_p_0[0]] 
        qs = omega_p_r2(src_p, r2=r2)
        dist_min = np.inf
        for q in qs:
            F_q = F[..., int(q[1]), int(q[0])]
            dist = torch.mean(torch.abs(F_q - f_i))
            if dist < dist_min:
                dist_min = dist
                q_min = q
        best_q.append(q_min)
    return best_q


class DragGAN:
    def __init__(self, ckpt, device) -> None:
        self.G = Generator(z_dim=512, c_dim=0, w_dim=512, img_resolution=512, img_channels=3).to(device)
        self.G.load_state_dict(torch.load(ckpt))
        self.device = device

    def gen_image(self, seed):
        label = torch.zeros([1, self.G.c_dim], device=self.device).to(self.device)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
        latent = self.G.mapping(z, label, truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
        img, feats = self.G.synthesis(latent, update_emas=False,  noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

    def train(self, src_points, tar_points, M, seed=100, yiled_result=False):
        label = torch.zeros([1, self.G.c_dim], device=self.device).to(self.device)
        z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
        latent = self.G.mapping(z, label, truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
        latent_trainable = latent[:, :12, :].detach().clone().requires_grad_(True)
        latent_untrainable = latent[:, 12:, :].detach().clone().requires_grad_(False)
        opt = torch.optim.Adam([latent_trainable], lr=2e-3)

        res = []
        res_points = []
        for i in range(200):
            latent = torch.cat([latent_trainable, latent_untrainable], dim=1)
            if i < 1:
                img, feats = self.G.synthesis(latent, update_emas=False,  noise_mode="const")
                F = feats[6]
                F0 = feats[6].detach()
                src_points_0 = src_points
                L_motion = motion_supervision(src_points, tar_points, F, M, F0, r1=3, lambd=20)
                opt.zero_grad()
                L_motion.backward()
                opt.step()
            else:
                img, feats = self.G.synthesis(latent, update_emas=False,  noise_mode="const")
                F = feats[6]
                src_points = point_tracking(src_points, F, F0, src_points_0, r2=12)
                L_motion = motion_supervision(src_points, tar_points, F, M, F0, r1=3, lambd=20)
                opt.zero_grad()
                L_motion.backward()
                opt.step()
                dist = 0
                for sp, tp in zip(src_points, tar_points):
                    dist += np.sqrt(np.sum(np.square(sp - tp)))
                if dist < 5:
                    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
                    res.append(img[0].cpu().numpy())
                    break

            img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
            if yiled_result:
                yield (img[0].cpu().numpy(), src_points)
            else:
                res.append(img[0].cpu().numpy())
                res_points.append(src_points)
            if i % 10 == 0:
                print("L_motion:", L_motion.item(), "Drag points:", src_points, "target points:", tar_points)

        return res, res_points


if __name__ == "__main__":
    src_points = [np.array([192, 227]), np.array([321, 221])]
    tar_points = [np.array([192, 216]), np.array([321, 211])]
    # mask = points2mask(src_points, tar_points)
    # M = torch.tensor(mask[None, None], dtype=torch.float32).cuda()
    M = torch.ones([1, 1, 512, 512]).cuda()
    seed = 600
    draggan = DragGAN("stylegan2-ffhq-512x512.pt", device="cuda")
    init_img = draggan.gen_image(seed=seed)
    Image.fromarray(np.uint8(init_img)).save("init_img.png")
    res, res_points = draggan.train(src_points, tar_points, M, seed=seed)
    if os.path.exists("./results"):
        shutil.rmtree("./results")
    os.mkdir("./results")
    for idx, (img, point) in enumerate(zip(res, res_points)):
        for p, t in zip(point, tar_points):
            red_patch = np.zeros([6, 6, 3])
            red_patch[..., 0] = np.ones([6, 6]) * 255
            blue_patch = np.zeros([6, 6, 3])
            blue_patch[..., 2] = np.ones([6, 6]) * 255

            img[p[1]-3:p[1]+3, p[0]-3:p[0]+3] = red_patch
            img[t[1]-3:t[1]+3, t[0]-3:t[0]+3] = blue_patch
        Image.fromarray(np.uint8(img)).save(f"./results/{idx+1}.png")
    os.system(f"ffmpeg -r 24 -i results/%1d.png -pix_fmt yuv420p -c:v libx264 {seed}.mp4")



