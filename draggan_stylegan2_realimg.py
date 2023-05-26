import torch 
import torch.nn.functional as functional
from training.networks_stylegan2 import Generator
import dnnlib
import numpy as np 
from PIL import Image 
import os 
import shutil 
import copy 

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

    def gen_image(self, seed, latent=None):
        label = torch.zeros([1, self.G.c_dim], device=self.device).to(self.device)
        if latent == None:
            z = torch.from_numpy(np.random.RandomState(seed).randn(1, self.G.z_dim)).to(self.device)
            latent = self.G.mapping(z, label, truncation_psi=1.0, truncation_cutoff=None, update_emas=False)
        img, feats = self.G.synthesis(latent, update_emas=False,  noise_mode="const")
        img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8)
        return img[0].cpu().numpy()

    def projector(self, target):
        assert target.shape == (self.G.img_channels, self.G.img_resolution, self.G.img_resolution)
        w_avg_samples = 10000
        num_steps = 1000
        initial_learning_rate = 0.1
        initial_noise_factor = 0.05
        noise_ramp_length = 0.75
        lr_rampdown_length = 0.25
        lr_rampup_length = 0.05
        regularize_noise_weight = 1e5


        G = copy.deepcopy(self.G).eval().requires_grad_(False).to(self.device) # type: ignore

        # Compute w stats.
        z_samples = np.random.RandomState(123).randn(w_avg_samples, G.z_dim)
        w_samples = G.mapping(torch.from_numpy(z_samples).to(self.device), None)  # [N, L, C]
        w_samples = w_samples[:, :1, :].cpu().numpy().astype(np.float32)       # [N, 1, C]
        w_avg = np.mean(w_samples, axis=0, keepdims=True)      # [1, 1, C]
        w_std = (np.sum((w_samples - w_avg) ** 2) / w_avg_samples) ** 0.5

        # Setup noise inputs.
        noise_bufs = { name: buf for (name, buf) in G.synthesis.named_buffers() if 'noise_const' in name }

        # Load VGG16 feature detector.
        url = 'https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/metrics/vgg16.pt'
        with dnnlib.util.open_url(url) as f:
            vgg16 = torch.jit.load(f).eval().to(self.device)

        # Features for target image.
        target_images = target.unsqueeze(0).to(self.device).to(torch.float32)
        if target_images.shape[2] > 256:
            target_images = functional.interpolate(target_images, size=(256, 256), mode='area')
        target_features = vgg16(target_images, resize_images=False, return_lpips=True)

        w_opt = torch.tensor(w_avg, dtype=torch.float32, device=self.device, requires_grad=True) # pylint: disable=not-callable
        w_out = torch.zeros([num_steps] + list(w_opt.shape[1:]), dtype=torch.float32, device=self.device)
        optimizer = torch.optim.Adam([w_opt] + list(noise_bufs.values()), betas=(0.9, 0.999), lr=initial_learning_rate)

        # Init noise.
        for buf in noise_bufs.values():
            buf[:] = torch.randn_like(buf)
            buf.requires_grad = True

        for step in range(num_steps):
            # Learning rate schedule.
            t = step / num_steps
            w_noise_scale = w_std * initial_noise_factor * max(0.0, 1.0 - t / noise_ramp_length) ** 2
            lr_ramp = min(1.0, (1.0 - t) / lr_rampdown_length)
            lr_ramp = 0.5 - 0.5 * np.cos(lr_ramp * np.pi)
            lr_ramp = lr_ramp * min(1.0, t / lr_rampup_length)
            lr = initial_learning_rate * lr_ramp
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # Synth images from opt_w.
            w_noise = torch.randn_like(w_opt) * w_noise_scale
            ws = (w_opt + w_noise).repeat([1, G.mapping.num_ws, 1])
            synth_images, feats = G.synthesis(ws, noise_mode='const')

            # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
            synth_images = (synth_images + 1) * (255/2)
            if synth_images.shape[2] > 256:
                synth_images = functional.interpolate(synth_images, size=(256, 256), mode='area')

            # Features for synth images.
            synth_features = vgg16(synth_images, resize_images=False, return_lpips=True)
            dist = (target_features - synth_features).square().sum()

            # Noise regularization.
            reg_loss = 0.0
            for v in noise_bufs.values():
                noise = v[None,None,:,:] # must be [1,1,H,W] for F.avg_pool2d()
                while True:
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=3)).mean()**2
                    reg_loss += (noise*torch.roll(noise, shifts=1, dims=2)).mean()**2
                    if noise.shape[2] <= 8:
                        break
                    noise = functional.avg_pool2d(noise, kernel_size=2)
            loss = dist + reg_loss * regularize_noise_weight

            # Step
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            print(f'step {step+1:>4d}/{num_steps}: dist {dist:<4.2f} loss {float(loss):<5.2f}')
            # Save projected W for each optimization step.
            w_out[step] = w_opt.detach()[0]

            # Normalize noise.
            with torch.no_grad():
                for buf in noise_bufs.values():
                    buf -= buf.mean()
                    buf *= buf.square().mean().rsqrt()
 
        return w_out.repeat([1, G.mapping.num_ws, 1])[-2:-1]



    def train(self, src_points, tar_points, M, seed=100, latent=None):
        label = torch.zeros([1, self.G.c_dim], device=self.device).to(self.device)
        if latent == None:
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
            res.append(img[0].cpu().numpy())
            res_points.append(src_points)
            if i % 10 == 0:
                print("L_motion:", L_motion.item(), "Drag points:", src_points, "target points:", tar_points)

        return res, res_points


if __name__ == "__main__":
    real_img_path = "lion.png"
    src_points = [np.array([281, 286]), np.array([279, 363])]
    tar_points = [np.array([359, 257]), np.array([356, 346])]
    # mask = points2mask(src_points, tar_points)
    # M = torch.tensor(mask[None, None], dtype=torch.float32).cuda()
    M = torch.ones([1, 1, 512, 512]).cuda()
    seed = 600
    draggan = DragGAN("stylegan2-afhqwild-512x512.pt", device="cuda")
    
    real_img = np.array(Image.open(real_img_path).resize([512, 512]))[..., :3]
    real_img = torch.tensor(real_img).permute(2, 0, 1)
    latent = draggan.projector(real_img)
    init_img = draggan.gen_image(seed=seed, latent=latent)
    Image.fromarray(np.uint8(init_img)).save("project.png")
    res, res_points = draggan.train(src_points, tar_points, M, seed=seed, latent=latent)
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


