import torch
import torch.nn.functional as functional
from training.networks_stylegan2 import Generator
import numpy as np
from PIL import Image
import os
import shutil
import gradio as gr
import cv2

from draggan_stylegan2 import DragGAN as DragGAN2

device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')
draggan = DragGAN2("stylegan2-ffhq-512x512.pt", device)
draggan.ckpt_name = "stylegan2-ffhq-512x512.pt"


def reload_ckpt(self, ckpt_name):
  if ckpt_name != self.ckpt_name:
    self.G.load_state_dict(torch.load(ckpt_name))
    self.ckpt_name = ckpt_name


draggan.reload_ckpt = reload_ckpt.__get__(draggan)


def drag_init(ckpt_name, seed, init_img):
  draggan.src_points = []
  draggan.tar_points = []
  if init_img is None:
    draggan.reload_ckpt(ckpt_name)
    init_img = draggan.gen_image(seed=int(seed))
    return Image.fromarray(np.uint8(init_img))


def select_handler(image, evt: gr.EventData):
  index = np.array(evt._data['index']).astype(np.int64)
  color = [(255, 0, 0), (0, 255, 0), (0, 0, 255)][len(draggan.tar_points) % 3]
  img = np.array(image)
  if len(draggan.src_points) > len(draggan.tar_points):
    draggan.tar_points.append(index)
    cv2.arrowedLine(img, (draggan.src_points[-1]), index, color, 1)
  else:
    draggan.src_points.append(index)
    cv2.circle(img, index, 0, color, thickness=1)
  return Image.fromarray(img)


def drag_inference(ckpt_name, seed, progress=gr.Progress()):
  draggan.reload_ckpt(ckpt_name)
  M = torch.ones([1, 1, 512, 512]).cuda()
  progress(0, total=200)

  src_points = draggan.src_points
  tar_points = draggan.tar_points
  # mask = points2mask(src_points, tar_points)
  # M = torch.tensor(mask[None, None], dtype=torch.float32).cuda()
  # res, res_points = draggan.train(src_points, tar_points, M, seed=seed)
  if os.path.exists("./results"):
    shutil.rmtree("./results")
  os.mkdir("./results")
  for idx, (img, point) in progress.tqdm(
      enumerate(
          draggan.train(src_points,
                        tar_points,
                        M,
                        seed=seed,
                        yiled_result=True))):
    for p, t in zip(point, tar_points):
      red_patch = np.zeros([6, 6, 3])
      red_patch[..., 0] = np.ones([6, 6]) * 255
      blue_patch = np.zeros([6, 6, 3])
      blue_patch[..., 2] = np.ones([6, 6]) * 255

      img[p[1] - 3:p[1] + 3, p[0] - 3:p[0] + 3] = red_patch
      img[t[1] - 3:t[1] + 3, t[0] - 3:t[0] + 3] = blue_patch
    tmp = Image.fromarray(np.uint8(img))
    tmp.save(f"./results/{idx+1}.png")
  os.system(
      f"ffmpeg -r 24 -i results/%1d.png -pix_fmt yuv420p -c:v libx264 results/{seed}.mp4"
  )
  return f'results/{seed}.mp4'


css = ".image-preview {height: auto !important;}"

with gr.Blocks(css=css) as demo:
  title = gr.Markdown('# DragGAN pytorch')
  mkdown = gr.Markdown(
      '''Re-implementation of [Drag Your GAN](https://vcai.mpi-inf.mpg.de/projects/DragGAN/) by [cutout.pro team](https://cutout.pro). To use the demo, follow the steps as below:
1. Select a model
2. Initialize a image
3. Pick the handle points and target points by simply clicking on the image
4. Generate
5. Review the animated results by pressing the "play" button.

Support of real world image is to be added.</br></br>

refer to: [Github](https://github.com/MingtaoGuo/DragGAN_pytorch)</br>
Special thanks to the original authors of DragGAN, & </br>
[MingtaoGuo](https://github.com/MingtaoGuo) & [cutout.pro](https://cutout.pro) by LibAI (formerly known as picup.ai).

''')

  ckpt_name = gr.Dropdown(['stylegan2-ffhq-512x512.pt'],
                          value='stylegan2-ffhq-512x512.pt',
                          label='Model name')
  seed = gr.Slider(1, 1 << 15, value=42, step=1, label='Seed')
  with gr.Row():
    reset = gr.Button(value='Reset')
    init = gr.Button(value='Init image')
    start = gr.Button(value='Generate')
  with gr.Row():
    image = gr.Image().style(width=512, height=512)
    # with gr.Column():
    video = gr.Video(label='Result')

  image.select(select_handler, inputs=[image], outputs=[image])
  reset.click(lambda x: None, image, image)
  init.click(drag_init, inputs=[ckpt_name, seed, image], outputs=[image])
  start.click(drag_inference, inputs=[ckpt_name, seed], outputs=[video])

  demo.queue(1).launch(share=True)