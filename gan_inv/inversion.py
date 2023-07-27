import math
import os
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(current)
sys.path.append(parent)
from viz import renderer
import torch
from torch import optim
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
import dataclasses
import dnnlib
from .lpips import util
import imageio
import pickle
from training.networks_stylegan2 import Generator
# import 




def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)

    return initial_lr * lr_ramp





def make_image(tensor):
    return (
        tensor.detach()
        .clamp_(min=-1, max=1)
        .add(1)
        .div_(2)
        .mul(255)
        .type(torch.uint8)
        .permute(0, 2, 3, 1)
        .to("cpu")
        .numpy()
    )


@dataclasses.dataclass
class InverseConfig:
    lr_warmup = 0.05
    lr_decay = 0.25
    lr = 0.001
    noise = 0.05
    noise_decay = 0.75
    # step = 1000
    noise_regularize = 1e5
    mse = 0.1



def inverse_image(
    g_ema,
    image,
    percept,
    image_size=256,
    w_plus = False,
    config=InverseConfig(),
    device='cuda:0',
    step = 1000,
    n_mean_latent = 10000
):
    args = config

    # model_style = 

    # n_mean_latent = 10000

    resize = min(image_size, 256)

    if torch.is_tensor(image)==False:

        transform = transforms.Compose(
            [
                transforms.Resize(resize,),
                transforms.CenterCrop(resize),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        img = transform(image)

    else:
        img = transforms.functional.resize(image,resize)
        transform = transforms.Compose(
            [
                transforms.CenterCrop(resize),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        img = transform(img)

    imgs = []
    imgs.append(img)
    imgs = torch.stack(imgs, 0).to(device)

    print("[Image] shape of image: ", img.shape)

    with torch.no_grad():

        #noise_sample = torch.randn(n_mean_latent, 512, device=device)
        noise_sample = torch.randn(n_mean_latent, g_ema.z_dim, device=device)
        #label = torch.zeros([n_mean_latent,g_ema.c_dim],device = device)
        print("[infor] noise sample: ", noise_sample.shape)
        w_samples = g_ema.mapping(noise_sample, None)
        print("[infor] w sample: ", w_samples.shape)
        w_samples = w_samples[:, :1, :]
        print("[infor] w  : ", w_samples.shape)
        w_avg = w_samples.mean(0)
        w_std = ((w_samples - w_avg).pow(2).sum() / n_mean_latent) ** 0.5




    noises = {name: buf for (name, buf) in g_ema.synthesis.named_buffers() if 'noise_const' in name}

    for noise in noises.values():
        noise = torch.randn_like(noise)
        noise.requires_grad = True



    w_opt = w_avg.detach().clone()
    print("[Info] w opt: ", w_opt.shape)

    print(w_opt.shape)

    if w_plus:
        w_opt = w_opt.repeat(1, g_ema.mapping.num_ws, 1)

    print(w_opt.shape)
    w_opt.requires_grad = True
    # exit(1)
    #if args.w_plus:
        #latent_in = latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)



    optimizer = optim.Adam([w_opt] + list(noises.values()), lr=args.lr)

    pbar = tqdm(range(step))
    latent_path = []

    for i in pbar:

        t = i / step
        lr = get_lr(t, args.lr)
        optimizer.param_groups[0]["lr"] = lr
        # AdiIn
        noise_strength = w_std * args.noise * max(0, 1 - t / args.noise_decay) ** 2

        w_noise = torch.randn_like(w_opt) * noise_strength
        if w_plus:
            ws = w_opt + w_noise
        else:
            ws = (w_opt + w_noise).repeat([1, g_ema.mapping.num_ws, 1])

        # print(ws.shape)
        # exit(1)

        img_gen = g_ema.synthesis(ws, noise_mode='const', force_fp32=True)

        #latent_n = latent_noise(latent_in, noise_strength.item())

        #latent, noise = g_ema.prepare([latent_n], input_is_latent=True, noise=noises)
        #img_gen, F = g_ema.generate(latent, noise)

        # Downsample image to 256x256 if it's larger than that. VGG was built for 224x224 images.
        if img_gen.shape[2] > 256:
            img_gen = F.interpolate(img_gen, size=(256, 256), mode='area')

        # print(img_gen)
        p_loss = percept(img_gen, imgs)


        # Noise regularization.
        reg_loss = 0.0
        for v in noises.values():
            noise = v[None, None, :, :]  # must be [1,1,H,W] for F.avg_pool2d()
            while True:
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=3)).mean() ** 2
                reg_loss += (noise * torch.roll(noise, shifts=1, dims=2)).mean() ** 2
                if noise.shape[2] <= 8:
                    break
                noise = F.avg_pool2d(noise, kernel_size=2)

        mse_loss = F.mse_loss(img_gen, imgs)

        loss = p_loss + args.noise_regularize * reg_loss + args.mse * mse_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Normalize noise.
        with torch.no_grad():
            for buf in noises.values():
                buf -= buf.mean()
                buf *= buf.square().mean().rsqrt()
        # print(((i + 1) % 100))
        if (i + 1) % 100 == 0:
            # print("ok")
            latent_path.append(w_opt.detach().clone())

        pbar.set_description(
            (
                f"Perceptual Loss: {p_loss.item():.4f}; Noise regularize Loss: {reg_loss:.4f};"
                f" Mse Loss: {mse_loss.item():.4f}; lr: {lr:.4f}"
            )
        )
    print("[Info] shape of image gen/real_img: {}/{}".format(img_gen.shape, imgs.shape))
    
    # print(g_ema.__dict__)
    # exit(1)
    # latent, noise = g_ema.prepare([latent_path[-1]], input_is_latent=True, noise=noises)
    # img_gen, _ = g_ema.generate(latent, noise)
    # print(latent_path)
    if w_plus:
        ws = latent_path[-1]
    else:
        ws = latent_path[-1].repeat([1, g_ema.mapping.num_ws, 1])

    # print(ws.shape)
    # print(w_opt.shape)
    # exit(1)
    # import cv2
    # import numpy as np

    img_gen = g_ema.synthesis(ws, noise_mode='const')
    # rgb = np.squeeze(img_gen.cpu().detach())
    # rgb = np.transpose(rgb, (1, 2, 0))

    # # print(rgb.shape)

    # bgr = cv2.cvtColor(rgb.numpy()*255, cv2.COLOR_RGB2BGR)

    # cv2.imwrite("synthis.png", bgr)

    result = {
        "latent": latent_path[-1],
        "sample": img_gen,
        "real": imgs,
    }

    return result

def toogle_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag





class PTI:
    def __init__(self,  G, percept, l2_lambda = 1, max_pti_step = 400, pti_lr = 3e-4 ):
        self.g_ema = G
        self.l2_lambda = l2_lambda
        self.max_pti_step = max_pti_step
        self.pti_lr = pti_lr
        self.percept = percept

    def cacl_loss(self, percept, generated_image, real_image):

        mse_loss = F.mse_loss(generated_image, real_image)
        p_loss = percept(generated_image, real_image).sum()
        loss = p_loss + self.l2_lambda * mse_loss
        return loss
    
    def train_real_image(self, inversed_result, w_plus, real_img):

        w_pivot = inversed_result['latent']
        # torch.save(w_pivot, path_latent)
        if w_plus:
            ws = w_pivot
        else:
            ws = w_pivot.repeat([1, self.g_ema.mapping.num_ws, 1])

        generated_image = self.g_ema.synthesis(ws, noise_mode='const')
        
        # print(len(ws))
        # toogle_grad(self.g_ema,True)

        optimizer = torch.optim.Adam(self.g_ema.parameters(), lr=self.pti_lr)
        # print('start PTI')
        pbar = tqdm(range(self.max_pti_step))

        for i in pbar:

            t = i / self.max_pti_step
            lr = get_lr(t, self.pti_lr)
            optimizer.param_groups[0]["lr"] = lr

            generated_image = self.g_ema.synthesis(ws, noise_mode='const')

            loss = self.cacl_loss(self.percept, generated_image, real_img)

            pbar.set_description(
                (
                    f"loss: {loss.item():.4f}"
                )
            )
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # with torch.no_grad():

        #     generated_image = self.g_ema.synthesis(ws, noise_mode='const')
        
        return ws

    def train(self, img, image_flip,  w_plus=False, step=1000, path_pt="custome_weight.pt", path_latent="latent.pt", path_pkl=None):

        # Preprocessing image
        if torch.is_tensor(img) == False:
            transform = transforms.Compose(
                [
                    transforms.Resize(self.g_ema.img_resolution, ),
                    transforms.CenterCrop(self.g_ema.img_resolution),
                    transforms.ToTensor(),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

            real_img = transform(img).to('cuda').unsqueeze(0)
            real_image_flip = transform(image_flip).to('cuda').unsqueeze(0)

        else:
            img = transforms.functional.resize(img, self.g_ema.img_resolution)
            image_flip = transforms.functional.resize(image_flip, self.g_ema.img_resolution)
            transform = transforms.Compose(
                [
                    transforms.CenterCrop(self.g_ema.img_resolution),
                    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
                ]
            )

            real_img = transform(img).to('cuda').unsqueeze(0)
            real_image_flip = transform(image_flip).to('cuda').unsqueeze(0)
        # ================================================================================

        #################################################Image_ORI#############################################################

        inversed_result_1 = inverse_image(self.g_ema, image=img, percept = self.percept, image_size=self.g_ema.img_resolution, w_plus=w_plus, step=step)

        laten_1 = self.train_real_image(inversed_result=inversed_result_1, w_plus=w_plus, real_img=real_img)

        with torch.no_grad():
            generated_image_1 = self.g_ema.synthesis(laten_1, noise_mode='const')

        #################################################Image_FLIP#############################################################
        inversed_result_2 = inverse_image(self.g_ema, image=image_flip, percept = self.percept, image_size=self.g_ema.img_resolution, w_plus=w_plus, step=step)

        laten_2 = self.train_real_image(inversed_result=inversed_result_2, w_plus=w_plus, real_img=real_image_flip)

        with torch.no_grad():
            generated_image_2 = self.g_ema.synthesis(laten_2, noise_mode='const')




        ############################################################Image-CONCAT##################################################
        ws = (laten_1 + laten_2) / 2

        with torch.no_grad():
            generated_image = self.g_ema.synthesis(ws, noise_mode='const')
        
        

        
        # print(generated_image)
        # import cv2
        # import numpy as np
        # image_gen = np.squeeze(generated_image.detach().cpu().numpy())
        # print(image_gen.shape)
        # image_gen = cv2.cvtColor(np.transpose(image_gen, (1, 2, 0))*255, cv2.COLOR_RGB2BGR)
        # print(image_gen.shape)

        imageio.imsave('test_right.png', make_image(generated_image_1)[0])

        imageio.imsave('test_left.png', make_image(generated_image_2)[0])

        imageio.imsave('test_align.png', make_image(generated_image)[0])


        # inversed_result_1 = {}
        # inversed_result_1['latent'] = inversed_result_1["latent"] 
        # inversed_result_1['sample'] = inversed_result_1["sample"] 
        # inversed_result['latent'] = (inversed_result_1["latent"] + inversed_result_2["latent"]) / 2
        # inversed_result['sample'] = (inversed_result_1["sample"] + inversed_result_2["sample"]) / 2
        # print()
        # inversed_result['latent'] = (inversed_result_1["latent"] + inversed_result_2["latent"]) / 2

    #     {
    #     "latent": latent_path[-1],
    #     "sample": img_gen,
    #     "real": imgs,
    # }


        # print(inversed_result.keys())
        # print(inversed_result["latent"].shape)
        # print(inversed_result["sample"].shape)
        # print(inversed_result["real"].shape)

        # exit(1)

        

        # torch.save(self.g_ema, f'{path_pt}')

        # with open(f'{path_pt}', 'rb') as f_new:
        #     new_G = torch.load(f_new).cuda()

        # print("Exporting large updated pickle based off new generator and ffhq.pkl")

        # with open(path_pkl, 'rb') as f:
        #     d = pickle.load(f)
        #     old_G = d['G_ema'].cuda()
        #     old_D = d['D'].eval().requires_grad_(False).cpu()

        # tmp = {}
        # tmp['G'] = old_G.eval().requires_grad_(False).cpu()
        # tmp['G_ema'] = new_G.eval().requires_grad_(False).cpu()
        # tmp['D'] = old_D
        # tmp['training_set_kwargs'] = None
        # tmp['augment_pipe'] = None


        # with open(path_pkl, 'wb') as f:
        #     pickle.dump(tmp, f)

        return generated_image, ws

if __name__ == "__main__":
    state = {
        "images": {
            # image_orig: the original image, change with seed/model is changed
            # image_raw: image with mask and points, change durning optimization
            # image_show: image showed on screen
        },
        "temporal_params": {
            # stop
        },
        'mask':
            None,  # mask for visualization, 1 for editing and 0 for unchange
        'last_mask': None,  # last edited mask
        'show_mask': True,  # add button
        "generator_params": dnnlib.EasyDict(),
        "params": {
            "seed": 0,
            "motion_lambda": 20,
            "r1_in_pixels": 3,
            "r2_in_pixels": 12,
            "magnitude_direction_in_pixels": 1.0,
            "latent_space": "w+",
            "trunc_psi": 0.7,
            "trunc_cutoff": None,
            "lr": 0.001,
        },
        "device": 'cuda:0',
        "draw_interval": 1,
        "renderer": renderer.Renderer(disable_timing=True),
        "points": {},
        "curr_point": None,
        "curr_type_point": "start",
        'editing_state': 'add_points',
        'pretrained_weight': 'stylegan2-ffhq-512x512'
    }
    cache_dir = '/Face_View/DragGAN_CRN/checkpoints'
    valid_checkpoints_dict = {
        f.split('/')[-1].split('.')[0]: os.path.join(cache_dir, f)
        for f in os.listdir(cache_dir)
        if (f.endswith('pkl') and os.path.exists(os.path.join(cache_dir, f)))
    }
    state['renderer'].init_network(state['generator_params'],  # res
        valid_checkpoints_dict[state['pretrained_weight']],  # pkl
        state['params']['seed'],  # w0_seed,
        None,  # w_load
        state['params']['latent_space'] == 'w+',  # w_plus
        'const',
        state['params']['trunc_psi'],  # trunc_psi,
        state['params']['trunc_cutoff'],  # trunc_cutoff,
        None,  # input_transform
        state['params']['lr']  # lr
    )
    image = Image.open('/Face_View/DragGAN_CRN/photo_2023-07-25_16-19-10.jpg')
    from PIL import ImageOps
    image_flip = ImageOps.mirror(image)

    image_flip.save("image_flip.png")
    # exit(1)

    G = state['renderer'].G
    # model = Generator()
    # print(state['renderer'].__dict__)
    # exit(1)
    percept = util.PerceptualLoss(
        model="net-lin", net="vgg", use_gpu=True
    )
    # print(percept)
    # exit(1)
    # result = inverse_image(G, image, percept=percept, image_size=G.img_resolution)


    pti = PTI(G, percept, max_pti_step=500)

    result = pti.train(
                    image,
                    image_flip,
                    state['params']['latent_space'] == 'w+', 
                    step = 500, 
                    path_pt = "/Face_View/DragGAN_CRN/checkpoints/custome.pt",
                    path_latent="/Face_View/DragGAN_CRN/latent/latent.pt",
                    path_pkl= "/Face_View/DragGAN_CRN/checkpoints/stylegan2-ffhq-512x512.pkl")
    
    imageio.imsave('test.png', make_image(result[0])[0])



