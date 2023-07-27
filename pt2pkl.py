import pickle
import torch
import torch_utils

if __name__ == "__main__":
    model = torch.load("/Face_View/DragGAN_CRN/checkpoint_swagan.pt")
    model = {k.lower(): v for k, v in model.items()}
    print(model)



    data = open("/Face_View/DragGAN_CRN/checkpoints/stylegan2-ffhq-512x512.pkl", 'rb')
    arch = pickle.load(data)
    # arch = {k.lower(): v for k, v in arch.items()}
    print(arch.keys())
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