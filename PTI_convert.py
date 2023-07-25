import os
from configs import paths_config
import pickle
import torch


class pt2pkl:
    def __init__(self, model_path):
        self.model_path = model_path
        self.model_id = []
        self.image_name = []
        file_list = os.listdir(self.model_path)
        for file in file_list:
            name, ext = os.path.splitext(file)
            if ext == ".pt":
                self.model_id.append(name.split('_')[1])
                self.image_name.append(name.split('_')[1])
    
    def load_generator(self, model_id, image_name):
        with open(paths_config.stylegan2_ada_ffhd, 'rb') as f:
            old_G = pickle.load(f)["G_ema"].cuda()

        with open(f'{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pt', 'rb') as f_new:
            new_G = torch.load(f_new).cuda()

        return old_G, new_G
    
    def export_update_pickle(self, new_G, model_id, image_name):
        print("Exporting large updated pickle based off new generator and ffhq.pkl")

        with open(paths_config.stylegan2_ada_ffhq, 'rb') as f:
            d =pickle.load(f)
            old_G = d["G_ema"].cuda() ##tensor
            old_D = d['D'].eval().requires_grad_(False).cpu()

        tmp = {}
        tmp["G_ema"] = old_G.eval().requires_grad_(False).cpu() # copy deep of old G
        tmp["G_"] = new_G.eval().requires_grad_(False).cpu() # copy deep of new G
        tmp["D"] = old_D # copy deep of new G
        tmp["training_set_kwargs"] = None
        tmp["augment_pipe"] = None

        with open(f"{paths_config.checkpoints_dir}/model_{model_id}_{image_name}.pkl", "wb") as f:
            pickle.dump(tmp, f)

    def pt2pkl_impl(self):
        for i in range(len(self.image_name)):
            use_multi_id_tranining = False
            generator_type = paths_config.multi_id_type if use_multi_id_tranining else self.image_name[i]
            old_G, new_G = self.load_generator(self.model_id[i], generator_type)
            self.export_update_pickle(new_G, self.model_id[i], self.image_name[i])


if __name__ == "__main__":
    modelpath = "./checkpoints"
    file_list = os.listdir(modelpath)
    for file in file_list:
        name, ext = os.path.splitext(file)
        print(name, ext)
    
    pt2pkl_demo = pt2pkl(model_path=modelpath)
    pt2pkl_demo.pt2pkl_impl()



        