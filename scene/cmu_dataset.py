import json
import os
from typing import NamedTuple

import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset


class CMUCamera(NamedTuple):
    # same as diff_gaussian_rasterization_df
    image_height: int
    image_width: int 
    tanfovx : float
    tanfovy : float
    bg : torch.Tensor
    scale_modifier : float
    viewmatrix : torch.Tensor
    projmatrix : torch.Tensor
    sh_degree : int
    campos : torch.Tensor
    prefiltered : bool
    debug : bool

    # for cmu panoptic dataset
    timestamp : float
    colmap_id : int
    image_path: str
    resolution: tuple[int, int]
    im_scale: float
    T: np.ndarray # translation matrix

def setup_camera(w, h, k, w2c, timestamp, cam_id, image_path, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    T = cam_center.cpu().numpy()
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = CMUCamera(
        image_height=h,
        image_width=w,
        tanfovx=w / (2 * fx),
        tanfovy=h / (2 * fy),
        bg=torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda"),
        scale_modifier=1.0,
        viewmatrix=w2c,
        projmatrix=full_proj,
        sh_degree=0,
        campos=cam_center,
        prefiltered=False,
        debug=True,
        timestamp=timestamp,
        colmap_id=cam_id,
        image_path=image_path,
        resolution=(w, h),
        im_scale=1.0,
        T=T
    )
    return cam

class PanopticDataset(Dataset):
    def __init__(self, datadir: str, json_path: str, lazy_loader: bool = False):
        # --- load metadata once ---
        meta_file = os.path.join(datadir, json_path)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        self.datadir = datadir
        self.w = meta["w"]
        self.h = meta["h"]
        self.max_time = len(meta["fn"])
        self.entries = []
        self.lazy_loader = lazy_loader

        # flatten (time × camera) into a single list
        for t_idx in range(self.max_time):
            time = t_idx
            Ks = meta["k"][t_idx]  # list of 3×3 intrinsics
            W2Cs = meta["w2c"][t_idx]
            FNs = meta["fn"][t_idx]
            CIDs = meta["cam_id"][t_idx]

            for K_list, w2c_list, fn, cid in zip(Ks, W2Cs, FNs, CIDs):
                # turn that nested list into a real 3×3 array
                K = np.array(K_list, dtype=np.float32).reshape(3, 3)
                fx = float(K[0, 0])
                fy = float(K[1, 1])

                self.entries.append(
                    {
                        "time": time,
                        "K": K,
                        "fx": fx,
                        "fy": fy,
                        "w2c": np.array(w2c_list, dtype=np.float32),
                        "fn": fn,
                        "cam_id": cid,
                    }
                )

        # simple PIL→Tensor loader
        self.transform = T.ToTensor()

    def get_metadata(self, idx):
        e = self.entries[idx]
        img_path = os.path.join(self.datadir, "ims", e["fn"])
        
        # build camera; pass K and w2c positionally, not as 'K='
        cam = setup_camera(
            self.w,  # image width
            self.h,  # image height
            e["K"],  # your 3×3 intrinsics matrix
            e["w2c"],  # world-to-camera 4×4
            e["time"],
            e["cam_id"],
            img_path,
            near=0.01,
            far=100.0,
        )
        return cam

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        if self.lazy_loader:
            return self.get_metadata(idx)

        e = self.entries[idx]

        # load image on‐the‐fly
        img_path = os.path.join(self.datadir, "ims", e["fn"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # build camera; pass K and w2c positionally, not as 'K='
        cam = setup_camera(
            self.w,  # image width
            self.h,  # image height
            e["K"],  # your 3×3 intrinsics matrix
            e["w2c"],  # world-to-camera 4×4
            e["time"],
            e["cam_id"],
            img_path,
            near=0.01,
            far=100.0,
        )

        return cam
