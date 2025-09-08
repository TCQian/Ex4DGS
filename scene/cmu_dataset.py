import json
import os
import torch

import numpy as np
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset

def setup_camera(w, h, k, w2c, near=0.01, far=100):
    from diff_gaussian_rasterization import GaussianRasterizationSettings as Camera
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    cam = Camera(
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
        debug=True
    )
    return cam

class PanopticDataset(Dataset):
    def __init__(self, datadir: str, json_path: str):
        # --- load metadata once ---
        meta_file = os.path.join(datadir, json_path)
        with open(meta_file, "r") as f:
            meta = json.load(f)

        self.datadir = datadir
        self.w = meta["w"]
        self.h = meta["h"]
        self.max_time = len(meta["fn"])
        self.entries = []

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

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        from scene.dataset_readers import CameraInfo2
        from utils.graphics_utils import focal2fov
        e = self.entries[idx]

        # load image on‐the‐fly
        img_path = os.path.join(self.datadir, "ims", e["fn"])
        img = Image.open(img_path).convert("RGB")
        img = self.transform(img)

        # Note: setup_camera is no longer used since we return CameraInfo2 instead
        # but keeping the logic for potential future use
        # cam = setup_camera(
        #     self.w,  # image width
        #     self.h,  # image height
        #     e["K"],  # your 3×3 intrinsics matrix
        #     e["w2c"],  # world-to-camera 4×4
        #     near=0.01,
        #     far=100.0,
        # )

        # Extract camera parameters for CameraInfo2
        K = e["K"]
        w2c = e["w2c"]
        
        # Convert w2c to R and T
        R = w2c[:3, :3].T  # R is stored transposed due to 'glm' in CUDA code
        T = w2c[:3, 3]
        
        # Calculate FOV from intrinsics
        FovX = focal2fov(K[0, 0], self.w)
        FovY = focal2fov(K[1, 1], self.h)
        
        # Create CameraInfo2 instance
        cam_info = CameraInfo2(
            uid=e["cam_id"],
            R=R,
            T=T,
            FovY=FovY,
            FovX=FovX,
            image_path=img_path,
            image_name=e["fn"],
            width=self.w,
            height=self.h,
            near=0.01,
            far=100.0,
            timestamp=e["time"],
            pose=None,
            hpdirecitons=None,
            cxr=0.0,
            cyr=0.0
        )
        
        # Set the image data
        cam_info.image = img

        return cam_info
