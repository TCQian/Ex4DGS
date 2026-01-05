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
    image_name: str
    
    # Store original intrinsics from JSON for accurate resolution scaling
    fx: float  # Original fx from K matrix (None if not available)
    fy: float  # Original fy from K matrix (None if not available)
    cx: float  # Original cx from K matrix (None if not available)
    cy: float  # Original cy from K matrix (None if not available)

def setup_camera(w, h, k, w2c, timestamp, cam_id, image_path, near=0.01, far=100):
    fx, fy, cx, cy = k[0][0], k[1][1], k[0][2], k[1][2]
    
    # Assert: Image dimensions should be 640x340 or 640x360 (or other valid dimensions)
    assert w > 0 and h > 0, f"Invalid image dimensions: w={w}, h={h}"
    assert fx > 0 and fy > 0, f"Invalid focal lengths: fx={fx}, fy={fy}"
    assert 0 <= cx < w and 0 <= cy < h, f"Invalid principal point: cx={cx}, cy={cy} (image size: {w}x{h})"
    
    w2c = torch.tensor(w2c).cuda().float()
    cam_center = torch.inverse(w2c)[:3, 3]
    T = cam_center.cpu().numpy()
    w2c = w2c.unsqueeze(0).transpose(1, 2)
    
    # Compute OpenGL projection matrix
    opengl_proj = torch.tensor([[2 * fx / w, 0.0, -(w - 2 * cx) / w, 0.0],
                                [0.0, 2 * fy / h, -(h - 2 * cy) / h, 0.0],
                                [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
    full_proj = w2c.bmm(opengl_proj)
    
    # Assert: Verify projection matrix computation
    # Extract opengl_proj from full_proj to verify
    viewmatrix_inv = torch.inverse(w2c)
    extracted_opengl_proj = viewmatrix_inv.bmm(full_proj).squeeze(0).transpose(0, 1)
    expected_opengl_proj = opengl_proj.squeeze(0).transpose(0, 1)
    assert torch.allclose(extracted_opengl_proj, expected_opengl_proj, atol=1e-5), \
        f"Projection matrix mismatch in setup_camera! Expected {expected_opengl_proj}, got {extracted_opengl_proj}"
    
    # Assert: Verify projection matrix matches image dimensions
    # opengl_proj[0,0] should be 2*fx/w, opengl_proj[1,1] should be 2*fy/h
    assert abs(extracted_opengl_proj[0, 0].item() - 2 * fx / w) < 1e-5, \
        f"Projection matrix[0,0] mismatch: expected {2*fx/w}, got {extracted_opengl_proj[0,0].item()}"
    assert abs(extracted_opengl_proj[1, 1].item() - 2 * fy / h) < 1e-5, \
        f"Projection matrix[1,1] mismatch: expected {2*fy/h}, got {extracted_opengl_proj[1,1].item()}"
    
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
        T=T,
        image_name=image_path,
        fx=fx,  # Store original intrinsics from JSON
        fy=fy,
        cx=cx,
        cy=cy
    )
    
    # Assert: Verify camera fields match input dimensions
    assert cam.image_width == w, f"Camera image_width mismatch: expected {w}, got {cam.image_width}"
    assert cam.image_height == h, f"Camera image_height mismatch: expected {h}, got {cam.image_height}"
    assert cam.resolution == (w, h), f"Camera resolution mismatch: expected {(w, h)}, got {cam.resolution}"
    
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
