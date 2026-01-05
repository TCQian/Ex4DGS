#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
from torch import nn
import numpy as np
from kornia import create_meshgrid
from utils.graphics_utils import getWorld2View2, getProjectionMatrix, getProjectionMatrixCV, pix2ndc, ndc2pix, fov2focal
from utils.general_utils import PILtoTorch
from PIL import Image
from scene.cmu_dataset import CMUCamera


class Camera(nn.Module):
    def __init__(self, colmap_id, R, T, FoVx, FoVy, image, gt_alpha_mask,
                 image_name, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda"
                 ):
        super(Camera, self).__init__()

        self.uid = uid
        self.colmap_id = colmap_id
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        # self.original_image = image.clamp(0.0, 1.0).to(self.data_device)
        self.original_image = image.clamp(0.0, 1.0)
        self.image_width = self.original_image.shape[2]
        self.image_height = self.original_image.shape[1]

        if gt_alpha_mask is not None:
            # self.original_image *= gt_alpha_mask.to(self.data_device)
            self.original_image *= gt_alpha_mask
        else:
            self.original_image *= torch.ones((1, self.image_height, self.image_width), device=self.data_device)

        self.zfar = 100.0
        self.znear = 0.01

        self.trans = trans
        self.scale = scale

        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale)).transpose(0, 1).cuda()
        self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

class MiniCam:
    def __init__(self, width, height, fovy, fovx, znear, zfar, world_view_transform, full_proj_transform):
        self.image_width = width
        self.image_height = height    
        self.FoVy = fovy
        self.FoVx = fovx
        self.znear = znear
        self.zfar = zfar
        self.world_view_transform = world_view_transform
        self.full_proj_transform = full_proj_transform
        view_inv = torch.inverse(self.world_view_transform)
        self.camera_center = view_inv[3][:3]


class Cameravideo():
    def __init__(self, colmap_id, R, T, FoVx, FoVy, gt_alpha_mask, image,
                 image_name, image_path, uid,
                 trans=np.array([0.0, 0.0, 0.0]), scale=1.0, data_device = "cuda", 
                 near=0.01, far=100.0, timestamp=0.0, 
                 rayo=None, rayd=None, rays=None, cxr=0.0, cyr=0.0, resolution=(1., 1.),
                 opticalflow_path=None, depth_path=None, im_scale=1.0
                 ):
        super(Cameravideo, self).__init__()
        
        self.uid = uid
        self.colmap_id = colmap_id
        self.image_width = resolution[0]
        self.image_height = resolution[1]
        self.R = R
        self.T = T
        self.FoVx = FoVx
        self.FoVy = FoVy
        self.image_name = image_name
        self.timestamp = timestamp
        self.fisheyemapper = None
        self.resolution = resolution
        self.image_path = image_path
        self.opticalflow_path = opticalflow_path
        self.depth_path = depth_path
        self.im_scale = im_scale

        try:
            self.data_device = torch.device(data_device)
        except Exception as e:
            print(e)
            print(f"[Warning] Custom device {data_device} failed, fallback to default cuda device" )
            self.data_device = torch.device("cuda")

        self.zfar = far
        self.znear = near
        self.trans = trans
        self.scale = scale

        # w2c 
        self.world_view_transform = torch.tensor(getWorld2View2(R, T, trans, scale), dtype=torch.float32).transpose(0, 1).cuda()
        if cyr != 0.0 :
            self.cxr = cxr
            self.cyr = cyr
            self.projection_matrix = getProjectionMatrixCV(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy, cx=cxr, cy=cyr).transpose(0,1).cuda()
        else:
            self.projection_matrix = getProjectionMatrix(znear=self.znear, zfar=self.zfar, fovX=self.FoVx, fovY=self.FoVy).transpose(0,1).cuda()
        self.full_proj_transform = (self.world_view_transform.unsqueeze(0).bmm(self.projection_matrix.unsqueeze(0))).squeeze(0)
        self.camera_center = self.world_view_transform.inverse()[3, :3]

        if rayd is not None:
            projectinverse = self.projection_matrix.T.inverse()
            camera2wold = self.world_view_transform.T.inverse()
            pixgrid = create_meshgrid(self.image_height, self.image_width, normalized_coordinates=False, device="cpu")[0]
            pixgrid = pixgrid.cuda()  # H,W,
            
            xindx = pixgrid[:,:,0] # x 
            yindx = pixgrid[:,:,1] # y
            
            ndcy, ndcx = pix2ndc(yindx, self.image_height), pix2ndc(xindx, self.image_width)
            ndcx = ndcx.unsqueeze(-1)
            ndcy = ndcy.unsqueeze(-1)# * (-1.0)
            
            ndccamera = torch.cat((ndcx, ndcy,   torch.ones_like(ndcy) * (1.0) , torch.ones_like(ndcy)), 2) # N,4 

            projected = ndccamera @ projectinverse.T 
            diretioninlocal = projected / projected[:,:,3:] # 

            direction = diretioninlocal[:,:,:3] @ camera2wold[:3,:3].T 
            rays_d = torch.nn.functional.normalize(direction, p=2.0, dim=-1)
            
            self.rayo = self.camera_center.expand(rays_d.shape).permute(2, 0, 1).unsqueeze(0)                                     #rayo.permute(2, 0, 1).unsqueeze(0)
            self.rayd = rays_d.permute(2, 0, 1).unsqueeze(0)                                                                          #rayd.permute(2, 0, 1).unsqueeze(0)
        else :
            self.rayo = None
            self.rayd = None
            
        self.image = image
        
        
WARNED = False


def loadCam(args, id, cam_info, resolution_scale):
    global WARNED
    orig_w, orig_h = cam_info.image.size

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)), round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    resized_image_rgb = PILtoTorch(cam_info.image, resolution)

    gt_image = resized_image_rgb[:3, ...]
    loaded_mask = None

    if resized_image_rgb.shape[1] == 4:
        loaded_mask = resized_image_rgb[3:4, ...]

    return Camera(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, 
                  image=gt_image, gt_alpha_mask=loaded_mask,
                  image_name=cam_info.image_name, uid=id, data_device=args.data_device)
    

def loadCamVideo(args, id, cam_info, resolution_scale):
    global WARNED
    # Check if this is a CMU Panoptic dataset entry (dictionary with 'camera' key)
    if isinstance(cam_info, CMUCamera):
        orig_w, orig_h = cam_info.image_width, cam_info.image_height
        
        # Assert: Original dimensions should be valid
        assert orig_w > 0 and orig_h > 0, f"Invalid original dimensions: w={orig_w}, h={orig_h}"
        assert hasattr(cam_info, 'fx') and cam_info.fx is not None, "Camera missing fx intrinsic"
        assert hasattr(cam_info, 'fy') and cam_info.fy is not None, "Camera missing fy intrinsic"
        assert hasattr(cam_info, 'cx') and cam_info.cx is not None, "Camera missing cx intrinsic"
        assert hasattr(cam_info, 'cy') and cam_info.cy is not None, "Camera missing cy intrinsic"

        if args.resolution in [1, 2, 4, 8]:
            resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
        else:  # should be a type that converts to float
            if args.resolution == -1:
                if orig_w > 1600:
                    if not WARNED:
                        print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                            "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                        WARNED = True
                    global_down = orig_w / 1600
                else:
                    global_down = 1
            else:
                global_down = orig_w / args.resolution

            scale = float(global_down) * float(resolution_scale)
            resolution = (int(orig_w / scale), int(orig_h / scale))

        new_w, new_h = resolution[0], resolution[1]
        
        # Assert: New resolution should be valid
        assert new_w > 0 and new_h > 0, f"Invalid new resolution: w={new_w}, h={new_h}"
        
        # Recompute projection matrix if resolution changed
        if new_w != orig_w or new_h != orig_h:
            # Extract original intrinsics
            orig_fx = cam_info.fx
            orig_fy = cam_info.fy
            orig_cx = cam_info.cx
            orig_cy = cam_info.cy
            
            # Scale intrinsics proportionally
            scale_x = new_w / orig_w
            scale_y = new_h / orig_h
            new_fx = orig_fx * scale_x
            new_fy = orig_fy * scale_y
            new_cx = orig_cx * scale_x
            new_cy = orig_cy * scale_y
            
            # Recompute OpenGL projection matrix with new resolution and intrinsics
            near = 0.01
            far = 100.0
            opengl_proj = torch.tensor([[2 * new_fx / new_w, 0.0, -(new_w - 2 * new_cx) / new_w, 0.0],
                                        [0.0, 2 * new_fy / new_h, -(new_h - 2 * new_cy) / new_h, 0.0],
                                        [0.0, 0.0, far / (far - near), -(far * near) / (far - near)],
                                        [0.0, 0.0, 1.0, 0.0]]).cuda().float().unsqueeze(0).transpose(1, 2)
            new_projmatrix = cam_info.viewmatrix.bmm(opengl_proj)
            
            # Assert: Verify recomputed projection matrix matches new resolution
            viewmatrix_inv = torch.inverse(cam_info.viewmatrix)
            extracted_opengl_proj = viewmatrix_inv.bmm(new_projmatrix).squeeze(0).transpose(0, 1)
            assert abs(extracted_opengl_proj[0, 0].item() - 2 * new_fx / new_w) < 1e-5, \
                f"Recomputed projection matrix[0,0] mismatch: expected {2*new_fx/new_w}, got {extracted_opengl_proj[0,0].item()}"
            assert abs(extracted_opengl_proj[1, 1].item() - 2 * new_fy / new_h) < 1e-5, \
                f"Recomputed projection matrix[1,1] mismatch: expected {2*new_fy/new_h}, got {extracted_opengl_proj[1,1].item()}"
        else:
            # Resolution unchanged - verify projection matrix still matches
            viewmatrix_inv = torch.inverse(cam_info.viewmatrix)
            extracted_opengl_proj = viewmatrix_inv.bmm(cam_info.projmatrix).squeeze(0).transpose(0, 1)
            expected_fx_scale = 2 * cam_info.fx / orig_w
            expected_fy_scale = 2 * cam_info.fy / orig_h
            assert abs(extracted_opengl_proj[0, 0].item() - expected_fx_scale) < 1e-5, \
                f"Projection matrix[0,0] mismatch for unchanged resolution: expected {expected_fx_scale}, got {extracted_opengl_proj[0,0].item()}"
            assert abs(extracted_opengl_proj[1, 1].item() - expected_fy_scale) < 1e-5, \
                f"Projection matrix[1,1] mismatch for unchanged resolution: expected {expected_fy_scale}, got {extracted_opengl_proj[1,1].item()}"
            new_projmatrix = cam_info.projmatrix

        new_cam = CMUCamera(
            image_height=resolution[1],
            image_width=resolution[0],
            tanfovx=cam_info.tanfovx,
            tanfovy=cam_info.tanfovy,
            bg=cam_info.bg,
            scale_modifier=cam_info.scale_modifier,
            viewmatrix=cam_info.viewmatrix,
            projmatrix=new_projmatrix,
            sh_degree=cam_info.sh_degree,
            campos=cam_info.campos,
            prefiltered=cam_info.prefiltered,
            debug=cam_info.debug,
            timestamp=cam_info.timestamp,
            colmap_id=cam_info.colmap_id,
            image_path=cam_info.image_path,
            resolution=resolution,
            im_scale=cam_info.im_scale,
            T=cam_info.T,
            image_name=cam_info.image_name,
            fx=cam_info.fx,  # Preserve original intrinsics
            fy=cam_info.fy,
            cx=cam_info.cx,
            cy=cam_info.cy
        )
        
        # Assert: Final camera dimensions match resolution
        assert new_cam.image_width == new_w, \
            f"Final camera image_width mismatch: expected {new_w}, got {new_cam.image_width}"
        assert new_cam.image_height == new_h, \
            f"Final camera image_height mismatch: expected {new_h}, got {new_cam.image_height}"
        assert new_cam.resolution == (new_w, new_h), \
            f"Final camera resolution mismatch: expected {(new_w, new_h)}, got {new_cam.resolution}"
        
        # Assert: Projection matrix matches image dimensions
        final_viewmatrix_inv = torch.inverse(new_cam.viewmatrix)
        final_opengl_proj = final_viewmatrix_inv.bmm(new_cam.projmatrix).squeeze(0).transpose(0, 1)
        if new_w != orig_w or new_h != orig_h:
            # For scaled resolution, use scaled intrinsics
            expected_fx_scale = 2 * new_fx / new_w
            expected_fy_scale = 2 * new_fy / new_h
        else:
            # For original resolution, use original intrinsics
            expected_fx_scale = 2 * cam_info.fx / new_w
            expected_fy_scale = 2 * cam_info.fy / new_h
        
        assert abs(final_opengl_proj[0, 0].item() - expected_fx_scale) < 1e-5, \
            f"Final projection matrix[0,0] mismatch: expected {expected_fx_scale}, got {final_opengl_proj[0,0].item()} (image_width={new_w})"
        assert abs(final_opengl_proj[1, 1].item() - expected_fy_scale) < 1e-5, \
            f"Final projection matrix[1,1] mismatch: expected {expected_fy_scale}, got {final_opengl_proj[1,1].item()} (image_height={new_h})"
        
        return new_cam
    
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

        scale = float(global_down) * float(resolution_scale)
        resolution = (int(orig_w / scale), int(orig_h / scale))

    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 
     
    if camerapose is not None:
        rays_o, rays_d = 1, cameradirect
    else :
        rays_o = None
        rays_d = None
    return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                  FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, 
                  image=cam_info.image,
                  image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                  near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                  rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution)
    

def loadCamVideoss(args, id, cam_info, resolution_scale, nogt=False):
    global WARNED
    orig_w, orig_h = cam_info.width, cam_info.height

    if args.resolution in [1, 2, 4, 8]:
        resolution = round(orig_w/(resolution_scale * args.resolution)),  round(orig_h/(resolution_scale * args.resolution))
    else:  # should be a type that converts to float
        if args.resolution == -1:
            if orig_w > 1600:
                if not WARNED:
                    print("[ INFO ] Encountered quite large input images (>1.6K pixels width), rescaling to 1.6K.\n "
                        "If this is not desired, please explicitly specify '--resolution/-r' as 1")
                    WARNED = True
                global_down = orig_w / 1600
            else:
                global_down = 1
        else:
            global_down = orig_w / args.resolution

    resolution = (int(orig_w / 2), int(orig_h / 2))
    cameradirect = cam_info.hpdirecitons
    camerapose = cam_info.pose 

    im_scale = 1
    # load gt image 
    if nogt == False :
        if "01_Welder" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
                
        if "12_Cave" in args.source_path:
            if "camera_0009" in cam_info.image_name:
                im_scale = 1.15
        
        if "04_Truck" in args.source_path:
            if "camera_0008" in cam_info.image_name:
                im_scale = 1.2
        
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution, im_scale=im_scale)
    else:
        if camerapose is not None:
            rays_o, rays_d = 1, cameradirect
        else :
            rays_o = None
            rays_d = None
            
        return Cameravideo(colmap_id=cam_info.uid, R=cam_info.R, T=cam_info.T, 
                    FoVx=cam_info.FovX, FoVy=cam_info.FovY, gt_alpha_mask=None, image=cam_info.image if not args.lazy_loader else None,
                    image_name=cam_info.image_name, image_path=cam_info.image_path, uid=id, data_device=args.data_device, 
                    near=cam_info.near, far=cam_info.far, timestamp=cam_info.timestamp, 
                    rayo=rays_o, rayd=rays_d,cxr=cam_info.cxr,cyr=cam_info.cyr, resolution=resolution)
        

def cameraList_from_camInfos(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCam(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo(cam_infos, resolution_scale, args):
    camera_list = []

    for id, c in enumerate(cam_infos):
        camera_list.append(loadCamVideo(args, id, c, resolution_scale))

    return camera_list


def cameraList_from_camInfosVideo2(cam_infos, resolution_scale, args, ss=False):
    camera_list = []

    if not ss: #
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideo(args, id, c, resolution_scale))
    else:
        for id, c in enumerate(cam_infos):
            camera_list.append(loadCamVideoss(args, id, c, resolution_scale))

    return camera_list


def camera_to_JSON(id, camera):
    # Check if this is a CMU Panoptic dataset entry (dictionary with 'camera' key)
    if isinstance(camera, CMUCamera):
        # CMU Panoptic camera format
        # Extract world-to-camera matrix from viewmatrix (this is a tensor)
        w2c = camera.viewmatrix.squeeze(0).detach().cpu().numpy()
        
        # Extract camera center (world position) (this is a tensor)
        pos = camera.campos.detach().cpu().numpy()
        
        # Extract rotation matrix from w2c
        rot = w2c[:3, :3]
        serializable_array_2d = [x.tolist() for x in rot]
        
        # Calculate focal lengths from tanfov (these are already numpy.float32)
        fx = camera.image_width / (2 * float(camera.tanfovx))
        fy = camera.image_height / (2 * float(camera.tanfovy))
        
        camera_entry = {
            'id': int(id), 
            'img_name': f'cmu_camera_{camera.colmap_id}_t{camera.timestamp}',
            'width': int(camera.image_width),
            'height': int(camera.image_height),
            'position': pos.tolist(),
            'rotation': serializable_array_2d,
            'fy': float(fy),
            'fx': float(fx)
        }
        return camera_entry
    
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = camera.R.transpose()
    Rt[:3, 3] = camera.T
    Rt[3, 3] = 1.0

    W2C = np.linalg.inv(Rt)
    pos = W2C[:3, 3]
    rot = W2C[:3, :3]
    serializable_array_2d = [x.tolist() for x in rot]
    camera_entry = {
        'id' : id,
        'img_name' : camera.image_name,
        'width' : camera.width,
        'height' : camera.height,
        'position': pos.tolist(),
        'rotation': serializable_array_2d,
        'fy' : fov2focal(camera.FovY, camera.height),
        'fx' : fov2focal(camera.FovX, camera.width)
    }
    return camera_entry
