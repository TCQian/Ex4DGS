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

import os
import json
import sys
import uuid
import math
import time
from random import randint
import gc
from itertools import compress

import torch
from tqdm import tqdm
from PIL import Image
import joblib
import numpy as np

from gaussian_renderer import render, network_gui
from scene import Scene, getmodel
from utils.image_utils import psnr
from utils.loss_utils import l1_loss, ssim
from utils.general_utils import safe_state, PILtoTorch
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False
TENSORBOARD_FOUND = False
torch.set_default_dtype(torch.float32)


def verify_gaussians_before_after_expansion(gaussians, test_cam, bg, pipe, dataset, iteration, stage="BEFORE", model_path=None):
    """
    Verify Gaussian attributes before/after expansion.
    Tests rendering of 5th frame of first train camera.
    Prints attributes of first 3 Gaussians and interpolation details.
    Saves rendered test image.
    """
    print(f"\n{'='*80}")
    print(f"[ITER {iteration}] VERIFICATION {stage} EXPANSION")
    print(f"{'='*80}")
    
    # Get test timestamp (5th frame)
    test_timestamp = test_cam.timestamp
    print(f"Test Camera: ID={test_cam.colmap_id}, Timestamp={test_timestamp}")
    
    # Render the test camera
    with torch.no_grad():
        render_pkg = render(test_cam, gaussians, pipe, bg, near=dataset.near, far=dataset.far)
        image = render_pkg["render"]
        visibility_filter = render_pkg["visibility_filter"]
        
        # Save rendered image
        if model_path is not None:
            # Create verification directory
            verify_dir = os.path.join(model_path, "verification_images")
            os.makedirs(verify_dir, exist_ok=True)
            
            # Save rendered image
            image_np = (torch.clamp(image, 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            image_pil = Image.fromarray(image_np)
            image_filename = f"iter_{iteration:06d}_{stage.lower()}_cam{test_cam.colmap_id}_t{test_timestamp}.png"
            image_path = os.path.join(verify_dir, image_filename)
            image_pil.save(image_path)
            print(f"Saved rendered image: {image_path}")
            
            # Try to load and save ground truth image if available
            gt_saved = False
            if hasattr(test_cam, 'image_path') and test_cam.image_path and os.path.exists(test_cam.image_path):
                try:
                    gt_image_pil = Image.open(test_cam.image_path)
                    gt_filename = f"iter_{iteration:06d}_{stage.lower()}_cam{test_cam.colmap_id}_t{test_timestamp}_gt.png"
                    gt_path = os.path.join(verify_dir, gt_filename)
                    gt_image_pil.save(gt_path)
                    print(f"Saved ground truth image: {gt_path}")
                    gt_saved = True
                except Exception as e:
                    print(f"Could not save ground truth image from path: {e}")
            
            if not gt_saved and hasattr(test_cam, 'image') and test_cam.image is not None:
                try:
                    # If image is already loaded as tensor
                    gt_image = test_cam.image
                    if isinstance(gt_image, torch.Tensor):
                        gt_image_np = (torch.clamp(gt_image[:3], 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        gt_image_pil = Image.fromarray(gt_image_np)
                        gt_filename = f"iter_{iteration:06d}_{stage.lower()}_cam{test_cam.colmap_id}_t{test_timestamp}_gt.png"
                        gt_path = os.path.join(verify_dir, gt_filename)
                        gt_image_pil.save(gt_path)
                        print(f"Saved ground truth image: {gt_path}")
                        gt_saved = True
                except Exception as e:
                    print(f"Could not save ground truth image from tensor: {e}")
            
            if not gt_saved:
                # Try to load from scene if lazy loading
                try:
                    if hasattr(test_cam, 'image_path') and test_cam.image_path:
                        from utils.general_utils import PILtoTorch
                        gt_image_pil = Image.open(test_cam.image_path)
                        gt_image_tensor = PILtoTorch(gt_image_pil, test_cam.resolution)[:3, ...]
                        if hasattr(test_cam, 'im_scale'):
                            gt_image_tensor = gt_image_tensor / test_cam.im_scale
                        gt_image_np = (torch.clamp(gt_image_tensor, 0.0, 1.0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
                        gt_image_pil = Image.fromarray(gt_image_np)
                        gt_filename = f"iter_{iteration:06d}_{stage.lower()}_cam{test_cam.colmap_id}_t{test_timestamp}_gt.png"
                        gt_path = os.path.join(verify_dir, gt_filename)
                        gt_image_pil.save(gt_path)
                        print(f"Saved ground truth image: {gt_path}")
                except Exception as e:
                    print(f"Could not load/save ground truth image: {e}")
        
        print(f"Rendered image stats: mean={image.mean().item():.4f}, std={image.std().item():.4f}, min={image.min().item():.4f}, max={image.max().item():.4f}")
        print(f"Visible Gaussians: {visibility_filter.sum().item() if visibility_filter is not None else 0}")
    
    # Get number of keyframes before checking dynamic Gaussians
    num_keyframes = gaussians.keyframe_num if hasattr(gaussians, 'keyframe_num') else 0
    print(f"\nNumber of keyframes: {num_keyframes}")
    print(f"Duration: {gaussians.duration}, Interval: {gaussians.interval}")
    
    # Print static Gaussians (first 3)
    static_num = gaussians._xyz.shape[0]
    print(f"\n{'='*80}")
    print(f"STATIC GAUSSIANS (showing first 3 of {static_num})")
    print(f"{'='*80}")
    
    num_to_show = min(3, static_num)
    for i in range(num_to_show):
        print(f"\n--- Static Gaussian {i} ---")
        print(f"  Position (xyz): {gaussians._xyz[i].cpu().numpy()}")
        print(f"  Displacement (xyz_disp): {gaussians._xyz_disp[i].cpu().numpy() if gaussians._xyz_disp.shape[0] > i else 'N/A'}")
        print(f"  Position at t={test_timestamp}: {gaussians.get_static_xyz_at_t(test_timestamp)[i].cpu().numpy()}")
        print(f"  Opacity: {gaussians.get_opacity[i].item():.6f}")
        print(f"  Scaling: {gaussians.get_static_scaling[i].cpu().numpy()}")
        print(f"  Rotation: {gaussians._rotation[i].cpu().numpy()}")
    
    # Print dynamic Gaussians (first 3)
    dynamic_num = gaussians._xyz_motion.shape[0] if gaussians._xyz_motion.numel() > 0 else 0
    print(f"\n{'='*80}")
    print(f"DYNAMIC GAUSSIANS (showing first 3 of {dynamic_num})")
    print(f"{'='*80}")
    
    if dynamic_num > 0:
        num_to_show = min(3, dynamic_num)
        for i in range(num_to_show):
            print(f"\n--- Dynamic Gaussian {i} ---")
            
            # Keyframe positions (interpolation inputs)
            keyframes_xyz = gaussians._xyz_motion[i].cpu().numpy()  # [num_keyframes, 3]
            print(f"  Keyframe positions (input to interpolation):")
            for kf_idx in range(keyframes_xyz.shape[0]):
                print(f"    Keyframe {kf_idx}: {keyframes_xyz[kf_idx]}")
            
            # Interpolated position at test_timestamp (output)
            t = test_timestamp + gaussians.time_shift
            t_idx = int(t // gaussians.interval)
            delta_t = (t % gaussians.interval) / gaussians.interval
            
            print(f"  Interpolation parameters for t={test_timestamp}:")
            print(f"    t_idx={t_idx}, delta_t={delta_t:.4f}")
            
            # Get interpolated position
            interp_xyz = gaussians.get_dynamic_xyz_at_t(test_timestamp)[i].cpu().numpy()
            print(f"  Interpolated position (output): {interp_xyz}")
            
            # Keyframe rotations
            keyframes_rot = gaussians._rotation_motion[i].cpu().numpy()  # [num_keyframes, 4]
            print(f"  Keyframe rotations (input to interpolation):")
            for kf_idx in range(keyframes_rot.shape[0]):
                print(f"    Keyframe {kf_idx}: {keyframes_rot[kf_idx]}")
            
            # Interpolated rotation
            interp_rot = gaussians.get_dynamic_rotation_at_t(test_timestamp)[i].cpu().numpy()
            print(f"  Interpolated rotation (output): {interp_rot}")
            
            # Opacity information
            base_opacity = gaussians.get_motion_opacity[i].item()
            opacity_center = gaussians._opacity_duration_center[i].cpu().numpy()
            opacity_var = gaussians._opacity_duration_var[i].cpu().numpy()
            
            print(f"  Opacity window center: {opacity_center}")
            print(f"  Opacity window variance: {opacity_var}")
            print(f"  Base opacity: {base_opacity:.6f}")
            
            # Opacity at test_timestamp
            t_normalized = (test_timestamp + gaussians.time_shift) / gaussians.interval
            opacity_at_t = gaussians.get_motion_opacity_at_t(test_timestamp, training=False)[i].item()
            print(f"  Opacity at t={test_timestamp} (normalized t={t_normalized:.4f}): {opacity_at_t:.6f}")
            
            # Scaling
            if gaussians._scaling_motion.numel() > 0 and gaussians._scaling_motion.shape[0] > i:
                scaling = gaussians.get_motion_scaling[i].cpu().numpy()
                print(f"  Scaling: {scaling}")
    else:
        print("  No dynamic Gaussians exist yet.")
    
    print(f"\n{'='*80}\n")


def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from, args):
    first_iter = 0
    tb_writer = prepare_output_and_logger(args)
    GaussianModel = getmodel(dataset.model) # gmodel, gmodelrgbonly
    gaussians = GaussianModel(dataset.sh_degree, dataset.start_duration, dataset.time_interval, dataset.time_pad, 
                              interp_type=dataset.interp_type, rot_interp_type=dataset.rot_interp_type, 
                              time_pad_type=dataset.time_pad_type, var_pad=dataset.var_pad, kernel_size=dataset.kernel_size)
    scene = Scene(dataset, gaussians, use_timepad=True)
    gaussians.training_setup(opt)
    args.duration = dataset.duration
    
    args.save_iterations.append(args.iterations)
    # args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)
    saving_iterations = args.save_iterations
    # testing_iterations = args.test_iterations
    checkpoint_iterations = args.checkpoint_iterations
        
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    # variables for progressive training
    scene.set_sampling_len(dataset.start_duration, sample_every=dataset.sample_every)
    expanded = gaussians.expand_duration(dataset.start_duration)
    
    sample_len = dataset.start_duration
    g_sample_len = dataset.start_duration

    mark_extract = False
    need_extract = True
    mark_last = False
    viewpoint_stack = None
    prune_inv = False
    train_images = None
    e_count = args.extract_every
    
    # Get test camera: 5th frame of first train camera (before shuffle, for consistent testing)
    test_cam = None
    test_cam_timestamp = 5  # 5th frame
    try:
        # Get all training cameras without shuffle to ensure consistency
        all_train_cams_list = list(compress(scene.train_cameras[1.0], scene.samplelist))
        
        if len(all_train_cams_list) > 0:
            # Find camera with timestamp closest to 5 (or exactly 5)
            for cam in all_train_cams_list:
                if abs(cam.timestamp - test_cam_timestamp) < 0.5:  # Allow small tolerance
                    test_cam = cam
                    break
            # If not found, use first camera with timestamp >= 5
            if test_cam is None:
                for cam in all_train_cams_list:
                    if cam.timestamp >= test_cam_timestamp:
                        test_cam = cam
                        break
            # If still not found, use first camera
            if test_cam is None:
                test_cam = all_train_cams_list[0] if len(all_train_cams_list) > 0 else None
            
            if test_cam is not None:
                print(f"\n[INIT] Test camera selected: ID={test_cam.colmap_id}, Timestamp={test_cam.timestamp}")
            else:
                print(f"\n[WARNING] Could not find test camera with timestamp ~{test_cam_timestamp}")
        else:
            print(f"\n[WARNING] No training cameras available for testing")
    except Exception as e:
        print(f"\n[WARNING] Error selecting test camera: {e}")
        test_cam = None
    
    ema_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    
    # Record training start time
    training_start_time = time.time()
    
    # Initial verification (before any expansion)
    if test_cam is not None and first_iter == 1:
        verify_gaussians_before_after_expansion(
            gaussians, test_cam, background, pipe, dataset, 0, stage="INITIAL", model_path=args.model_path
        )
    
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer, far=dataset.far)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack, train_images = scene.getTrainCameras(return_as='generator', shuffle=True, n_job=1, job_batch_size=1)
            viewpoint_stack = viewpoint_stack.copy()
                        
            if iteration > opt.prune_invisible_interval:
                prune_inv = True
            
        viewpoint_cam = viewpoint_stack.pop(0)
        gt_image = next(train_images).cuda()
        
        if mark_last:
            if viewpoint_cam.timestamp >= scene.sample_len - gaussians.interval:
                mark_extract = True
                mark_last = False

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, near=dataset.near, far=dataset.far)
        image, viewspace_point_tensor, viewspace_point_error_tensor, visibility_filter, radii, depth, flow, acc, idxs = \
            render_pkg["render"], render_pkg["viewspace_points"], render_pkg["viewspace_l1points"], render_pkg["visibility_filter"], \
            render_pkg["radii"], render_pkg["depth"], render_pkg["opticalflow"], render_pkg["acc"], render_pkg["dominent_idxs"]

        # Loss
        Ll1 = l1_loss(image, gt_image)
        loss = (1.0 - opt.lambda_dssim) * Ll1 + opt.lambda_dssim * (1.0 - ssim(image, gt_image))
        
        # backtrack register
        if opt.l1_accum:
            l1_errors = (image - gt_image).abs().mean(dim=0)
            ssim_errors = ssim(image, gt_image, reduce=False).mean(dim=0)
            hook_tensor = torch.stack([acc[0], l1_errors, ssim_errors])
            flow_h = flow.register_hook(lambda grad: hook_tensor)
            loss += flow.mean() * 0
                
        # Regularization
        if opt.static_reg > 0 and iteration > opt.progressive_growing_steps + opt.make_dynamic_interval:
            loss += opt.static_reg * torch.log(gaussians._xyz_disp.norm(dim=-1)+0.001).mean()

        if opt.motion_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            diff1 = (gaussians._xyz_motion[:, :1] - gaussians._xyz_motion[:, 1:])
            loss += opt.motion_reg * diff1.norm(dim=-1).mean()
            
        if opt.rot_reg > 0 and iteration > opt.progressive_growing_steps * opt.extract_every + opt.make_dynamic_interval and gaussians._xyz_motion.shape[0] > 0:
            r1 = gaussians._rotation_motion[:, 1:] 
            r2 = gaussians._rotation_motion[:, :-1]
            
            r_i = 1 - (r1 * r2).sum(dim=-1) / r1.norm(dim=-1).clamp_min(1e-6) / r2.norm(dim=-1).clamp_min(1e-6)
            loss += opt.rot_reg * r_i.mean()
            
        loss.backward()
        
        if opt.l1_accum:
            flow_h.remove()
        
        iter_end.record()

        gaussians.mark_error(loss.item(), viewpoint_cam.timestamp)

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = loss.item()
            psnr_log = psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().item()
            if iteration % 10 == 0:
                progress_bar.set_postfix({"PSNR": f"{psnr_log:.{2}f}", "Loss": f"{ema_loss_for_log:.{6}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, (pipe, background), dataset.near, dataset.far)
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

            if opt.l1_accum:
                gaussians.mark_prune_stats(radii, viewspace_point_error_tensor)
            
            # Densification
            if iteration < opt.densify_until_iter:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                static_radii = radii[:static_num]
                dynamic_vis_filter = visibility_filter[static_num:]
                dynamic_radii = radii[static_num:]
                
                gaussians.max_radii2D[static_vis_filter] = torch.max(gaussians.max_radii2D[static_vis_filter], static_radii[static_vis_filter])
                gaussians.motion_max_radii2D[dynamic_vis_filter] = torch.max(gaussians.motion_max_radii2D[dynamic_vis_filter], dynamic_radii[dynamic_vis_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, static_vis_filter, dynamic_vis_filter, static_num)

                if opt.l1_accum:
                    gaussians.add_l1_ssim_stats(viewspace_point_error_tensor, static_vis_filter, dynamic_vis_filter, static_num, viewpoint_cam.timestamp)
                    
                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold, dynamic_size_threshold = None, None
                    s_max_ssim = opt.s_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    s_l1_thres = opt.s_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    
                    d_max_ssim = opt.d_max_ssim  if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.ssim_prune_every) == 0 else 0
                    d_l1_thres = opt.d_l1_thres if iteration > opt.error_base_prune_steps and iteration % (opt.densification_interval * opt.l1_prune_every) == 0 else 100
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 
                                                opt.densify_dgrad_threshold, 
                                                0.01, 0.01, scene.cameras_extent, size_threshold, dynamic_size_threshold,
                                                s_max_ssim=s_max_ssim, s_l1_thres=s_l1_thres, d_max_ssim=d_max_ssim, d_l1_thres=d_l1_thres)
                elif iteration > opt.extract_from_iter and iteration % opt.extracton_interval == 0:
                    static_num = gaussians._xyz.shape[0]
                    candidate = gaussians.get_errorneous_timestamp()
                    if not candidate is None:
                        gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), candidate, static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
            if iteration % (opt.densification_interval*4) == 0 and iteration < opt.densify_until_iter - 3000:
                gaussians.adjust_temp_opa(max_dur=sample_len)

            # if prune_inv and iteration < opt.iterations - 5000:
            if prune_inv and iteration < opt.iterations and iteration > 3000:
                gaussians.prune_invisible()
                if opt.l1_accum:
                    gaussians.prune_small()
                prune_inv = False
                
            # Optimizer step
            if iteration < opt.iterations:
                # prevent nan grad of dynamic opacity
                if gaussians._opacity_duration_var.shape[0] != 0:
                    if not gaussians._opacity_duration_var.grad is None:
                        gaussians._opacity_duration_var.grad = gaussians._opacity_duration_var.grad.nan_to_num()
                        
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none=True)
                
                gaussians.prune_nan_points()
                
                torch.cuda.empty_cache()
                
            if iteration > opt.extract_from_iter and (iteration % opt.progressive_growing_steps == opt.make_dynamic_interval) and need_extract :
                mark_last = True
                need_extract = False
                
            # increase duration
            if iteration > opt.extract_from_iter and iteration % opt.progressive_growing_steps == 0 and iteration > opt.progressive_growing_steps and ~need_extract:
                # Verify BEFORE expansion
                if test_cam is not None:
                    verify_gaussians_before_after_expansion(
                        gaussians, test_cam, background, pipe, dataset, iteration, stage="BEFORE", model_path=args.model_path
                    )
                
                sample_len = min(dataset.duration + gaussians.time_shift, int(dataset.time_interval * dataset.progressive_step) + scene.sample_len)

                scene.set_sampling_len(sample_len, sample_every=dataset.sample_every)
                g_sample_len = min(dataset.duration + gaussians.time_shift, sample_len)
                expanded = gaussians.expand_duration(g_sample_len)
                
                # Verify AFTER expansion
                if expanded and test_cam is not None:
                    verify_gaussians_before_after_expansion(
                        gaussians, test_cam, background, pipe, dataset, iteration, stage="AFTER", model_path=args.model_path
                    )
                
                if expanded:
                    e_count += 1
                    if e_count >= opt.extract_every:
                        mark_last = True
                        need_extract = True
                        e_count = 0

            # create dynamic points from static points
            if mark_extract:
                static_num = gaussians._xyz.shape[0]
                static_vis_filter = visibility_filter[:static_num]
                gaussians.extract_dynamic_points_from_static(torch.tensor(viewpoint_cam.T).unsqueeze(0), viewpoint_cam.timestamp, 
                                                             static_vis_filter, scene.cameras_extent, percentile=opt.extract_percentile, max_dur=sample_len)
                mark_extract = False

    # Calculate and print total training time
    training_end_time = time.time()
    total_training_time = training_end_time - training_start_time
    hours = int(total_training_time // 3600)
    minutes = int((total_training_time % 3600) // 60)
    seconds = int(total_training_time % 60)
    
    print(f"\n{'='*50}")
    print(f"TRAINING COMPLETED")
    print(f"{'='*50}")
    print(f"Total training time: {hours:02d}:{minutes:02d}:{seconds:02d} ({total_training_time:.2f} seconds)")
    print(f"Total iterations: {opt.iterations}")
    print(f"Average time per iteration: {total_training_time/opt.iterations:.4f} seconds")
    print(f"{'='*50}")


def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    
    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer


def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderArgs, near, far):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)
        
    # Report test and samples of training set
    if iteration in testing_iterations:
        torch.cuda.empty_cache()

        
        test_viewpoint_stack, test_images = scene.getTestCameras(shuffle=False, return_as='generator', n_job=1, job_batch_size=1)
        test_viewpoint_stack = test_viewpoint_stack.copy()
        
        train_viewpoint_stack, train_images = scene.getTrainCameras(shuffle=False, return_as='generator', n_job=1, job_batch_size=1)
        train_viewpoint_stack = train_viewpoint_stack.copy()
            
        validation_configs = ({'name': 'test', 'cameras': test_viewpoint_stack, 
                                                'images': test_images}, 
                              {'name': 'train', 'cameras': train_viewpoint_stack, 
                                                 'images': train_images})
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                l1_test = 0.0
                psnr_test = 0.0
                count = 0
                
                for idx, viewpoint in enumerate(config['cameras']):
                    if config['name'] == 'train':
                        sampled_idx_list = [idx % len(train_viewpoint_stack) for idx in range(5, 30, 5)]
                        if not idx in sampled_idx_list:
                            _ = next(config['images'])
                            continue
                    
                    gt_image = next(config['images']).cuda()
                    rend_pkg = render(viewpoint, scene.gaussians, near=near, far=far, *renderArgs)
                    image = torch.clamp(rend_pkg["render"], 0.0, 1.0).cuda()
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double().detach().item()
                    psnr_test += psnr(image.unsqueeze(0), gt_image.unsqueeze(0)).mean().double().detach().item()
                    count += 1
                    
                    del(rend_pkg)
                    del(image)
                    del(gt_image)
                    del(viewpoint)
                    torch.cuda.empty_cache()
                    
                psnr_test /= count
                l1_test /= count
                print("\n[ITER {}] Evaluating {}: L1 {} PSNR {}".format(iteration, config['name'], l1_test, psnr_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_histogram("scene/motion_opacity_histogram", scene.gaussians.get_motion_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians._xyz.shape[0]+scene.gaussians._xyz_motion.shape[0], iteration)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7000_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[30_000, 40_000])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    parser.add_argument("--configpath", type=str, default = "None")
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    # args.test_iterations.append(args.iterations)
    args.checkpoint_iterations.append(args.iterations)

    print("Optimizing " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)
    
    # incase we provide config file not directly pass to the file
    if os.path.exists(args.configpath) and args.configpath != "None":
        print("overload config from " + args.configpath)
        config = json.load(open(args.configpath))
        for k in config.keys():
            try:
                value = getattr(args, k) 
                newvalue = config[k]
                setattr(args, k, newvalue)
            except:
                print("failed set config: " + k)
        print("finish load config from " + args.configpath)
    else:
        raise ValueError("config file not exist or not provided")

    # Start GUI server, configure and run training
    while True:
        try:
            network_gui.init(args.ip, args.port)
            break
        except:
            args.port += 1
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from, args)

    # All done
    print("\nTraining complete.")
