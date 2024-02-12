import os
from os.path import join as p_join
import time

primary_device = "cuda:0"

#scenes = ["freiburg3_walking_halfsphere" ,"freiburg2_desk_with_person","freiburg3_walking_static", "freiburg3_walking_xyz", "freiburg3_sitting_rpy", "freiburg1_desk", "freiburg1_desk2", "freiburg1_room", "freiburg2_xyz", "freiburg3_long_office_household", "freiburg3_sitting_rpy"]
# scenes = ['freiburg2_desk_with_person']
# scene_name = scenes[int(0)]

seed = int(0)

time_str = time.strftime("y%ym%md%dh%Hm%M")

map_every = 1
keyframe_every = 5
mapping_window_size = 20
tracking_iters = 100
mapping_iters = 30
scene_radius_depth_ratio = 2

queue = [dict(scene_name="freiburg3_sitting_rpy",
             yolo_mapping=True,
             yolo_tracking=True,
             yolo_boxmask=True,
             yolo_dilation=None,
             inpainting=False),
             
             dict(scene_name="freiburg3_sitting_rpy",
             yolo_mapping=False,
             yolo_tracking=True,
             yolo_boxmask=True,
             yolo_dilation=None,
             inpainting=False)]

group_name = "TUM"

configs = []

for i,expr in enumerate(queue):
    #run_name = f"{expr["scene_name"]}_{time_str}_{i}_yolomap{expr["yolo_mapping"]}_yolotrack{expr["yolo_tracking"]}_junk"
    run_name = expr["scene_name"]+"_" + time_str + "_" + str(i)
    run_name += "_yolomap" + str(expr["yolo_mapping"])
    run_name += "_yolotrack" + str(expr["yolo_tracking"])
    run_name += "_dil"+ str(expr["yolo_dilation"])
    run_name += "_box" if expr["yolo_boxmask"] else "_seg"

    cnfg = dict(
    workdir=f"./experiments/{group_name}",
    run_name=run_name,
    max_frames=200,
    seed=seed,
    yolo_mapping=expr["yolo_mapping"],
    yolo_tracking=expr["yolo_tracking"],
    yolo_dilation=expr["yolo_dilation"],
    yolo_boxmask=expr["yolo_boxmask"],
    inpainting = expr['inpainting']
    primary_device=primary_device,
    map_every=map_every, # Mapping every nth frame
    keyframe_every=keyframe_every, # Keyframe every nth frame
    mapping_window_size=mapping_window_size, # Mapping window size
    report_global_progress_every=500, # Report Global Progress every nth frame
    eval_every=500, # Evaluate every nth frame (at end of SLAM)
    scene_radius_depth_ratio=scene_radius_depth_ratio, # Max First Frame Depth to Scene Radius Ratio (For Pruning/Densification)
    mean_sq_dist_method="projective", # ["projective", "knn"] (Type of Mean Squared Distance Calculation for Scale of Gaussians)
    report_iter_progress=False,
    load_checkpoint=False,
    checkpoint_time_idx=0,
    save_checkpoints=False, # Save Checkpoints
    checkpoint_interval=100, # Checkpoint Interval
    use_wandb=True,
    wandb=dict(
        entity="azizi_delkhah",
        project="SplaTAM",
        group=group_name,
        name=run_name,
        save_qual=False,
        eval_save_qual=True,
    ),
    data=dict(
        basedir="./data/TUM_RGBD",
        gradslam_data_cfg=f"./configs/data/TUM/" + expr["scene_name"]+".yaml",
        sequence=f"rgbd_dataset_" + expr["scene_name"],
        desired_image_height=480,
        desired_image_width=640,
        start=0,
        end=-1,
        stride=1,
        num_frames=-1,
    ),
    tracking=dict(
        use_gt_poses=False, # Use GT Poses for Tracking
        forward_prop=True, # Forward Propagate Poses
        num_iters=tracking_iters,
        use_sil_for_loss=True,
        sil_thres=0.99,
        use_l1=True,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        visualize_tracking_loss=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0,
            rgb_colors=0.0,
            unnorm_rotations=0.0,
            logit_opacities=0.0,
            log_scales=0.0,
            cam_unnorm_rots=0.002,
            cam_trans=0.002,
        ),
    ),
    mapping=dict(
        num_iters=mapping_iters,
        add_new_gaussians=True,
        sil_thres=0.5, # For Addition of new Gaussians
        use_l1=True,
        use_sil_for_loss=False,
        ignore_outlier_depth_loss=False,
        use_uncertainty_for_loss_mask=False,
        use_uncertainty_for_loss=False,
        use_chamfer=False,
        loss_weights=dict(
            im=0.5,
            depth=1.0,
        ),
        lrs=dict(
            means3D=0.0001,
            rgb_colors=0.0025,
            unnorm_rotations=0.001,
            logit_opacities=0.05,
            log_scales=0.001,
            cam_unnorm_rots=0.0000,
            cam_trans=0.0000,
        ),
        prune_gaussians=True, # Prune Gaussians during Mapping
        pruning_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=0,
            remove_big_after=0,
            stop_after=20,
            prune_every=20,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities=False,
            reset_opacities_every=500, # Doesn't consider iter 0
        ),
        use_gaussian_splatting_densification=False, # Use Gaussian Splatting-based Densification during Mapping
        densify_dict=dict( # Needs to be updated based on the number of mapping iterations
            start_after=500,
            remove_big_after=3000,
            stop_after=5000,
            densify_every=100,
            grad_thresh=0.0002,
            num_to_split_into=2,
            removal_opacity_threshold=0.005,
            final_removal_opacity_threshold=0.005,
            reset_opacities_every=3000, # Doesn't consider iter 0
        ),
    ),
    viz=dict(
        render_mode='color', # ['color', 'depth' or 'centers']
        offset_first_viz_cam=True, # Offsets the view camera back by 0.5 units along the view direction (For Final Recon Viz)
        show_sil=False, # Show Silhouette instead of RGB
        visualize_cams=True, # Visualize Camera Frustums and Trajectory
        viz_w=600, viz_h=340,
        viz_near=0.01, viz_far=100.0,
        view_scale=2,
        viz_fps=5, # FPS for Online Recon Viz
        enter_interactive_post_online=False, # Enter Interactive Mode after Online Recon Viz
    ),
)
    configs.append(cnfg)
