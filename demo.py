import sys
sys.path.append('droid_slam')

from tqdm import tqdm
import numpy as np
import torch
import lietorch
import cv2
import os
import glob 
import time
import argparse

from torch.multiprocessing import Process
from droid import Droid
from lietorch import SE3
import torch.nn.functional as F
import droid_backends
from skimage.transform import resize
import pytransform3d.transformations as pt
import pickle

def show_image(image):
    image = image.permute(1, 2, 0).cpu().numpy()
    cv2.imshow('image', image / 255.0)
    cv2.waitKey(1)

def image_stream(imagedir, calib, stride, posedir=None, validposedir=None, trans_scale=1.):
    """ image generator """

    calib = np.loadtxt(calib, delimiter=" ")
    fx, fy, cx, cy = calib[:4]

    K = np.eye(3)
    K[0,0] = fx
    K[0,2] = cx
    K[1,1] = fy
    K[1,2] = cy

    extension = '.' + os.listdir(imagedir)[0].split('.')[-1]
    prefix = "".join([s for s in os.listdir(imagedir)[0].strip(extension) if not s.isdigit()])
    image_list = sorted(os.listdir(imagedir))[::stride]

    if posedir is not None:
        pose_arr = np.load(posedir)
        pose_R = pose_arr['R'][::stride]  # (N_img, 3, 3)
        pose_T = pose_arr['T'][::stride] * trans_scale  # (N_img, 3)

        # Convert PyTorch3D coordinate frame to DROID-SLAM coordinate frame
        R_convert = np.array([[-1., 0, 0], [0, -1., 0], [0, 0, 1.]])
        pose_R = pose_R @ R_convert[None, ...]
        pose_T = pose_T @ R_convert
    
        # Keep track of valid poses provided from posedir
        if validposedir is not None:
            with open(validposedir, 'rb') as f:
                valid_pose_list = pickle.load(f)
            pose_valid = np.zeros([pose_T.shape[0]], dtype=bool)

            for img_idx, img_file in valid_pose_list:
                if img_idx % stride == 0:
                    pose_valid[img_idx // stride] = True
        else:
            pose_valid = np.ones([pose_T.shape[0]], dtype=bool)

    for t, imfile in enumerate(image_list):
        image = cv2.imread(os.path.join(imagedir, imfile))
        if len(calib) > 4:
            image = cv2.undistort(image, K, calib[4:])

        h0, w0, _ = image.shape
        h1 = int(h0 * np.sqrt((384 * 512) / (h0 * w0)))
        w1 = int(w0 * np.sqrt((384 * 512) / (h0 * w0)))

        image = cv2.resize(image, (w1, h1))
        image = image[:h1-h1%8, :w1-w1%8]
        image = torch.as_tensor(image).permute(2, 0, 1)

        intrinsics = torch.as_tensor([fx, fy, cx, cy])
        intrinsics[0::2] *= (w1 / w0)
        intrinsics[1::2] *= (h1 / h0)

        if posedir is not None:
            curr_R = pose_R[t]
            curr_T = pose_T[t]
            curr_pose = np.eye(4)

            curr_pose[:3, :3] = curr_R.T
            curr_pose[:3, 3] = curr_T

            curr_pq = pt.pq_from_transform(curr_pose)  # (x, y, z, qw, qx, qy, qz)
            
            # Convert (x, y, z, qw, qx, qy, qz) -> (x, y, z, qx, qy, qz, qw)
            qw = curr_pq[3]
            curr_pq[3:6] = curr_pq[4:]
            curr_pq[-1] = qw
            curr_pq = torch.from_numpy(curr_pq).to("cuda")  # (6, )

            """
            Note the following command will give the original 4x4 matrix
            
                curr_pose = SE3(curr_pq).matrix()
            
            """

            # Yield poses only if valid
            if pose_valid[t]:
                yield t, image[None], intrinsics, imfile, curr_pq
            else:
                yield t, image[None], intrinsics, imfile, None
        else:
            yield t, image[None], intrinsics, imfile, None


def save_reconstruction(droid, reconstruction_path, trans_scale):

    from pathlib import Path
    import random
    import string
    device = droid.video.poses.device
    t = droid.video.counter.value
    tstamps = droid.video.tstamp[:t].cpu().numpy()
    images = droid.video.images[:t].cpu().numpy()
    disps = droid.video.disps_up[:t].cpu().numpy()
    poses = droid.video.poses[:t].cpu().numpy()
    poses_mtx = SE3(droid.video.poses[:t]).inv().matrix().cpu().numpy()
    intrinsics = droid.video.intrinsics[:t].cpu().numpy()

    # Save masks
    default_filter_thresh = 0.005  # Tunable parameter for filtering depth
    thresh = torch.ones_like(droid.video.disps.mean(dim=[1,2])) * default_filter_thresh
    count = droid_backends.depth_filter(
        droid.video.poses[:t], droid.video.disps[:t], droid.video.intrinsics[0], torch.arange(t).to(device), thresh[:t])
    masks = ((count >= 2) & (droid.video.disps[:t] > .5*droid.video.disps[:t].mean(dim=[1,2], keepdim=True)))
    masks = np.transpose(masks.cpu().numpy(), (1, 2, 0))
    masks = resize(masks, (masks.shape[0] * 8, masks.shape[1] * 8), order=0, preserve_range=True)
    masks = np.transpose(masks, (2, 0, 1))

    Path("reconstructions/{}".format(reconstruction_path)).mkdir(parents=True, exist_ok=True)
    np.save("reconstructions/{}/tstamps.npy".format(reconstruction_path), tstamps)
    np.save("reconstructions/{}/images.npy".format(reconstruction_path), images)
    np.save("reconstructions/{}/disps.npy".format(reconstruction_path), disps)
    np.save("reconstructions/{}/poses.npy".format(reconstruction_path), poses)
    np.save("reconstructions/{}/poses_mtx.npy".format(reconstruction_path), poses_mtx)
    np.save("reconstructions/{}/intrinsics.npy".format(reconstruction_path), intrinsics)
    np.save("reconstructions/{}/masks.npy".format(reconstruction_path), masks)
    with open("reconstructions/{}/scale.txt".format(reconstruction_path), 'w') as f:
        f.write(f"{trans_scale}")

    with open("reconstructions/{}/images.txt".format(reconstruction_path), 'w') as f:
        for imfile in droid.video.imfile_list:
            f.write(os.path.join('images/', imfile + '\n'))

    # Save pcd
    poses = droid.video.poses[:t]
    disps = droid.video.disps[:t]
    Ps = SE3(poses).inv().matrix().cpu().numpy()

    images = droid.video.images[:t]
    images = images.cpu()[:,[2,1,0],3::8,3::8].permute(0,2,3,1) / 255.0
    points = droid_backends.iproj(SE3(poses).inv().data, disps, droid.video.intrinsics[0]).cpu()

    thresh = default_filter_thresh * torch.ones_like(disps.mean(dim=[1,2]))

    count = count.cpu()
    disps = disps.cpu()
    masks = ((count >= 2) & (disps > .5*disps.mean(dim=[1,2], keepdim=True)))

    pts_list = []
    clr_list = []
    
    for i in range(t):
        mask = masks[i].reshape(-1)
        pts = points[i].reshape(-1, 3)[mask].cpu().numpy()
        clr = images[i].reshape(-1, 3)[mask].cpu().numpy()
        
        pts_list.append(pts)
        clr_list.append(clr)
    
    pts = np.concatenate(pts_list, axis=0) / trans_scale
    clr = np.concatenate(clr_list, axis=0)

    pcd = np.concatenate([pts, clr], axis=-1)
    np.savetxt("reconstructions/{}/point_cloud.txt".format(reconstruction_path), pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--imagedir", type=str, help="path to image directory")
    parser.add_argument("--calib", type=str, help="path to calibration file")
    parser.add_argument("--t0", default=0, type=int, help="starting frame")
    parser.add_argument("--stride", default=3, type=int, help="frame stride")

    parser.add_argument("--weights", default="droid.pth")
    parser.add_argument("--buffer", type=int, default=512)
    parser.add_argument("--image_size", default=[240, 320])
    parser.add_argument("--disable_vis", action="store_true")

    parser.add_argument("--beta", type=float, default=0.3, help="weight for translation / rotation components of flow")
    parser.add_argument("--filter_thresh", type=float, default=2.4, help="how much motion before considering new keyframe")
    parser.add_argument("--warmup", type=int, default=8, help="number of warmup frames")
    parser.add_argument("--keyframe_thresh", type=float, default=4.0, help="threshold to create a new keyframe")
    parser.add_argument("--frontend_thresh", type=float, default=16.0, help="add edges between frames whithin this distance")
    parser.add_argument("--frontend_window", type=int, default=25, help="frontend optimization window")
    parser.add_argument("--frontend_radius", type=int, default=2, help="force edges between frames within radius")
    parser.add_argument("--frontend_nms", type=int, default=1, help="non-maximal supression of edges")

    parser.add_argument("--pose_file", type=str, help="path to pose file", default=None)
    parser.add_argument("--valid_pose_file", type=str, help="path to file labeling valid poses", default=None)
    parser.add_argument("--backend_thresh", type=float, default=22.0)
    parser.add_argument("--backend_radius", type=int, default=2)
    parser.add_argument("--backend_nms", type=int, default=3)
    parser.add_argument("--upsample", action="store_true")
    parser.add_argument("--reconstruction_path", help="path to saved reconstruction")
    args = parser.parse_args()

    args.stereo = False
    torch.multiprocessing.set_start_method('spawn')

    droid = None

    # If pose is known, only optimize disparity
    if args.pose_file is not None:
        disp_only = True
        trans_scale = 0.1  # Hard-coded for optimal reconstruction with metric scale poses
    else:
        disp_only = False
        trans_scale = 1.0

    # need high resolution depths
    if args.reconstruction_path is not None:
        args.upsample = True

    tstamps = []
    for (t, image, intrinsics, imfile, curr_pose) in tqdm(image_stream(args.imagedir, args.calib, args.stride, args.pose_file, args.valid_pose_file, trans_scale)):
        if t < args.t0:
            continue

        if not args.disable_vis:
            show_image(image[0])

        if droid is None:
            args.image_size = [image.shape[2], image.shape[3]]
            droid = Droid(args)
        
        # Start tracking if a valid pose is found, or if poses are also optimized
        if disp_only:  # If using pre-computed poses, track only for valid poses
            if curr_pose is not None:
                droid.track(t, image, intrinsics=intrinsics, imfile=imfile, disp_only=True, pose=curr_pose)
        else:  # Poses are also optimized, so start tracking right away
            droid.track(t, image, intrinsics=intrinsics, imfile=imfile, disp_only=False, pose=curr_pose)
    
    if not disp_only:  # Only do global BA if poses are not given
        traj_est = droid.terminate(image_stream(args.imagedir, args.calib, args.stride), disp_only=False)
    
    if args.reconstruction_path is not None:
        save_reconstruction(droid, args.reconstruction_path, trans_scale)
