import argparse
import open3d as o3d
import numpy as np
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--recondir", type=str, help="Root directory of SLAM reconstruction")

    args = parser.parse_args()

    warmup = 8
    default_filter_thresh = 0.005

    disp_arr = np.load(os.path.join(args.recondir, 'disps.npy'))
    img_arr = np.load(os.path.join(args.recondir, 'images.npy'))
    intrinsics_arr = np.load(os.path.join(args.recondir, 'intrinsics.npy')) * 8.0  # Droid-SLAM applies x8 multiplication
    poses_arr = np.load(os.path.join(args.recondir, 'poses_mtx.npy'))
    masks_arr = np.load(os.path.join(args.recondir, 'masks.npy'))

    pcd = np.loadtxt(os.path.join(args.recondir, 'point_cloud.txt'))

    pts = pcd[:, :3]
    clr = pcd[:, 3:]

    CAM_POINTS = np.array([
            [ 0,   0,   0],
            [-1,  -1, 1.5],
            [ 1,  -1, 1.5],
            [ 1,   1, 1.5],
            [-1,   1, 1.5],
            [-0.5, 1, 1.5],
            [ 0.5, 1, 1.5],
            [ 0, 1.2, 1.5]])

    CAM_LINES = np.array([
        [1,2], [2,3], [3,4], [4,1], [1,0], [0,2], [3,0], [0,4], [5,7], [7,6]])

    def animation_callback(vis):
        for i in range(len(disp_arr)):
            pose = poses_arr[i]
            ### add camera actor ###
            cam_actor = o3d.geometry.LineSet(
                points=o3d.utility.Vector3dVector(0.05 * CAM_POINTS),
                lines=o3d.utility.Vector2iVector(CAM_LINES))

            g = True
            color = (g * 1.0, 0.5 * (1-g), 0.9 * (1-g))
            cam_actor.paint_uniform_color(color)
            cam_actor.transform(pose)
            vis.add_geometry(cam_actor)
        point_actor = o3d.geometry.PointCloud()
        point_actor.points = o3d.utility.Vector3dVector(pts)
        point_actor.colors = o3d.utility.Vector3dVector(clr)

        vis.add_geometry(point_actor)

        vis.poll_events()
        vis.update_renderer()
        vis.register_animation_callback(None)
        return False

    ### create Open3D visualization ###
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.register_animation_callback(animation_callback)

    vis.create_window(height=540, width=960)
    vis.get_render_option().load_from_json("misc/renderoption.json")

    vis.run()
    vis.destroy_window()
