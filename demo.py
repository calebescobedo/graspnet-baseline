""" Demo to show prediction results.
    Author: chenxi-wang
"""

import os
import sys
import numpy as np
import open3d as o3d
import argparse
import importlib
import scipy.io as scio
from PIL import Image
import pandas as pd
import numpy as np
import re

# Caleb import for kinect stream
import rospy
from sensor_msgs.msg import Image as ros_img
from sensor_msgs.msg import PointCloud2, PointField

# Noah import for Ros Publisher
from rospy.numpy_msg import numpy_msg as np_msg
from std_msgs.msg import Float32MultiArray

import cv2
from cv_bridge import CvBridge, CvBridgeError
import sensor_msgs.point_cloud2 as pc2
import ctypes
from std_msgs.msg import Header
import struct

import torch
from graspnetAPI import GraspGroup

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(ROOT_DIR, 'models'))
sys.path.append(os.path.join(ROOT_DIR, 'dataset'))
sys.path.append(os.path.join(ROOT_DIR, 'utils'))

from graspnet import GraspNet, pred_decode
from graspnet_dataset import GraspNetDataset
from collision_detector import ModelFreeCollisionDetector
from data_utils import CameraInfo, create_point_cloud_from_depth_image

parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', required=True, help='Model checkpoint path')
parser.add_argument('--num_point', type=int, default=20000, help='Point Number [default: 20000]')
parser.add_argument('--num_view', type=int, default=300, help='View Number [default: 300]')
parser.add_argument('--collision_thresh', type=float, default=0.01, help='Collision Threshold in collision detection [default: 0.01]')
parser.add_argument('--voxel_size', type=float, default=0.01, help='Voxel Size to process point clouds before collision detection [default: 0.01]')
cfgs = parser.parse_args()

bridge = CvBridge()

def get_net():
    # Init the model
    net = GraspNet(input_feature_dim=0, num_view=cfgs.num_view, num_angle=12, num_depth=4,
            cylinder_radius=0.05, hmin=-0.02, hmax_list=[0.01,0.02,0.03,0.04], is_training=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    # Load checkpoint
    checkpoint = torch.load(cfgs.checkpoint_path)
    net.load_state_dict(checkpoint['model_state_dict'])
    start_epoch = checkpoint['epoch']
    net.eval()
    return net

def get_grasps(net, end_points):
    # Forward pass
    with torch.no_grad():
        end_points = net(end_points)
        grasp_preds = pred_decode(end_points)
    gg_array = grasp_preds[0].detach().cpu().numpy()
    gg = GraspGroup(gg_array)
    return gg

def collision_detection(gg, cloud):
    mfcdetector = ModelFreeCollisionDetector(cloud, voxel_size=cfgs.voxel_size)
    collision_mask = mfcdetector.detect(gg, approach_dist=0.05, collision_thresh=cfgs.collision_thresh)
    gg = gg[~collision_mask]
    return gg

def add_geos(gg, vis):
    vis.add_geometry(gg[0].to_open3d_geometry(color=(0,0,0)))
    vis.add_geometry(gg[1].to_open3d_geometry(color=((0,0,255))))
    vis.add_geometry(gg[2].to_open3d_geometry(color=(255,0,0)))
    vis.add_geometry(gg[3].to_open3d_geometry(color=(0,255,0)))
    return vis

def find_grasps(gg, cloud):
    rotated = False
    flag = True
    dct = {}
    colors = ("black","blue","red","green")
    for i in range(4):
        dct[colors[i]] = gg[i]
    ret = []
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=900)
    # print("grasp after processing:", "1", gg[0], "2", gg[1], "3", gg[2], "4", gg[3])
    vis = add_geos(gg, vis)
    vis.add_geometry(cloud)


    while flag:
        
        if not vis.poll_events():
            print('Please input corresponding color to best grasp\n case sensitive: red, blue, green, black')
            color = str(input())
            print(color)
            ret.append((dct[color], color))
            flag = False
            break 
        if not rotated:
            ctr = vis.get_view_control()
            ctr.rotate(x=0.0, y=900.0)
            rotated = True
        vis.update_renderer()
    vis.destroy_window()
    return ret

def save_grasp(grasp, full_file_path, grasp_info):
    print(grasp)
    location = int(input())
    df = {'object_type': [grasp_info], 'location':[location], 'grasp_rotation':[grasp.rotation_matrix.flatten()], 'grasp_translation': [grasp.translation], 'rotation_position':0}
    columns = ['object_type', 'location', 'grasp_rotation', 'grasp_translation']
    df = pd.DataFrame.from_dict(df)
    df.to_csv(full_file_path, mode='a', header=False)
    
    return
 
def vis_grasps(gg, cloud):
    colors = {"black":(0,0,0),"blue":(0,0,255),"red":(255,0,0),"green":(0,255,0)}
    gg.nms()
    gg.sort_by_score()
    gg = sort_by_rot(gg)
    gg = find_grasps(gg[:4], cloud)
    best_grasp = gg[0][0]
    print("BEST GRASP", best_grasp)
    print("Width:", best_grasp.width, "Height:", best_grasp.height, "Depth:", best_grasp.depth)
    rotated = False
    
    # grippers = gg[0][0].to_open3d_geometry(color=colors[gg[0][1]])
    grasp_info = 'box'
    grasp_save_file_path = "/home/noah/datasets/grasp_data.csv"
    save_grasp(gg[0][0], grasp_save_file_path, grasp_info)
    
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=900)
    
    #Create sphere around best grasp to remove object from scene
    # center = np.array([best_grasp.width, best_grasp.height, best_grasp.depth])
    # radius = 0.05
    # points = np.asarray(cloud.points)
    # distances = np.linalg.norm(points - center, axis=1)
    # cloud.points = o3d.utility.Vector3dVector(points[distances <= radius])

    vis.add_geometry(gg[0][0].to_open3d_geometry(color=colors[gg[0][1]]))
    vis.add_geometry(cloud)
    ggg = gg[0][0]
    grasp_talker(ggg)
    while True:
        if not vis.poll_events(): break
        if not rotated:
            ctr = vis.get_view_control()
            ctr.rotate(x=0.0, y=900.0)
            rotated = True
        
        vis.update_renderer()

# caleb -  
def sort_by_rot(gg):

    y_value = gg.grasp_group_array[:, 14]
    index = []
    for idx, y_val in enumerate(y_value):
        if y_val > -0.383 :
            index.append(idx)
    new_list = [gg[int(i)] for i in index]
    sorted_grasp = GraspGroup()
    for i in new_list:
        sorted_grasp.add(i)
    return sorted_grasp

def grasp_talker(grasp):
    pub = rospy.Publisher('/grasp_net/grasp6d', Float32MultiArray, queue_size=1)
    msg = Float32MultiArray()
    msg.data = np.concatenate((grasp.translation, list(grasp.rotation_matrix.flatten())))
    pub.publish(msg)

class Kinect():
    def __init__(self):
        self.cur_depth = None
        self.cur_color = None
        self.data_dir = 'doc/example_data'
    
    def kinect_depth_cb(self, msg):
        cv2_img = bridge.imgmsg_to_cv2(msg)
        self.cur_depth = np.array(cv2_img)

        self.demo(self.data_dir)    

    def kinect_color_cb(self, msg):
        global cur_color
        cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
        cur_color = np.array(cv2_img)

    def demo(self, data_dir):
        net = get_net()
        end_points, cloud = self.process_kinect_data(data_dir)
        gg = GraspGroup()
        for i in range(1):
            graspsps = get_grasps(net, end_points)
            gg.add(graspsps)


        if cfgs.collision_thresh > 0:
            gg = collision_detection(gg, np.array(cloud.points))

        # What should the structure of grasp info be?
        # location[1-18], rotation[0, 45, 90, 135], object[box, cylinder, mustard]
        vis_grasps(gg,cloud)

    def process_kinect_data(self, data_dir):
        file_name = "kumar_converted.png"
        img = Image.fromarray(cur_color, "RGB")
        img.save(file_name)
        # color = np.array(cur_color, dtype=np.float32)

        color = np.array(cur_color, dtype=np.float32) / 255.0
        depth = self.cur_depth
        print("DEPTH START", depth, depth.size)

        workspace_mask = np.array(Image.open(os.path.join(data_dir, 'kumar_converted.png')))
        meta = scio.loadmat(os.path.join(data_dir, 'meta.mat'))
        intrinsic = meta['intrinsic_matrix']
        factor_depth = meta['factor_depth']
        camera = CameraInfo(1920.0, 1080.0, 1.0534e+03, 1.0528e+03, 9.4948e+02, 5.5079e+02, factor_depth)
        cloud = create_point_cloud_from_depth_image(depth, camera, organized=True)

        # get valid points
        mask = (workspace_mask & (depth > 0))
        cloud_masked = cloud[mask]
        color_masked = color[mask]
        # cloud_masked = np.array(cloud_masked, dtype="float32")

        # sample points
        if len(cloud_masked) >= cfgs.num_point:
            idxs = np.random.choice(len(cloud_masked), cfgs.num_point, replace=False)
        else:
            idxs1 = np.arange(len(cloud_masked))
            idxs2 = np.random.choice(len(cloud_masked), cfgs.num_point-len(cloud_masked), replace=True)
            idxs = np.concatenate([idxs1, idxs2], axis=0)
        cloud_sampled = cloud_masked[idxs]
        color_sampled = color_masked[idxs]

        # convert data
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(cloud_masked.astype(np.float32))
        cloud.colors = o3d.utility.Vector3dVector(color_masked.astype(np.float32))
        end_points = dict()
        cloud_sampled = torch.from_numpy(cloud_sampled[np.newaxis].astype(np.float32))
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        cloud_sampled = cloud_sampled.to(device)
        end_points['point_clouds'] = cloud_sampled
        end_points['cloud_colors'] = color_sampled

        return end_points, cloud


if __name__=='__main__':
    kinect = Kinect()
    rospy.init_node('kinect_to_grasp_prop')
    image_color_topic = "/kinect2/hd/image_color_rect"
    image_depth_topic = "/kinect2/hd/image_depth_rect"
    rospy.Subscriber(image_color_topic, ros_img, kinect.kinect_color_cb, queue_size=1)
    rospy.Subscriber(image_depth_topic, ros_img, kinect.kinect_depth_cb, queue_size=1)
    rospy.spin()
