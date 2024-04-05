import os
import numpy as np
import rospy
import cv2
from PIL import Image
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image as ros_img
import open3d as o3d
import time

bridge = CvBridge()

class Kinect():
    def __init__(self):
        self.cur_depth = None
        self.cur_color = None
        self.cv2_img = None
        self.data_dir = 'doc/example_data'
        self.depth_call_count = 0
        self.depth_call_lim = 5
        self.last_save_time = time.time()
        self.seconds_between_saves = 5.0
        self.save_location = "/home/noah/robochem_data/curr_scene.png"

    def kinect_color_cb(self, msg):
        self.cv2_img = bridge.imgmsg_to_cv2(msg, "rgb8")
        self.cur_color = np.array(self.cv2_img)
        time_diff = time.time() - self.last_save_time
        print(time_diff)
        if  time_diff >= self.seconds_between_saves:
            self.last_save_time = time.time()
            self.process_kinect_data()

    def process_kinect_data(self):
            img = Image.fromarray(self.cur_color, "RGB")
            img.save(self.save_location)
            # color = np.array(self.cur_color, dtype=np.float32) / 255.0
            # workspace_mask = np.array(Image.open(os.path.join(data_dir, 'kumar_converted.png')))
    
            # mask = (workspace_mask & (self.cur_depth > 0))
            # color_masked = color[mask]
            # color_masked = Image.fromarray(color_masked)

if __name__=='__main__':
    kinect = Kinect()
    rospy.init_node('kinect_grasp_prop')
    image_color_topic = "/kinect2/hd/image_color_rect"
    rospy.Subscriber(image_color_topic, ros_img, kinect.kinect_color_cb, queue_size=1)
    rospy.spin()
