import torch
import pybullet as p
from pathlib import Path
import sys
import os
import numpy as np

# Add parent directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fk import DLRHandFK
from src.dataset import GraspDataset

def main():
    # --- Configuration ---
    device = 'cpu'
    urdf_path = 'urdfs/dlr2.urdf'
    data_path = 'data/processed'
    scene_idx = 0
    grasp_idx = 0 

    # --- Setup ---
    # PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    
    # Load hand
    hand_id = p.loadURDF(
        urdf_path,
        useFixedBase=True,
        flags=p.URDF_MAINTAIN_LINK_ORDER
    )

    # Load data
    dataset = GraspDataset(Path(data_path), split='val', preload=False)
    sdf, translation, scale, grasps, scores = dataset._get_scene_data(scene_idx)
    grasp_config_np = grasps[grasp_idx]
    grasp_config = torch.from_numpy(grasp_config_np).float().to(device)
    grasp_config[:3] = grasp_config[:3] / scale + translation

    # --- Forward Kinematics ---
    fk_model = DLRHandFK(urdf_path, device=device)
    control_points = fk_model.forward(grasp_config)
    
    print("Computed control points (world frame):")
    print(control_points)

    # --- Visualization ---
    # Set hand pose in PyBullet to match the grasp config
    hand_pos = grasp_config[:3].numpy()
    hand_orn = grasp_config[3:7].numpy()
    joint_angles = grasp_config[7:].numpy()
    
    p.resetBasePositionAndOrientation(hand_id, hand_pos, hand_orn)

    revolute_joints = []
    for i in range(p.getNumJoints(hand_id)):
        info = p.getJointInfo(hand_id, i)
        if info[2] == p.JOINT_REVOLUTE:
            revolute_joints.append(i)
    
    for i, joint_idx in enumerate(revolute_joints):
        if i < len(joint_angles):
            p.resetJointState(hand_id, joint_idx, joint_angles[i])

    # Draw spheres at the control points
    control_points_np = control_points.detach().cpu().numpy()
    for point in control_points_np:
        p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=[1, 0, 0, 1]), basePosition=point)

    print("\nVisualizing hand pose and control points in PyBullet.")
    print("Red spheres indicate the calculated control points.")
    print("Press Ctrl+C in the terminal to exit.")

    try:
        while p.isConnected():
            p.stepSimulation()
    except KeyboardInterrupt:
        p.disconnect()
    finally:
        if p.isConnected():
            p.disconnect()


if __name__ == "__main__":
    main() 