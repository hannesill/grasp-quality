import torch
import torch.nn.functional as F
import pybullet as p
from pathlib import Path
import sys
import os
import numpy as np
import argparse

# Add parent directory to path to import project modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.fk import DLRHandFK
from src.dataset import GraspDataset

def main():
    parser = argparse.ArgumentParser(description='Test forward kinematics and visualize grasp.')
    parser.add_argument('--scene_idx', type=int, default=0, help='Index of the scene to load.')
    parser.add_argument('--grasp_idx', type=int, default=0, help='Index of the grasp to visualize.')
    args = parser.parse_args()

    # --- Configuration ---
    device = 'cpu'
    urdf_path = 'urdfs/dlr2.urdf'
    data_path = 'data/processed'
    scene_idx = args.scene_idx
    grasp_idx = args.grasp_idx

    # --- Setup ---
    # PyBullet
    p.connect(p.GUI)
    p.setGravity(0, 0, -9.8)
    
    # Load hand
    hand_id = p.loadURDF(
        urdf_path,
        globalScaling=1,
        basePosition=[0, 0, 0],
        baseOrientation=p.getQuaternionFromEuler([0, 0, 0]),
        useFixedBase=True,
        flags=p.URDF_MAINTAIN_LINK_ORDER
    )

    # Load data
    dataset = GraspDataset(Path(data_path), split='val', preload=False)
    sdf, translation, scale, grasps, scores = dataset._get_scene_data(scene_idx)
    sdf = torch.from_numpy(sdf).float()
    
    # Load object mesh
    scene_name = Path(dataset.data_files[scene_idx]).parts[-2]
    raw_data_path = Path(data_path).parent / 'raw'
    mesh_path = raw_data_path / scene_name / 'mesh.obj'

    if os.path.exists(mesh_path):
        obj_visual = p.createVisualShape(p.GEOM_MESH, fileName=str(mesh_path))
        obj_collision = p.createCollisionShape(p.GEOM_MESH, fileName=str(mesh_path))
        p.createMultiBody(baseMass=0, baseCollisionShapeIndex=obj_collision, baseVisualShapeIndex=obj_visual, basePosition=[0,0,0])
    else:
        print(f"Could not find mesh at {mesh_path}. Object will not be loaded.")

    grasp_config_np = grasps[grasp_idx]
    grasp_config = torch.from_numpy(grasp_config_np).float().to(device)
    grasp_config[:3] = grasp_config[:3] / scale + translation

    # --- Forward Kinematics ---
    fk_model = DLRHandFK(urdf_path, device=device)
    control_points = fk_model.forward(grasp_config)

    # --- Sampling ---
    sampled_sdf_values = F.grid_sample(
        sdf.unsqueeze(0).unsqueeze(0), 
        control_points.view(1, -1, 1, 1, 3),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    sampled_sdf_values = sampled_sdf_values.squeeze()
    fingertip_sdf_values = sampled_sdf_values[:4]
    print("SDF values at fingertips:")
    print(fingertip_sdf_values)

    contact_loss = torch.square(torch.relu(fingertip_sdf_values)).mean()
    collision_loss = torch.square(torch.relu(-sampled_sdf_values)).mean()
    print(f"Contact loss: {contact_loss.item()}, Collision loss: {collision_loss.item()}")

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
    for i, point in enumerate(control_points_np):
        # The first 4 points are the fingertips
        is_fingertip = i < 4
        color = [0, 1, 0, 1] if is_fingertip else [1, 0, 0, 1] # Green for fingertips, red for others
        p.createMultiBody(baseVisualShapeIndex=p.createVisualShape(p.GEOM_SPHERE, radius=0.01, rgbaColor=color), basePosition=point)

    print("\nVisualizing hand pose and control points in PyBullet.")
    print("Green spheres are fingertips, red spheres are other control points.")
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