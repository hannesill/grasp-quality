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
    dataset = GraspDataset(Path(data_path), split='train', preload=False)

    print(f"Using scene {scene_idx}, stored at {dataset.data_files[scene_idx]}")
    sdf, translation, scale, grasps, scores = dataset._get_scene_data(scene_idx)
    sdf = torch.from_numpy(sdf).float()
    translation = torch.from_numpy(translation).float()
    scale = torch.tensor(scale).float()
    
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

    # Sort grasps by quality score in descending order
    scores = torch.from_numpy(scores).float()
    sorted_indices = torch.argsort(scores, descending=True)
    grasps = grasps[sorted_indices]
    scores = scores[sorted_indices]
    
    print(f"Using grasp {grasp_idx} with quality score {scores[grasp_idx]:.4f}")
    grasp_config_np = grasps[grasp_idx]
    grasp_config = torch.from_numpy(grasp_config_np).float().to(device)
    grasp_config[:3] = grasp_config[:3] / scale + translation

    # --- Forward Kinematics ---
    fk_model = DLRHandFK(urdf_path, device=device)
    control_points = fk_model.forward(grasp_config)
    normalized_control_points = (control_points - translation) / scale

    # --- Sampling ---
    sampled_sdf_values = F.grid_sample(
        sdf.unsqueeze(0).unsqueeze(0), 
        normalized_control_points.view(1, -1, 1, 1, 3),
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    sampled_sdf_values = sampled_sdf_values.squeeze()
    fingertip_sdf_values = sampled_sdf_values[:4]
    print("SDF values at fingertips:")
    print(fingertip_sdf_values)

    contact_loss = torch.square(torch.relu(100 * fingertip_sdf_values)).mean()
    collision_loss = torch.square(torch.relu(-100 * sampled_sdf_values)).mean()
    print(f"Contact loss: {contact_loss.item()}, Collision loss: {collision_loss.item()}")

    # --- Visualization ---
    # Set hand pose in PyBullet to match the grasp config
    hand_pos = grasp_config[:3].numpy()
    hand_orn = grasp_config[3:7].numpy()
    joint_angles = grasp_config[7:].numpy()
    
    p.resetBasePositionAndOrientation(bodyUniqueId=hand_id, posObj=hand_pos, ornObj=hand_orn)

    # Set joint angles
    for k, j in enumerate([1,2,3, 7,8,9, 13,14,15, 19,20,21]):
        p.resetJointState(hand_id, jointIndex=j, targetValue=joint_angles[k], targetVelocity=0)
        # Set coupled joint
        if j in [3, 9, 15, 21]:
            p.resetJointState(hand_id, jointIndex=j + 1, targetValue=joint_angles[k], targetVelocity=0)

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