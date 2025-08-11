import argparse
import numpy as np
import pybullet as p
import time

def set_grasp(hand_id, grasp_data):
    """
    Sets the robot hand to a specific grasp configuration, using the correct
    joint mapping from the forward kinematics model.
    
    Args:
        hand_id: The PyBullet ID for the hand.
        grasp_data: A numpy array containing the 19 grasp parameters.
    """
    # Set hand base position and orientation
    p.resetBasePositionAndOrientation(bodyUniqueId=hand_id, 
                                      posObj=grasp_data[:3], 
                                      ornObj=grasp_data[3:7])

    joint_angles = grasp_data[7:]

    # This mapping MUST match the one in src/fk.py
    joint_map = {
        'forefinger_proximal_joint': joint_angles[6],
        'forefinger_knuckle_joint': joint_angles[7],
        'forefinger_middle_joint': joint_angles[8],
        'forefinger_distal_joint_passive': joint_angles[8], # Coupled
        'middlefinger_proximal_joint': joint_angles[3],
        'middlefinger_knuckle_joint': joint_angles[4],
        'middlefinger_middle_joint': joint_angles[5],
        'middlefinger_distal_joint_passive': joint_angles[5], # Coupled
        'ringfinger_proximal_joint': joint_angles[0],
        'ringfinger_knuckle_joint': joint_angles[1],
        'ringfinger_middle_joint': joint_angles[2],
        'ringfinger_distal_joint_passive': joint_angles[2], # Coupled
        'thumb_proximal_joint': joint_angles[9],
        'thumb_knuckle_joint': joint_angles[10],
        'thumb_middle_joint': joint_angles[11],
        'thumb_distal_joint_passive': joint_angles[11], # Coupled
    }

    # Map joint names to their PyBullet indices
    joint_name_to_id = {p.getJointInfo(hand_id, i)[1].decode('UTF-8'): i for i in range(p.getNumJoints(hand_id))}

    for name, value in joint_map.items():
        if name in joint_name_to_id:
            p.resetJointState(hand_id, joint_name_to_id[name], targetValue=value, targetVelocity=0)
        else:
            print(f"Warning: Joint '{name}' not found in URDF.")

def visualize_grasp_and_sdf_target(sdf_path, grasp_file_path, grasp_index):
    """
    Visualizes a grasp configuration relative to the SDF target points.
    """
    p.connect(p.GUI)
    p.setGravity(0, 0, 0) # No gravity needed for this visualization

    # Load hand URDF
    hand_id = p.loadURDF("urdfs/dlr2.urdf",
                         useFixedBase=True,
                         flags=p.URDF_MAINTAIN_LINK_ORDER)

    # Load and visualize the SDF target points (where SDF is near zero)
    try:
        data = np.load(sdf_path)
        sdf = data['sdf']
    except Exception as e:
        print(f"Error loading SDF from {sdf_path}: {e}")
        p.disconnect()
        return

    size = sdf.shape[0]
    scale = 0.5  # Assuming SDF is in a cube scaled to 0.5 world units

    min_sdf_val = sdf.min()
    print(f"Minimum SDF value found in file: {min_sdf_val:.4f}")
    threshold = 1e-2  # Visualize points with SDF value close to the minimum

    target_indices = np.where(np.abs(sdf - min_sdf_val) < threshold)
    
    if target_indices[0].size > 0:
        points = np.stack(target_indices, axis=1)
        points_world = (points / (size - 1) * 2 - 1) * scale
        p.addUserDebugPoints(points_world, [[1, 0, 0]] * len(points_world), pointSize=5)
        print(f"Visualized {len(points_world)} target points (where SDF is near 0).")
    else:
        print("No points found with SDF value near 0.")

    # Load and visualize the grasp
    try:
        grasp_data_all = np.load(grasp_file_path)
        grasp = grasp_data_all["grasps"][grasp_index]
        set_grasp(hand_id, grasp)
        
        if "scores" in grasp_data_all:
            score = grasp_data_all["scores"][grasp_index]
            print(f"Showing grasp {grasp_index} with score {score:.4f}")
        else:
            print(f"Showing grasp {grasp_index}")
            
    except Exception as e:
        print(f"Error loading grasp from {grasp_file_path}: {e}")
        p.disconnect()
        return

    print("\nPress Ctrl+C in the terminal to exit.")
    try:
        while True:
            # Keep the simulation running to view
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        pass
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize a grasp and SDF target points.")
    parser.add_argument("sdf_path", type=str, help="Path to the .npz file with SDF data.")
    parser.add_argument("--grasp_file_path", type=str, required=True, help="Path to the recording.npz file with grasp data.")
    parser.add_argument("--grasp_index", type=int, default=0, help="Index of the grasp to visualize from the file.")
    args = parser.parse_args()
    
    visualize_grasp_and_sdf_target(args.sdf_path, args.grasp_file_path, args.grasp_index)
