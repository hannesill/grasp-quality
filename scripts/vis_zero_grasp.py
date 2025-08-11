import pybullet as p
import time
import os
import sys

# Ensure the root directory is in the system path to find the URDF
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

def set_zero_pose(hand_id):
    """
    Resets the hand to a zero configuration.
    - Palm at origin (0,0,0) with no rotation.
    - All movable joint angles set to 0.
    """
    # Set base position and orientation to the world origin
    p.resetBasePositionAndOrientation(hand_id, [0, 0, 0], [0, 0, 0, 1])

    # Get the total number of joints in the URDF
    num_joints = p.getNumJoints(hand_id)
    
    # Iterate through all joints and reset the movable ones to 0
    for i in range(num_joints):
        joint_info = p.getJointInfo(hand_id, i)
        # Check if the joint is a revolute joint (i.e., it can rotate)
        if joint_info[2] == p.JOINT_REVOLUTE:
            p.resetJointState(hand_id, i, targetValue=0, targetVelocity=0)

def main():
    """
    Initializes PyBullet and visualizes the hand in its zero pose.
    """
    p.connect(p.GUI)
    p.setGravity(0, 0, 0) # No gravity needed for static visualization

    # Add coordinate axes at the origin for a clear frame of reference
    # X-axis is Red, Y-axis is Green, Z-axis is Blue
    p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0], 2)
    p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0], 2)
    p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1], 2)

    # Load the DLR hand URDF
    try:
        hand_id = p.loadURDF("urdfs/dlr2.urdf", useFixedBase=True)
    except p.error as e:
        print(f"Error loading URDF: {e}")
        print("Please ensure the path 'urdfs/dlr2.urdf' is correct.")
        p.disconnect()
        return

    # Set the hand to the defined zero pose
    set_zero_pose(hand_id)

    print("\nVisualizing hand in zero configuration.")
    print(" - Palm position: [0, 0, 0]")
    print(" - Palm rotation: [0, 0, 0, 1] (quaternion)")
    print(" - All finger joint angles: 0")
    print("\nPress Ctrl+C in the terminal to exit.")

    try:
        # Loop to keep the simulation window open
        while True:
            p.stepSimulation()
            time.sleep(1./240.)
    except KeyboardInterrupt:
        print("\nExiting visualization.")
    finally:
        if p.isConnected():
            p.disconnect()

if __name__ == "__main__":
    main()

