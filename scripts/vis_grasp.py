# Usage example: python3 vis_grasp.py student_grasps_v1/02808440/148ec8030e52671d44221bef0fa3c36b/0/
from pathlib import Path
import pybullet
import argparse
import numpy as np
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="Path to the data")
parser.add_argument("--mesh_path", type=str, default="data/raw", help="Path to the mesh")
parser.add_argument("--filter", type=str, default="highest", help="Which grasp to visualize: 'highest', 'lowest', or an integer index")
args = parser.parse_args()

pybullet.connect(pybullet.GUI)

# Load hand
hand_id = pybullet.loadURDF(
    "urdfs/dlr2.urdf",
    globalScaling=1,
    basePosition=[0, 0, 0],
    baseOrientation=pybullet.getQuaternionFromEuler([0, 0, 0]),
    useFixedBase=True,
    flags=pybullet.URDF_MAINTAIN_LINK_ORDER,
)

# Load object
scene_id = Path(args.data_path).parts[-1]
mesh_path = Path(args.mesh_path) / scene_id / "mesh.obj"
visualShapeId = pybullet.createVisualShape(
                shapeType=pybullet.GEOM_MESH,
                fileName=str(mesh_path),
                rgbaColor=[1,1,1,1],
                specularColor=[0.4, .4, 0],
                visualFramePosition=[0, 0, 0],
                meshScale=1)
object_id = pybullet.createMultiBody(
            baseMass=1,
            baseInertialFramePosition=[0, 0, 0],
            baseVisualShapeIndex=visualShapeId,
            baseCollisionShapeIndex=visualShapeId,
            basePosition=[0,0,0],
            baseOrientation=[0,0,0,1])
                        
# Load grasps
data = np.load(Path(args.data_path) / "recording.npz")

# Sort by grasp score
sorted_indx = np.argsort(data["scores"])[::-1]

print(args.filter.replace('.','',1).isdigit())
if args.filter == "highest":
    print("Highest scoring grasp")
    sorted_indx = sorted_indx[0]
elif args.filter == "lowest":
    print("Lowest scoring grasp")
    sorted_indx = sorted_indx[-1]
elif '.' in args.filter:
    print(f"Grasp with score below {args.filter}")
    threshold = float(args.filter)
    filtered_indices = sorted_indx[data["scores"][sorted_indx] <= threshold]
    if len(filtered_indices) == 0:
        print(f"No grasps found with score below {threshold}")
        sys.exit(1)
    sorted_indx = filtered_indices[0]
else:
    print(f"Grasp with index {args.filter}")
    sorted_indx = int(args.filter)

grasp = data["grasps"][sorted_indx]

# Set hand pose
pybullet.resetBasePositionAndOrientation(bodyUniqueId=hand_id, posObj=grasp[:3], ornObj=grasp[3:7])

# Set joint angles
for k, j in enumerate([1,2,3, 7,8,9, 13,14,15, 19,20,21]):
    pybullet.resetJointState(hand_id, jointIndex=j, targetValue=grasp[7 + k], targetVelocity=0)
    # Set coupled joint
    if j in [3, 9, 15, 21]:
        pybullet.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp[7 + k], targetVelocity=0)

print(f"Score {data['scores'][sorted_indx]}")

while pybullet.isConnected():
    pybullet.stepSimulation()
