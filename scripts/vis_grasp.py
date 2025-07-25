from pathlib import Path
import pybullet
import argparse
import numpy as np
import sys
import os
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

parser = argparse.ArgumentParser()
parser.add_argument("data_path", type=str, help="Path to the data")
parser.add_argument("--mesh_path", type=str, default=None, help="Path to the mesh")
parser.add_argument("--filter", type=str, default="highest", help="Which grasp to visualize: 'highest', 'lowest', 'all', an integer index, a score threshold (e.g. '0.8'), or a range of indices (e.g. '0:10')")
parser.add_argument("--delay", type=float, default=0.1, help="Delay between grasps in animation")
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
mesh_path = args.mesh_path if args.mesh_path else Path(args.data_path) / "mesh.obj"
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

indices_to_show = []

if ":" in args.filter:
    try:
        start, end = map(int, args.filter.split(':'))
        indices_to_show = list(range(start, end))
        print(f"Showing grasps from index {start} to {end-1}")
    except ValueError:
        print(f"Invalid range for filter: {args.filter}")
        sys.exit(1)
elif args.filter == "all":
    indices_to_show = list(range(len(data["scores"])))
    print("Showing all grasps")
elif args.filter == "highest":
    print("Highest scoring grasp")
    sorted_indices = np.argsort(data["scores"])[::-1]
    indices_to_show = [sorted_indices[0]]
elif args.filter == "lowest":
    print("Lowest scoring grasp")
    sorted_indices = np.argsort(data["scores"])[::-1]
    indices_to_show = [sorted_indices[-1]]
elif '.' in args.filter and args.filter.replace('.','',1).isdigit():
    threshold = float(args.filter)
    print(f"Grasps with score below {threshold}")
    
    passed_indices = np.where(data["scores"] <= threshold)[0]
    
    if len(passed_indices) == 0:
        print(f"No grasps found with score below {threshold}")
        sys.exit(1)
    
    indices_to_show = passed_indices

elif args.filter.isdigit():
    print(f"Grasp with index {args.filter}")
    try:
        indices_to_show = [int(args.filter)]
    except IndexError:
        print(f"Index {args.filter} out of bounds")
        sys.exit(1)
else:
    print(f"Invalid filter option: {args.filter}")
    sys.exit(1)


if not indices_to_show:
    print("No grasps to visualize.")
    sys.exit(0)

def set_grasp(grasp_data):
    # Set hand pose
    pybullet.resetBasePositionAndOrientation(bodyUniqueId=hand_id, posObj=grasp_data[:3], ornObj=grasp_data[3:7])

    # Set joint angles
    for k, j in enumerate([1,2,3, 7,8,9, 13,14,15, 19,20,21]):
        pybullet.resetJointState(hand_id, jointIndex=j, targetValue=grasp_data[7 + k], targetVelocity=0)
        # Set coupled joint
        if j in [3, 9, 15, 21]:
            pybullet.resetJointState(hand_id, jointIndex=j + 1, targetValue=grasp_data[7 + k], targetVelocity=0)

# Simulation loop
try:
    if len(indices_to_show) > 1:
        last_update = time.time()
        current_index = 0
        while pybullet.isConnected():
            pybullet.stepSimulation()
            
            if time.time() - last_update >= args.delay:
                index = indices_to_show[current_index]
                grasp = data["grasps"][index] 
                set_grasp(grasp)
                print(f"Trial {index} of {len(indices_to_show)}, score {data['scores'][index]}")
                
                current_index = (current_index + 1) % len(indices_to_show)
                last_update = time.time()
    else:
        grasp_index = indices_to_show[0]
        grasp = data["grasps"][grasp_index]
        set_grasp(grasp)
        print(f"Score {data['scores'][grasp_index]}")

        while pybullet.isConnected():
            pybullet.stepSimulation()
except KeyboardInterrupt:
    pybullet.disconnect()
    sys.exit(0)
finally:
    if pybullet.isConnected():
        pybullet.disconnect()