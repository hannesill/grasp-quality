import numpy as np
import torch
import xml.etree.ElementTree as ET
from pathlib import Path
import transforms3d

class HandKinematics:
    """
    Hand kinematics for DLR-II hand based on URDF file.
    Computes fingertip and palm positions from 19D grasp pose.
    
    Grasp pose format (19D):
    - [0:3]: palm position (x, y, z)
    - [3:7]: palm orientation (quaternion: w, x, y, z)  
    - [7:19]: finger joint angles (3 joints per finger * 4 fingers = 12)
    """
    
    def __init__(self, urdf_path="urdfs/dlr2.urdf"):
        self.urdf_path = Path(urdf_path)
        self.finger_names = ["thumb", "forefinger", "middlefinger", "ringfinger"]
        
        # Parse URDF to get joint and link information
        self.joints = {}
        self.links = {}
        self.finger_chains = {}
        
        self._parse_urdf()
        self._build_finger_chains()
        
        print(f"✅ Hand kinematics initialized with {len(self.finger_names)} fingers")
    
    def _parse_urdf(self):
        """Parse URDF file to extract joint and link information."""
        tree = ET.parse(self.urdf_path)
        root = tree.getroot()
        
        # Parse joints
        for joint in root.findall('joint'):
            name = joint.get('name')
            joint_type = joint.get('type')
            
            parent = joint.find('parent').get('link')
            child = joint.find('child').get('link')
            
            # Get origin (transformation)
            origin = joint.find('origin')
            if origin is not None:
                xyz = [float(x) for x in origin.get('xyz', '0 0 0').split()]
                rpy = [float(x) for x in origin.get('rpy', '0 0 0').split()]
            else:
                xyz = [0, 0, 0]
                rpy = [0, 0, 0]
            
            # Get axis
            axis = joint.find('axis')
            if axis is not None:
                axis_xyz = [float(x) for x in axis.get('xyz', '0 0 1').split()]
            else:
                axis_xyz = [0, 0, 1]
            
            # Get limits
            limit = joint.find('limit')
            if limit is not None:
                lower = float(limit.get('lower', '0'))
                upper = float(limit.get('upper', '0'))
            else:
                lower = upper = 0
            
            self.joints[name] = {
                'type': joint_type,
                'parent': parent,
                'child': child,
                'xyz': xyz,
                'rpy': rpy,
                'axis': axis_xyz,
                'lower': lower,
                'upper': upper
            }
        
        # Parse links
        for link in root.findall('link'):
            name = link.get('name')
            self.links[name] = {'name': name}
    
    def _build_finger_chains(self):
        """Build kinematic chains for each finger."""
        for finger in self.finger_names:
            chain = []
            
            # Each finger has 3 active joints (proximal, knuckle, middle)
            joint_names = [
                f"{finger}_proximal_joint",
                f"{finger}_knuckle_joint", 
                f"{finger}_middle_joint"
            ]
            
            for joint_name in joint_names:
                if joint_name in self.joints:
                    chain.append(self.joints[joint_name])
            
            self.finger_chains[finger] = chain
            
            # Add base transformation (fixed joint from hand_base to finger_base)
            base_joint_name = f"{finger}_base_transformation_fixed"
            if base_joint_name in self.joints:
                self.finger_chains[finger].insert(0, self.joints[base_joint_name])
    
    def _transform_matrix(self, xyz, rpy):
        """Create 4x4 transformation matrix from translation and rotation."""
        # Convert RPY to rotation matrix
        R = transforms3d.euler.euler2mat(rpy[0], rpy[1], rpy[2], 'sxyz')
        
        # Create homogeneous transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = xyz
        
        return T
    
    def compute_fingertip_positions(self, grasp_pose):
        """
        Compute fingertip positions from grasp pose.
        
        Args:
            grasp_pose: (19,) tensor or numpy array
                - [0:3]: palm position 
                - [3:7]: palm orientation (quaternion w,x,y,z)
                - [7:19]: finger joint angles (3 per finger * 4 fingers)
        
        Returns:
            fingertip_positions: (4, 3) numpy array of fingertip positions
        """
        if isinstance(grasp_pose, torch.Tensor):
            grasp_pose = grasp_pose.cpu().numpy()
        
        # Extract palm pose
        palm_pos = grasp_pose[0:3]
        palm_quat = grasp_pose[3:7]  # w, x, y, z
        
        # Convert quaternion to rotation matrix
        palm_R = transforms3d.quaternions.quat2mat(palm_quat)
        
        # Base transformation (hand_base)
        T_base = np.eye(4)
        T_base[:3, :3] = palm_R
        T_base[:3, 3] = palm_pos
        
        # Extract finger joint angles (3 per finger * 4 fingers = 12)
        finger_angles = grasp_pose[7:19].reshape(4, 3)
        
        fingertip_positions = []
        
        for i, finger in enumerate(self.finger_names):
            # Start with base transformation
            T_current = T_base.copy()
            
            # Apply finger chain transformations
            chain = self.finger_chains[finger]
            angles = finger_angles[i]
            
            for j, joint in enumerate(chain):
                # Static transformation from joint origin
                T_joint = self._transform_matrix(joint['xyz'], joint['rpy'])
                T_current = T_current @ T_joint
                
                # Apply joint angle if it's a revolute joint
                if joint['type'] == 'revolute' and j > 0:  # Skip base transformation
                    angle = angles[j-1]  # j-1 because first joint is base transform
                    axis = joint['axis']
                    
                    # Rotation about joint axis
                    if axis == [0, 0, 1]:  # Z-axis rotation
                        R_joint = transforms3d.euler.euler2mat(0, 0, angle)
                    elif axis == [0, 1, 0]:  # Y-axis rotation  
                        R_joint = transforms3d.euler.euler2mat(0, angle, 0)
                    elif axis == [1, 0, 0]:  # X-axis rotation
                        R_joint = transforms3d.euler.euler2mat(angle, 0, 0)
                    else:
                        # General axis rotation
                        R_joint = transforms3d.axangles.axangle2mat(axis, angle)
                    
                    T_rot = np.eye(4)
                    T_rot[:3, :3] = R_joint
                    T_current = T_current @ T_rot
            
            # Add fingertip offset (approximate)
            fingertip_offset = np.array([0.025, 0, 0])  # 2.5cm forward
            T_fingertip = np.eye(4)
            T_fingertip[:3, 3] = fingertip_offset
            T_current = T_current @ T_fingertip
            
            # Extract fingertip position
            fingertip_pos = T_current[:3, 3]
            fingertip_positions.append(fingertip_pos)
        
        return np.array(fingertip_positions)
    
    def compute_palm_position(self, grasp_pose):
        """
        Compute palm position from grasp pose.
        
        Args:
            grasp_pose: (19,) tensor or numpy array
        
        Returns:
            palm_position: (3,) numpy array of palm position
        """
        if isinstance(grasp_pose, torch.Tensor):
            grasp_pose = grasp_pose.cpu().numpy()
        
        return grasp_pose[0:3]
    
    def compute_distance_fields(self, grasp_pose, grid_size=48, grid_bounds=(-0.15, 0.15)):
        """
        Compute distance fields for palm and fingertips on a 3D grid.
        
        Args:
            grasp_pose: (19,) tensor or numpy array
            grid_size: int, size of the 3D grid
            grid_bounds: tuple, (min, max) bounds of the grid in meters
        
        Returns:
            palm_distances: (48, 48, 48) numpy array of distances to palm
            fingertip_distances: (48, 48, 48) numpy array of distances to closest fingertip
        """
        # Get palm and fingertip positions
        palm_pos = self.compute_palm_position(grasp_pose)
        fingertip_pos = self.compute_fingertip_positions(grasp_pose)
        
        # Create 3D coordinate grid
        grid_coords = np.linspace(grid_bounds[0], grid_bounds[1], grid_size)
        x, y, z = np.meshgrid(grid_coords, grid_coords, grid_coords)
        grid_points = np.stack([x, y, z], axis=-1)  # (48, 48, 48, 3)
        
        # Compute distance to palm
        palm_distances = np.linalg.norm(grid_points - palm_pos, axis=-1)
        
        # Compute distance to closest fingertip
        fingertip_distances = np.inf * np.ones((grid_size, grid_size, grid_size))
        for fingertip in fingertip_pos:
            distances = np.linalg.norm(grid_points - fingertip, axis=-1)
            fingertip_distances = np.minimum(fingertip_distances, distances)
        
        return palm_distances, fingertip_distances

# Test function
def test_hand_kinematics():
    """Test the hand kinematics implementation."""
    hand = HandKinematics()
    
    # Test with a sample grasp pose
    grasp_pose = np.zeros(19)
    grasp_pose[0:3] = [0, 0, 0]  # Palm at origin
    grasp_pose[3:7] = [1, 0, 0, 0]  # Identity quaternion
    grasp_pose[7:19] = np.random.uniform(-0.5, 0.5, 12)  # Random joint angles
    
    # Compute positions
    palm_pos = hand.compute_palm_position(grasp_pose)
    fingertip_pos = hand.compute_fingertip_positions(grasp_pose)
    
    print(f"Palm position: {palm_pos}")
    print(f"Fingertip positions:\n{fingertip_pos}")
    
    # Test distance fields
    palm_distances, fingertip_distances = hand.compute_distance_fields(grasp_pose)
    
    print(f"Palm distance field shape: {palm_distances.shape}")
    print(f"Fingertip distance field shape: {fingertip_distances.shape}")
    print(f"Palm distance range: [{palm_distances.min():.3f}, {palm_distances.max():.3f}]")
    print(f"Fingertip distance range: [{fingertip_distances.min():.3f}, {fingertip_distances.max():.3f}]")

if __name__ == "__main__":
    test_hand_kinematics() 