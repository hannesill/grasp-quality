import torch
import pytorch_kinematics as pk

class DLRHandFK:
    def __init__(self, urdf_path, device='cpu', dtype=torch.float32):
        self.device = torch.device(device)
        self.dtype = dtype
        with open(urdf_path, 'rb') as f:
            self.chain = pk.build_chain_from_urdf(f.read()).to(device=self.device, dtype=self.dtype)
        
        self.control_point_links = [
            # Finger tips
            'forefinger_distal_link', 'middlefinger_distal_link', 
            'ringfinger_distal_link', 'thumb_distal_link',
            # Middle links
            'forefinger_middle_link', 'middlefinger_middle_link',
            'ringfinger_middle_link', 'thumb_middle_link',
            # Knuckles
            'forefinger_knuckle_link', 'middlefinger_knuckle_link',
            'ringfinger_knuckle_link', 'thumb_knuckle_link',
            # Proximal links
            'forefinger_proximal_link', 'middlefinger_proximal_link',
            'ringfinger_proximal_link', 'thumb_proximal_link',
            # Base links
            'forefinger_base_link', 'middlefinger_base_link',
            'ringfinger_base_link', 'thumb_base_link',
            # Palm
            'hand_base'
        ]

    def forward(self, grasp_config):
        """
        Calculates the world coordinates of the control points for a given grasp configuration.
        Args:
            grasp_config: A (B, 19) or (19,) tensor with hand pose and joint angles.
        Returns:
            A (B, N, 3) tensor of control point coordinates in the world frame, 
            where N is the number of control points.
        """
        single_grasp = len(grasp_config.shape) == 1
        if single_grasp:
            grasp_config = grasp_config.unsqueeze(0)
        
        grasp_config = grasp_config.to(device=self.device, dtype=self.dtype)
        batch_size = grasp_config.shape[0]

        # Convert hand pose to transformation (3x3 rotation matrix + 3D translation)
        pos = grasp_config[:, :3]
        rot = grasp_config[:, 3:7] # xyzw quaternion format
        rot = rot / torch.norm(rot, dim=1, keepdim=True)
        q_wxyz = torch.cat([rot[:, 3].unsqueeze(1), rot[:, :3]], dim=1) # pytorch-kinematics uses wxyz quaternion format
        matrix = pk.transforms.quaternion_to_matrix(q_wxyz)
        base_transform = pk.transforms.Transform3d(rot=matrix, pos=pos, device=self.device, dtype=self.dtype)
        
        # Get joint angles
        joint_angles = grasp_config[:, 7:]
        
        # Map grasp config to joint angles
        joint_map = {
            'forefinger_proximal_joint': joint_angles[:, 6],
            'forefinger_knuckle_joint': joint_angles[:, 7],
            'forefinger_middle_joint': joint_angles[:, 8],
            'forefinger_distal_joint_passive': joint_angles[:, 8], # Coupled
            'middlefinger_proximal_joint': joint_angles[:, 3],
            'middlefinger_knuckle_joint': joint_angles[:, 4],
            'middlefinger_middle_joint': joint_angles[:, 5],
            'middlefinger_distal_joint_passive': joint_angles[:, 5], # Coupled
            'ringfinger_proximal_joint': joint_angles[:, 0],
            'ringfinger_knuckle_joint': joint_angles[:, 1],
            'ringfinger_middle_joint': joint_angles[:, 2],
            'ringfinger_distal_joint_passive': joint_angles[:, 2], # Coupled
            'thumb_proximal_joint': joint_angles[:, 9],
            'thumb_knuckle_joint': joint_angles[:, 10],
            'thumb_middle_joint': joint_angles[:, 11],
            'thumb_distal_joint_passive': joint_angles[:, 11], # Coupled
        }
        
        # Get transformations for all links (in the hand's base frame)
        transforms = self.chain.forward_kinematics(joint_map)

        # Get the world coordinates of the control points (in the hand's base frame)
        world_coords = []
        for link_name in self.control_point_links:
            link_matrix = transforms[link_name].get_matrix() # get transformation matrix for the link (3x3)
            origin = link_matrix[:, :3, 3]
            world_coords.append(origin)
            
        # Stack into a (B, N, 3) tensor. These are in the hand's base frame.
        coords_tensor = torch.stack(world_coords, dim=1)

        # Transform points to world frame
        coords_tensor_world = base_transform.transform_points(coords_tensor)

        if single_grasp:
            return coords_tensor_world.squeeze(0)
            
        return coords_tensor_world 