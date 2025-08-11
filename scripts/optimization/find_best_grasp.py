import numpy as np
import argparse

def find_best_grasp(file_path):
    """
    Loads a recording.npz file and finds the grasp with the best (lowest) score.
    """
    try:
        data = np.load(file_path)
    except FileNotFoundError:
        print(f"Error: The file '{file_path}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while loading the file: {e}")
        return

    if "scores" not in data or "grasps" not in data:
        print("Error: The .npz file must contain 'scores' and 'grasps' arrays.")
        return

    scores = data["scores"]
    if len(scores) == 0:
        print("The 'scores' array is empty. No grasps to evaluate.")
        return

    # Find the index of the minimum score
    best_grasp_index = np.argmin(scores)
    best_score = scores[best_grasp_index]

    print(f"Found {len(scores)} total grasps in the file.")
    print(f"The best grasp is at index: {best_grasp_index}")
    print(f"It has the lowest score (FK loss) of: {best_score:.6f}")
    print("\nTo visualize this grasp, use the following command:")
    print(f"python scripts/vis_sdf_target.py data/sphere.npz --grasp_file_path {file_path} --grasp_index {best_grasp_index}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find the best grasp in a recording.npz file.")
    parser.add_argument("file_path", type=str, help="Path to the recording.npz file.")
    args = parser.parse_args()
    
    find_best_grasp(args.file_path)

