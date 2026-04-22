import os
import shutil
from pathlib import Path

def reorganize_cq500_data(root_dir,target_dir):
    # Setup paths
    root_path = Path(root_dir)
    target_dir = Path(target_dir)
    ct_target_dir = target_dir / "ct_scans"
    mask_target_dir = target_dir / "masks"

    # Create target directories if they don't exist
    ct_target_dir.mkdir(exist_ok=True)
    mask_target_dir.mkdir(exist_ok=True)

    # Iterate through each folder in the root directory
    # Filters for folders starting with 'CQ500' to avoid moving the target folders themselves
    for folder in root_path.iterdir():
        if folder.is_dir() and folder.name.startswith("CQ500"):
            folder_id = folder.name
            
            # Define source paths
            ct_src = folder / "CT.nii"
            mask_src = folder / "ICH_mask.nii.gz"

            # Define new filenames and destinations
            # Keeping extensions intact (.nii for CT, .nii.gz for mask)
            ct_dst = ct_target_dir / f"{folder_id}.nii"
            mask_dst = mask_target_dir / f"{folder_id}.nii.gz"

            # Move and rename CT scan
            if ct_src.exists():
                shutil.move(str(ct_src), str(ct_dst))
                print(f"Moved CT: {folder_id}")
            else:
                print(f"Warning: CT not found in {folder_id}")

            # Move and rename Mask
            if mask_src.exists():
                shutil.move(str(mask_src), str(mask_dst))
                print(f"Moved Mask: {folder_id}")
            else:
                print(f"Warning: Mask not found in {folder_id}")

    print("\nReorganization complete.")

if __name__ == "__main__":
    # Update this path to your actual root directory
    ROOT_DIRECTORY = "/path/to/volumes" 
    TARGET_DIRECTORY = "/path/to/CQ500-51" # the root output folder named CQ500-51
    reorganize_cq500_data(ROOT_DIRECTORY,TARGET_DIRECTORY)