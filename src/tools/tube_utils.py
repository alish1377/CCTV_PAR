import os
import shutil
import random
import pandas as pd

def select_images_by_ids(xlsx_path, source_dirs, destination_dir, num_ids=130):
    """
    Selects 100 random IDs from the .xlsx file, searches for corresponding images in the tube directories,
    and copies them to a destination directory, renaming them with the selected IDs.

    Parameters:
    - xlsx_path: Path to the .xlsx file containing the IDs column.
    - source_dirs: List of source directories (e.g., ["tubes_c1_v1", "tubes_c2_v2", "tubes_c2_v11"]).
    - destination_dir: Path to the destination directory (e.g., "data/NATIVE/new_original_test_tubes").
    - num_ids: Number of random IDs to select (default is 100).
    """
    # Load the .xlsx file
    df = pd.read_excel(xlsx_path)
    
    # Ensure the destination directory exists
    os.makedirs(destination_dir, exist_ok=True)
    
    # Randomly select 100 IDs from the column
    selected_ids = random.sample(df['ID'].tolist(), min(num_ids, len(df['ID'])))
    
    for selected_id in selected_ids:
        # Parse the selected ID to extract camera, video, and tube numbers
        camera_video, tube_name = selected_id.split('_')[:2], selected_id.split('_')[2]
        
        # Search for the corresponding image in the source directories
        found = False
        for source_dir in source_dirs:
            if camera_video[0] in source_dir and camera_video[1] in source_dir:
                tube_path = os.path.join(source_dir, tube_name)
                if os.path.isdir(tube_path):
                    image_name = f"{tube_name}_{selected_id.split('_')[-1]}.jpg"
                    image_path = os.path.join(tube_path, image_name)
                    if os.path.isfile(image_path):
                        # Copy and rename the image
                        destination_path = os.path.join(destination_dir, f"{selected_id}.jpg")
                        if not os.path.isfile(destination_path):
                            shutil.copy(image_path, destination_path)
                        print(f"Copied and renamed: {image_path} -> {destination_path}")
                        found = True
                        break
        if not found:
            print(f"Image for ID {selected_id} not found in the source directories.")

def create_image_dict(folder_path):
    """
    Creates a dictionary where keys are tube names (tube1, tube2, ...) and values are image names.

    Parameters:
    - folder_path: Path to the folder containing images.

    Returns:
    - image_dict: Dictionary with tube names as keys and image names as values.
    """
    image_dict = {}
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.jpg')]  # Filter image files
    image_files.sort()  # Sort images alphabetically or numerically

    for i, image_file in enumerate(image_files):
        tube_key = f"tube{i+1}"  # Create key like tube1, tube2, ...
        image_dict[tube_key] = image_file

    return image_dict

def copy_related_tubes(image_dict, source_dirs, destination_dir):
    """
    Copies all related tubes of each key in the image_dict to a directory named new_tubes.

    Parameters:
    - image_dict: Dictionary where keys are tube names (tube1, tube2, ...) and values are image IDs.
    - source_dirs: List of source directories (e.g., ["tubes_c1_v1", "tubes_c2_v2", "tubes_c2_v11"]).
    - destination_dir: Path to the destination directory (e.g., "data/NATIVE/new_tubes").
    """
    os.makedirs(destination_dir, exist_ok=True)

    for tube_key, image_id in image_dict.items():
        # Parse the image ID to extract camera, video, and tube numbers
        camera_video, tube_name = image_id.split('_')[:2], image_id.split('_')[2]

        # Search for the corresponding tube in the source directories
        found = False
        for source_dir in source_dirs:
            if camera_video[0] in source_dir and camera_video[1] in source_dir:
                tube_path = os.path.join(source_dir, tube_name)
                if os.path.isdir(tube_path):
                    destination_tube_path = os.path.join(destination_dir, tube_key)
                    shutil.copytree(tube_path, destination_tube_path, dirs_exist_ok=True)
                    print(f"Copied tube: {tube_path} -> {destination_tube_path}")
                    found = True
                    break
        if not found:
            print(f"Tube for image ID {image_id} not found in the source directories.")

def rename_images_in_tubes(tube_dir):
    """
    Renames images in each tube directory so that the first number matches the tube number.

    Parameters:
    - tube_dir: Path to the directory containing tube folders (e.g., "data/NATIVE/new_tubes").
    """
    tube_folders = [folder for folder in os.listdir(tube_dir) if os.path.isdir(os.path.join(tube_dir, folder))]
    tube_folders.sort(key=lambda name: int(name.replace("tube", "")))  # Sort tubes numerically

    for tube_folder in tube_folders:
        tube_number = int(tube_folder.replace("tube", ""))  # Extract tube number
        tube_path = os.path.join(tube_dir, tube_folder)
        image_files = [f for f in os.listdir(tube_path) if f.endswith('.jpg')]

        for image_file in image_files:
            old_path = os.path.join(tube_path, image_file)
            new_name = f"{tube_number}_{image_file.split('_')[1]}"  # Rename to match tube number
            new_path = os.path.join(tube_path, new_name)
            os.rename(old_path, new_path)
            print(f"Renamed: {old_path} -> {new_path}")



if __name__ == "__main__":
    # Example usage:
    
    xlsx_path = "data/NATIVE/Native_dataset.xlsx"
    source_dirs = [
        "data/NATIVE/tubes_c1_v1",
        "data/NATIVE/tubes_c2_v2",
        "data/NATIVE/tubes_c2_v11"
    ]
    """
    # Copy slected random frames in the destination_dir
    destination_dir = "data/NATIVE/new_original_test_tubes"
    select_images_by_ids(xlsx_path, source_dirs, destination_dir)
    """
    folder_path = "data/NATIVE/new_original_test_tubes"
    tube_dict = create_image_dict(folder_path)
    destination_dir = "data/NATIVE/new_tubes"
    copy_related_tubes(tube_dict, source_dirs, destination_dir)

    rename_images_in_tubes(destination_dir)
    



