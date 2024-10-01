import os

def rename_folders_and_contents(directory, prefix):
    folder_names = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]
    folder_names.sort() 

    for i, folder_name in enumerate(folder_names):
        folder_path = os.path.join(directory, folder_name)
        new_folder_name = f"{prefix}_{i}" 
        new_folder_path = os.path.join(directory, new_folder_name)

        if os.path.exists(new_folder_path):
            print(f"Folder '{new_folder_name}' already exists, skipping folder renaming.")
        else:
            if folder_path != new_folder_path:
                os.rename(folder_path, new_folder_path)
                print(f"Renamed folder '{folder_name}' to '{new_folder_name}'")

      
        file_names = os.listdir(new_folder_path)
        for j, file_name in enumerate(file_names):
            file_path = os.path.join(new_folder_path, file_name)
            file_extension = os.path.splitext(file_name)[1]
            new_file_name = f"{prefix}_{i}_{j}{file_extension}" 
            new_file_path = os.path.join(new_folder_path, new_file_name)

          
            if file_path != new_file_path:
                if os.path.exists(new_file_path):
                    os.remove(new_file_path)
                os.rename(file_path, new_file_path)
                print(f"Renamed file '{file_name}' to '{new_file_name}'")


directory_path = input("Enter the path of the directory: ")
prefix = input("Enter the prefix for folder and file names: ")

rename_folders_and_contents(directory_path, prefix)
