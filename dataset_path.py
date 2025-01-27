import os
import requests
import tarfile

class DatasetPaths:
    def __init__(self, root_dir="imagenette2-320"):
        """
        Initialize and prepare the dataset paths.
        
        Args:
            root_dir (str): Directory to store the dataset.
        """
        self.dataset_url = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
        self.root_dir = root_dir
        self.train_path = os.path.join(root_dir, "train")
        self.val_path = os.path.join(root_dir, "val")

        # Prepare dataset
        self._download_and_extract_dataset()

    def _download_and_extract_dataset(self):
        """
        Download and extract the Imagenette dataset if not already available.
        """
        if not os.path.exists(self.root_dir):
            os.makedirs(self.root_dir, exist_ok=True)

        # Download the dataset
        archive_path = os.path.join(self.root_dir, "imagenette2-320.tgz")
        if not os.path.exists(archive_path):
            print("Downloading Imagenette dataset...")
            response = requests.get(self.dataset_url, stream=True)
            with open(archive_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=1024):
                    f.write(chunk)
            print("Download complete.")

        # Extract the dataset
        if not os.path.exists(self.train_path) or not os.path.exists(self.val_path):
            print("Extracting Imagenette dataset...")
            with tarfile.open(archive_path, "r:gz") as tar:
                tar.extractall(path=self.root_dir)
            print("Extraction complete.")
