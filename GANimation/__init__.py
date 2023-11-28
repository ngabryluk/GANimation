import sys
from os import path

sys.path.append(path.abspath(path.dirname(__file__)))

from tf_utils import *
# import tensorflow_datasets as tfds
import pdb

import ssl
import urllib
import requests
import matplotlib.pyplot as plt

requests.packages.urllib3.disable_warnings()

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


__all__ = [
    "GANimation",
    "model",
    "losses",
    "tf_utils",
    "exceptions",
    "generators",
    "discriminators",
]
__spec__ = "GANimation"

if __name__ == "__main__":

    DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'Data\\dataset')

    # Reads in the images from our synthetic dataset into a numpy array
    images = os.listdir(DATA_PATH)
    img_arr = np.empty((len(images), 256, 256))
    for i in range(len(images)):
        # Read the image as a numpy array
        filepath = os.path.join(DATA_PATH, images[i])
        arr = np.array(imageio.imread(filepath))
        
        # Put the image in grayscale to make the shape (256, 256)
        arr = np.dot(arr[..., :3], [0.2989, 0.5870, 0.1140])
        arr = np.round(arr, 0)
        
        img_arr[i] = arr # Add to the numpy array of images
    
    # Preview a gif of the first sample
    # to_gif(img_arr[:50])

    

    # train, test = tfds.load("ucf101", split=["train", "test"], shuffle_files=False)
    # URL = "https://storage.googleapis.com/thumos14_files/UCF101_videos.zip"
    # NUM_CLASSES = 10
    # FILES_PER_CLASS = 50
    # files = list_files_from_zip_url(URL)
    # files = [f for f in files if f.endswith(".avi")]
    # files[:10]
    # files_for_class = get_files_per_class(files)
    # classes = list(files_for_class.keys())
    # print("Num classes:", len(classes))
    # print("Num videos for class[0]:", len(files_for_class[classes[0]]))
    # files_subset = select_subset_of_classes(
    #     files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS
    # )
    # list(files_subset.keys())
    # download_dir = pathlib.Path("./UCF101_subset/")
    # subset_paths = download_ucf_101_subset(
    #     URL,
    #     num_classes=NUM_CLASSES,
    #     splits={"train": 30, "val": 10, "test": 10},
    #     download_dir=download_dir,
    # )
    # video_count_train = len(list(download_dir.glob("train/*/*.avi")))
    # video_count_val = len(list(download_dir.glob("val/*/*.avi")))
    # video_count_test = len(list(download_dir.glob("test/*/*.avi")))
    # video_total = video_count_train + video_count_val + video_count_test
    # print(f"Total videos: {video_total}")
    # video_path = "End_of_a_jam.ogv"
    # sample_video = frames_from_video_file(video_path, n_frames=10)
    # sample_video.shape
    # to_gif(sample_video)
    # # docs-infra: no-execute
    # ucf_sample_video = frames_from_video_file(
    #     next(subset_paths["train"].glob("*/*.avi")), 50
    # )
    # to_gif(ucf_sample_video)
