"""
Minimal example script for converting a dataset collected on the DROID platform to LeRobot format.

Usage:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data

If you want to push your dataset to the Hugging Face Hub, you can use the following command:
uv run examples/droid/convert_droid_data_to_lerobot.py --data_dir /path/to/your/data --push_to_hub

The resulting dataset will get saved to the $LEROBOT_HOME directory.
"""
import os
from collections import defaultdict
import copy
import glob
import json
from pathlib import Path
import shutil

import cv2
import h5py
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
import numpy as np
from PIL import Image
from tqdm import tqdm
import argparse
import mediapy


def resize_image(image, size):
    image = Image.fromarray(image)
    return np.array(image.resize(size, resample=Image.BICUBIC))


def main(input_data_dir: str, push_to_hub: bool = False):
    lerobot_home = os.getenv('HF_LEROBOT_HOME')
    REPO_NAME = "my_droid_stack"  # Name of the output dataset, also used for the Hugging Face Hub
    # Clean up any existing dataset in the output directory
    output_path = os.path.join(lerobot_home, REPO_NAME)
    output_path = Path(output_path)
    if output_path.exists():
        shutil.rmtree(output_path)
    # data_dir = Path(data_dir)

    # Create LeRobot dataset, define features to store
    # We will follow the DROID data naming conventions here.
    # LeRobot assumes that dtype of image data is `image`
    dataset = LeRobotDataset.create(
        repo_id=REPO_NAME,
        robot_type="panda",
        fps=15,  # DROID data is typically recorded at 15fps
        features={
            # We call this "left" since we will only use the left stereo camera (following DROID RLDS convention)
            "exterior_image_1_left": {
                "dtype": "image",
                "shape": (180, 320, 3),  # This is the resolution used in the DROID RLDS dataset
                "names": ["height", "width", "channel"],
            },
            "exterior_image_2_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "wrist_image_left": {
                "dtype": "image",
                "shape": (180, 320, 3),
                "names": ["height", "width", "channel"],
            },
            "joint_position": {
                "dtype": "float32",
                "shape": (7,),
                "names": ["joint_position"],
            },
            "gripper_position": {
                "dtype": "float32",
                "shape": (1,),
                "names": ["gripper_position"],
            },
            "actions": {
                "dtype": "float32",
                "shape": (8,),  # We will use joint *velocity* actions here (7D) + gripper position (1D)
                "names": ["actions"],
            },
        },
        image_writer_threads=10,
        image_writer_processes=5,
    )

    # Load language annotations
    # Note: we load the DROID language annotations for this example, but you can manually define them for your own data
    # with (data_dir / "aggregated-annotations-030724.json").open() as f:
    #     language_annotations = json.load(f)

    # # Loop over raw DROID fine-tuning datasets and write episodes to the LeRobot dataset
    # # We assume the following directory structure:
    # # RAW_DROID_PATH/
    # #   - <...>/
    # #     - recordings/
    # #        - MP4/
    # #          - <camera_id>.mp4  # single-view video of left stereo pair camera
    # #     - trajectory.hdf5
    # #   - <...>/
    # episode_paths = list(data_dir.glob("**/trajectory.h5"))

    data_dirs = [os.path.join(input_data_dir, 'stack_1014'), os.path.join(input_data_dir, 'stack_synthetic'), os.path.join(input_data_dir, 'stack_synthetic2')]
    instru = ['stack blue block on red block', 'stack blue block on red block', 'stack blue block on red block']
    
    episode_paths = []
    instructions = []

    for idx, data_dir in enumerate(data_dirs):
        traj_info_dir = f'{data_dir}/annotation_all_skip1/val'
        traj_info_path = os.listdir(traj_info_dir)
        traj_info_path.sort()
        traj_info_path = [os.path.join(traj_info_dir, i) for i in traj_info_path if i.endswith('.json')]
        episode_paths += traj_info_path
        instructions += instru[idx:idx+1]*len(traj_info_path)


    print(f"Found {len(episode_paths)} episodes for conversion")

    # We will loop over each dataset_name and write episodes to the LeRobot dataset
    for episode_path in tqdm(episode_paths, desc="Converting episodes"):
        video_path = episode_path.replace('annotation_all_skip1', 'videos_ori').replace('.json', '')

        with open(episode_path, 'r') as f:
            info = json.load(f)
            joint_pos = np.array(info['observation.state.joint_position'])
            joint_vel = np.array(info['action.joint_velocity'])
            gripper_pos = np.array(info['action.gripper_position'])[...,np.newaxis]
        
        v1,v2,v3 = mediapy.read_video(f'{video_path}/0.mp4'), mediapy.read_video(f'{video_path}/1.mp4'), mediapy.read_video(f'{video_path}/2.mp4')
        language_instruction = instructions.pop(0)  # Get the corresponding instruction for this episode

        print(f"Converting episode with language instruction: {language_instruction}")

        # Write to LeRobot dataset
        print(joint_pos.shape)
        print(joint_pos[0].shape)
        start_idx = 0
        end_idx = joint_pos.shape[0]
        # for idx in range(len(joint_pos)):
        for idx in range(start_idx, end_idx):
            dataset.add_frame(
                {
                    # Note: need to flip BGR --> RGB for loaded images
                    "exterior_image_1_left": resize_image(v2[idx], (320, 180)), # always use camera in left
                    "exterior_image_2_left": resize_image(v2[idx], (320, 180)),
                    "wrist_image_left": resize_image(v3[idx], (320, 180)),
                    "joint_position": np.asarray(
                        joint_pos[idx], dtype=np.float32
                    ),
                    "gripper_position": np.asarray(
                        gripper_pos[idx], dtype=np.float32
                    ),
                    # Important: we use joint velocity actions here since pi05-droid was pre-trained on joint velocity actions
                    "actions": np.concatenate(
                        [joint_vel[idx], gripper_pos[idx]], dtype=np.float32
                    ),
                    "task": language_instruction,
                }
            )
        
        dataset.save_episode()

    # Optionally push to the Hugging Face Hub
    if push_to_hub:
        dataset.push_to_hub(
            tags=["libero", "panda", "rlds"],
            private=False,
            push_videos=True,
            license="apache-2.0",
        )


##########################################################################################################
################ The rest of this file are functions to parse the raw DROID data #########################
################ You don't need to worry about understanding this part           #########################
################ It was copied from here: https://github.com/JonathanYang0127/r2d2_rlds_dataset_builder/blob/parallel_convert/r2_d2/r2_d2.py
##########################################################################################################


camera_type_dict = {
    "hand_camera_id": 0,
    "varied_camera_1_id": 1,
    "varied_camera_2_id": 1,
}

camera_type_to_string_dict = {
    0: "hand_camera",
    1: "varied_camera",
    2: "fixed_camera",
}


def get_camera_type(cam_id):
    if cam_id not in camera_type_dict:
        return None
    type_int = camera_type_dict[cam_id]
    return camera_type_to_string_dict[type_int]


class MP4Reader:
    def __init__(self, filepath, serial_number):
        # Save Parameters #
        self.serial_number = serial_number
        self._index = 0

        # Open Video Reader #
        self._mp4_reader = cv2.VideoCapture(filepath)
        if not self._mp4_reader.isOpened():
            raise RuntimeError("Corrupted MP4 File")

    def set_reading_parameters(
        self,
        image=True,  # noqa: FBT002
        concatenate_images=False,  # noqa: FBT002
        resolution=(0, 0),
        resize_func=None,
    ):
        # Save Parameters #
        self.image = image
        self.concatenate_images = concatenate_images
        self.resolution = resolution
        self.resize_func = cv2.resize
        self.skip_reading = not image
        if self.skip_reading:
            return

    def get_frame_resolution(self):
        width = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH)
        height = self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT)
        return (width, height)

    def get_frame_count(self):
        if self.skip_reading:
            return 0
        return int(self._mp4_reader.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))

    def set_frame_index(self, index):
        if self.skip_reading:
            return

        if index < self._index:
            self._mp4_reader.set(cv2.CAP_PROP_POS_FRAMES, index - 1)
            self._index = index

        while self._index < index:
            self.read_camera(ignore_data=True)

    def _process_frame(self, frame):
        frame = copy.deepcopy(frame)
        if self.resolution == (0, 0):
            return frame
        return self.resize_func(frame, self.resolution)

    def read_camera(self, ignore_data=False, correct_timestamp=None):  # noqa: FBT002
        # Skip if Read Unnecesary #
        if self.skip_reading:
            return {}

        # Read Camera #
        success, frame = self._mp4_reader.read()

        self._index += 1
        if not success:
            return None
        if ignore_data:
            return None

        # Return Data #
        data_dict = {}

        if self.concatenate_images or "stereo" not in self.serial_number:
            data_dict["image"] = {self.serial_number: self._process_frame(frame)}
        else:
            single_width = frame.shape[1] // 2
            data_dict["image"] = {
                self.serial_number + "_left": self._process_frame(frame[:, :single_width, :]),
                self.serial_number + "_right": self._process_frame(frame[:, single_width:, :]),
            }

        return data_dict

    def disable_camera(self):
        if hasattr(self, "_mp4_reader"):
            self._mp4_reader.release()


class RecordedMultiCameraWrapper:
    def __init__(self, recording_folderpath, camera_kwargs={}):  # noqa: B006
        # Save Camera Info #
        self.camera_kwargs = camera_kwargs

        # Open Camera Readers #
        mp4_filepaths = glob.glob(recording_folderpath + "/*.mp4")  # noqa: PTH207
        all_filepaths = mp4_filepaths

        self.camera_dict = {}
        for f in all_filepaths:
            serial_number = f.split("/")[-1][:-4]
            cam_type = get_camera_type(serial_number)
            camera_kwargs.get(cam_type, {})

            if f.endswith(".mp4"):
                Reader = MP4Reader  # noqa: N806
            else:
                raise ValueError

            self.camera_dict[serial_number] = Reader(f, serial_number)

    def read_cameras(self, index=None, camera_type_dict={}, timestamp_dict={}):  # noqa: B006
        full_obs_dict = defaultdict(dict)

        # Read Cameras In Randomized Order #
        all_cam_ids = list(self.camera_dict.keys())
        # random.shuffle(all_cam_ids)

        for cam_id in all_cam_ids:
            if "stereo" in cam_id:
                continue
            try:
                cam_type = camera_type_dict[cam_id]
            except KeyError:
                print(f"{self.camera_dict} -- {camera_type_dict}")
                raise ValueError(f"Camera type {cam_id} not found in camera_type_dict")  # noqa: B904
            curr_cam_kwargs = self.camera_kwargs.get(cam_type, {})
            self.camera_dict[cam_id].set_reading_parameters(**curr_cam_kwargs)

            timestamp = timestamp_dict.get(cam_id + "_frame_received", None)
            if index is not None:
                self.camera_dict[cam_id].set_frame_index(index)

            data_dict = self.camera_dict[cam_id].read_camera(correct_timestamp=timestamp)

            # Process Returned Data #
            if data_dict is None:
                return None
            for key in data_dict:
                full_obs_dict[key].update(data_dict[key])

        return full_obs_dict


def get_hdf5_length(hdf5_file, keys_to_ignore=[]):  # noqa: B006
    length = None

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            curr_length = get_hdf5_length(curr_data, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            curr_length = len(curr_data)
        else:
            raise ValueError

        if length is None:
            length = curr_length
        assert curr_length == length

    return length


def load_hdf5_to_dict(hdf5_file, index, keys_to_ignore=[]):  # noqa: B006
    data_dict = {}

    for key in hdf5_file:
        if key in keys_to_ignore:
            continue

        curr_data = hdf5_file[key]
        if isinstance(curr_data, h5py.Group):
            data_dict[key] = load_hdf5_to_dict(curr_data, index, keys_to_ignore=keys_to_ignore)
        elif isinstance(curr_data, h5py.Dataset):
            data_dict[key] = curr_data[index]
        else:
            raise ValueError

    return data_dict


class TrajectoryReader:
    def __init__(self, filepath, read_images=True):  # noqa: FBT002
        self._hdf5_file = h5py.File(filepath, "r")
        is_video_folder = "observations/videos" in self._hdf5_file
        self._read_images = read_images and is_video_folder
        self._length = get_hdf5_length(self._hdf5_file)
        self._video_readers = {}
        self._index = 0

    def length(self):
        return self._length

    def read_timestep(self, index=None, keys_to_ignore=[]):  # noqa: B006
        # Make Sure We Read Within Range #
        if index is None:
            index = self._index
        else:
            assert not self._read_images
            self._index = index
        assert index < self._length

        # Load Low Dimensional Data #
        keys_to_ignore = [*keys_to_ignore.copy(), "videos"]
        timestep = load_hdf5_to_dict(self._hdf5_file, self._index, keys_to_ignore=keys_to_ignore)

        # Increment Read Index #
        self._index += 1

        # Return Timestep #
        return timestep

    def close(self):
        self._hdf5_file.close()


def load_trajectory(
    filepath=None,
    read_cameras=True,  # noqa: FBT002
    recording_folderpath=None,
    camera_kwargs={},  # noqa: B006
    remove_skipped_steps=False,  # noqa: FBT002
    num_samples_per_traj=None,
    num_samples_per_traj_coeff=1.5,
):
    read_recording_folderpath = read_cameras and (recording_folderpath is not None)

    traj_reader = TrajectoryReader(filepath)
    if read_recording_folderpath:
        camera_reader = RecordedMultiCameraWrapper(recording_folderpath, camera_kwargs)

    horizon = traj_reader.length()
    timestep_list = []

    # Choose Timesteps To Save #
    if num_samples_per_traj:
        num_to_save = num_samples_per_traj
        if remove_skipped_steps:
            num_to_save = int(num_to_save * num_samples_per_traj_coeff)
        max_size = min(num_to_save, horizon)
        indices_to_save = np.sort(np.random.choice(horizon, size=max_size, replace=False))
    else:
        indices_to_save = np.arange(horizon)

    # Iterate Over Trajectory #
    for i in indices_to_save:
        # Get HDF5 Data #
        timestep = traj_reader.read_timestep(index=i)

        # If Applicable, Get Recorded Data #
        if read_recording_folderpath:
            timestamp_dict = timestep["observation"]["timestamp"]["cameras"]
            camera_type_dict = {
                k: camera_type_to_string_dict[v] for k, v in timestep["observation"]["camera_type"].items()
            }
            camera_obs = camera_reader.read_cameras(
                index=i, camera_type_dict=camera_type_dict, timestamp_dict=timestamp_dict
            )
            camera_failed = camera_obs is None

            # Add Data To Timestep If Successful #
            if camera_failed:
                break
            timestep["observation"].update(camera_obs)

        # Filter Steps #
        step_skipped = not timestep["observation"]["controller_info"].get("movement_enabled", True)
        delete_skipped_step = step_skipped and remove_skipped_steps

        # Save Filtered Timesteps #
        if delete_skipped_step:
            del timestep
        else:
            timestep_list.append(timestep)

    # Remove Extra Transitions #
    timestep_list = np.array(timestep_list)
    if (num_samples_per_traj is not None) and (len(timestep_list) > num_samples_per_traj):
        ind_to_keep = np.random.choice(len(timestep_list), size=num_samples_per_traj, replace=False)
        timestep_list = timestep_list[ind_to_keep]

    # Close Readers #
    traj_reader.close()

    # Return Data #
    return timestep_list


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_data_dir", type=str, help="Base directory containing stack_1014, stack_synthetic, stack_synthetic2")
    parser.add_argument("--push_to_hub", action="store_true", help="Push the resulting dataset to the Hugging Face Hub")
    args = parser.parse_args()
    main(input_data_dir=args.input_data_dir, push_to_hub=args.push_to_hub)
