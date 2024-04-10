import glob
import os
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch.utils import data
from natsort import natsorted

from .basedataset import GradSLAMDataset

class RELLISDataset(GradSLAMDataset):
    def __init__(
        self,
        config_dict,
        basedir,
        sequence,
        stride: Optional[int] = None,
        start: Optional[int] = 0,
        end: Optional[int] = -1,
        desired_height: Optional[int] = 768,
        desired_width: Optional[int] = 1024,
        load_embeddings: Optional[bool] = False,
        embedding_dir: Optional[str] = "embeddings",
        embedding_dim: Optional[int] = 512,
        **kwargs,
    ):
        self.input_folder = os.path.join(basedir, sequence)
        self.pose_path = None
        super().__init__(
            config_dict,
            stride=stride,
            start=start,
            end=end,
            desired_height=desired_height,
            desired_width=desired_width,
            load_embeddings=load_embeddings,
            embedding_dir=embedding_dir,
            embedding_dim=embedding_dim,
            **kwargs,
        )

    def parse_list(self, filepath, skiprows=0):
        """ read list data """
        data = np.loadtxt(filepath, delimiter=' ',
                          dtype=np.unicode_, skiprows=skiprows)
        return data

    def associate_frames(self, tstamp_image, tstamp_depth, tstamp_pose, max_dt=0.08):
        """ pair images, depths, and poses """
        associations = []
        for i, t in enumerate(tstamp_image):
            if tstamp_pose is None:
                j = np.argmin(np.abs(tstamp_depth - t))
                if (np.abs(tstamp_depth[j] - t) < max_dt):
                    associations.append((i, j))

            else:
                j = np.argmin(np.abs(tstamp_depth - t))
                k = np.argmin(np.abs(tstamp_pose - t))

                if (np.abs(tstamp_depth[j] - t) < max_dt) and \
                        (np.abs(tstamp_pose[k] - t) < max_dt):
                    associations.append((i, j, k))

        return associations
    
    def get_filepaths(self):
        color_paths = sorted(glob.glob(os.path.join(self.input_folder, 'images', '*.jpg')))
        depth_paths = sorted(glob.glob(os.path.join(self.input_folder, 'depth', '*.npy')))
        
        embedding_paths = None

        return color_paths, depth_paths, embedding_paths
    
    def load_poses(self):
        df = pd.read_csv(self.pose_path if self.pose_path is not None else os.path.join(self.input_folder, 'poses.txt'), header=None, sep=' ')
        poses = df.values.reshape(-1, 3, 4).astype(np.float32)

        pose_addition = np.full((poses.shape[0], 1, 4), [0, 0, 0, 1])
        poses = np.concatenate([poses, pose_addition], axis=1)
        
        poses = [torch.from_numpy(pose).float() for pose in poses]
        
        return poses
    
    def read_embedding_from_file(self, embedding_file_path):
        embedding = torch.load(embedding_file_path, map_location="cpu")
        return embedding.permute(0, 2, 3, 1)
    
