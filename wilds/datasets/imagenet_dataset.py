import csv
import os
from typing import Any, Dict, Optional, Tuple, Union

import torch
import pandas as pd
from PIL import Image
import numpy as np

from wilds.common.utils import map_to_id_array
from wilds.common.metrics.all_metrics import Accuracy
from wilds.common.grouper import CombinatorialGrouper
from wilds.datasets.wilds_dataset import WILDSDataset


class ImagenetDataset(WILDSDataset):
    """
    DomainNet dataset.
    586,576 images in 345 categories (airplane, ball, cup, etc.) across 6 domains (clipart, infograph, painting,
    quickdraw, real and sketch).

    Supported `split_scheme`:
        'official': use the official split from DomainNet

    Input (x):
        224 x 224 x 3 RGB image.

    Label (y):
        y is one of the 345 categories in DomainNet.

    Metadata:
        domain associated with each image

    Website:
        http://ai.bu.edu/M3SDA

    Original publication:

        @inproceedings{peng2019moment,
          title={Moment matching for multi-source domain adaptation},
          author={Peng, Xingchao and Bai, Qinxun and Xia, Xide and Huang, Zijun and Saenko, Kate and Wang, Bo},
          booktitle={Proceedings of the IEEE International Conference on Computer Vision},
          pages={1406--1415},
          year={2019}
        }

    SENTRY publication:

        @article{prabhu2020sentry
           author = {Prabhu, Viraj and Khare, Shivam and Kartik, Deeksha and Hoffman, Judy},
           title = {SENTRY: Selective Entropy Optimization via Committee Consistency for Unsupervised Domain Adaptation},
           year = {2020},
           journal = {arXiv preprint: 2012.11460},
        }

    Fair Use Notice:
        "This dataset contains some copyrighted material whose use has not been specifically authorized by the copyright owners.
        In an effort to advance scientific research, we make this material available for academic research. We believe this
        constitutes a fair use of any such copyrighted material as provided for in section 107 of the US Copyright Law.
        In accordance with Title 17 U.S.C. Section 107, the material on this site is distributed without profit for
        non-commercial research and educational purposes. For more information on fair use please click here. If you wish
        to use copyrighted material on this site or in our dataset for purposes of your own that go beyond non-commercial
        research and academic purposes, you must obtain permission directly from the copyright owner."
    """

    _dataset_name: str = "imagenet"

    def __init__(
        self,
        root_dir: str = "data",
        split_scheme: str = "official",
        download: bool = False, 
        type: str = 'imagenet'
    ):
        # Dataset information
        self._split_scheme: str = split_scheme
        # self._original_resolution = (224, 224)
        self._y_type: str = "long"
        self._y_size: int = 1
        # Path of the dataset
        self._meta_dir = root_dir
        self._data_dir: str = root_dir[0:root_dir.rindex("/")]

        self._n_classes = 1000

        metadata_df: pd.DataFrame = pd.read_pickle(self._meta_dir)
        
        if type == 'yfcc_imagenet':
            metadata_df.rename(columns = {'imagenet_classes' : 'y'}, inplace = True)
            self._input_image_paths = metadata_df['filepath'].values
        elif type == 'imagenet':
            metadata_df.rename(columns = {'label' : 'y'}, inplace = True)
            self._input_image_paths = None
        elif type == 'combined':
            metadata_df.rename(columns = {'imagenet_classes' : 'y'}, inplace = True)
            self._input_image_paths = metadata_df['filepath'].values
        
        self._y_array = torch.from_numpy(metadata_df['y'].values).type(torch.LongTensor)
    
        # Initialize dummy split_array with all zeros
        self.initialize_split_dicts()
        self.initialize_split_array(metadata_df)

        # Populate metadata fields
        if type == "combined":
            # self._metadata_fields = ["dataset", "original_idx", "y"]
            # metadata_df = metadata_df[self._metadata_fields]
            # possible_metadata_values = {
            #     "dataset":["imagenet", "yfcc_imagenet"],
            #     "original_idx" : range(len(metadata_df)),
            #     "y": range(self._n_classes), 
            # }
            self._metadata_fields = ["dataset", "y"]
            metadata_df = metadata_df[self._metadata_fields]
            possible_metadata_values = {
                "dataset":["imagenet", "yfcc_imagenet"],
                "y": range(self._n_classes), 
            }
            self._metadata_map, metadata = map_to_id_array(
                metadata_df, possible_metadata_values
            )
            self._metadata_array = torch.from_numpy(metadata.astype("long"))
        else:
            self._metadata_fields = ["y"]
            metadata_df = metadata_df[self._metadata_fields]
            possible_metadata_values = {
                "y": range(self._n_classes)
            }
            self._metadata_map, metadata = map_to_id_array(
                metadata_df, possible_metadata_values
            )
            self._metadata_array = torch.from_numpy(metadata.astype("long"))

        # Eval
        # self.initialize_eval_grouper()
        super().__init__(root_dir, download, self._split_scheme)

    def get_input(self, idx) -> str:
        if self._input_image_paths is None:
            return None
        else:
            # img_path = os.path.join(self.data_dir, self._input_image_paths[idx])
            img_path = self._input_image_paths[idx]
            img = Image.open(img_path).convert("RGB")
            return img

    def eval(
        self,
        y_pred: torch.Tensor,
        y_true: torch.LongTensor,
        metadata: torch.Tensor,
        prediction_fn=None,
    ) -> Tuple[Dict[str, Any], str]:
        """
        Computes all evaluation metrics.
        Args:
            - y_pred (Tensor): Predictions from a model. By default, they are predicted labels (LongTensor).
                               But they can also be other model outputs such that prediction_fn(y_pred)
                               are predicted labels.
            - y_true (LongTensor): Ground-truth labels
            - metadata (Tensor): Metadata
            - prediction_fn (function): A function that turns y_pred into predicted labels
        Output:
            - results (dictionary): Dictionary of evaluation metrics
            - results_str (str): String summarizing the evaluation metrics
        """
        metric: Accuracy = Accuracy(prediction_fn=prediction_fn)
        return self.standard_group_eval(
            metric, self._eval_grouper, y_pred, y_true, metadata
        )


    def initialize_split_dicts(self):
        if self.split_scheme == "official":
            self._split_dict: Dict[str, int] = {
                "train": 0,
                "val": 1,
            }
            self._split_names: Dict[str, str] = {
                "train": "Train",
                "val": "Test",
            }
        else:
            raise ValueError(f"Split scheme {self.split_scheme} is not recognized.")

    def initialize_split_array(self, metadata_df):
        def get_split(row):
            return 0
        self._split_array = metadata_df.apply(
            lambda row: get_split(row), axis=1
        ).to_numpy()

        # np.random.seed(1230)
        # size = metadata_df.shape[0]
        # val_idx = np.random.choice(range(size), round(0.2*size), replace=False)
        # self._split_array = np.zeros(size)
        # self._split_array[val_idx] = 1

    def initialize_eval_grouper(self):
        if self.split_scheme == "official":
            self._eval_grouper = CombinatorialGrouper(
                dataset=self, groupby_fields=["y"]
            )
        else:
            raise ValueError(f"Split scheme {self.split_scheme} not recognized.")
