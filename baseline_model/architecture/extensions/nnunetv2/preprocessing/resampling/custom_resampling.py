from collections import OrderedDict
from typing import Union, Tuple, List

import numpy as np
import pandas as pd
import torch
from batchgenerators.augmentations.utils import resize_segmentation
from scipy.ndimage.interpolation import map_coordinates
from skimage.transform import resize
from nnunetv2.configuration import ANISO_THRESHOLD

def no_resampling_data_or_seg_to_shape(data: Union[torch.Tensor, np.ndarray],
                                       new_shape: Union[Tuple[int, ...], List[int], np.ndarray],
                                       current_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                       new_spacing: Union[Tuple[float, ...], List[float], np.ndarray],
                                       is_seg: bool = False,
                                       order: int = 3, order_z: int = 0,
                                       force_separate_z: Union[bool, None] = False,
                                       separate_z_anisotropy_threshold: float = ANISO_THRESHOLD):
    """
    Hacky resampling function which actually doesn't perform any resampling.
    Way easier to implement no resampling training/inference this way, trust me.
    """
    return data
