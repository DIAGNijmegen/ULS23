import numpy as np
from copy import deepcopy
from monai.data import NibabelReader, MetaTensor
from monai.transforms import (
    Orientationd,
    EnsureChannelFirstd,
    Compose,
    SaveImaged,
    LoadImaged,
    Lambdad,
    BorderPadd,
    RandCropByPosNegLabeld,
    KeepLargestConnectedComponentd,
    NormalizeIntensityd,
)
from scipy import ndimage
from skimage.measure import regionprops
from .folder_layout import FolderLayoutULS


class FullyAnnotatedLesionExtractor(object):
    def __init__(
        self,
        output_path,
        depth=32,
        height_width=64,
        resample=False,
        figs=False,
        min_lesion_size_pixel=5,
        exclude_using_short_axis_size=False,
        is_instance_labeled=False,
        lesion_label=1,
        num_samples=1,
        one_lesion_per_scan=False,
    ):
        self.output_path = output_path
        self.depth = depth
        self.height_width = height_width
        self.resample = resample
        self.figs = figs
        self.min_lesion_size_pixel = min_lesion_size_pixel
        self.exclude_using_short_axis_size = exclude_using_short_axis_size
        self.is_instance_labeled = is_instance_labeled
        self.lesion_label = lesion_label
        self.num_samples = num_samples

        self.lesions_to_process = []
        self.lesion_nr = 0
        self.one_lesion_per_scan = one_lesion_per_scan

    def process_dataset(self, data):
        for case in data:
            self.amin = 0
            case_id = case["img"].split("/")[-1].split(".")[0]
            print("Case:", case["img"])

            initial_operations = [
                LoadImaged(keys=["img", "seg"], image_only=False),
                EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
                # NormalizeIntensityd(keys="img"), # Uncomment if you want to normalize based on the full volumes
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                Lambdad(keys=["seg"], func=self.instance_segmentation_converter),
            ]
            if self.one_lesion_per_scan:
                initial_operations.append(
                    KeepLargestConnectedComponentd(keys=["seg"], independent=False)
                )

            load_data = Compose(initial_operations)
            case_data = load_data(case)

            pad_data = Compose(
                [
                    BorderPadd(
                        keys=["img"],
                        spatial_border=(
                            self.height_width // 2,
                            self.height_width // 2,
                            self.depth // 2,
                        ),
                        mode="constant",
                        constant_values=np.amin(case_data["img"]), # -1 so we can re-identify padded area later
                        value=np.amin(case_data["img"]),
                    ),
                    BorderPadd(
                        keys=["seg"],
                        spatial_border=(
                            self.height_width // 2,
                            self.height_width // 2,
                            self.depth // 2,
                        ),
                        mode="constant",
                        constant_values=0,
                        value=0,
                    ),
                ]
            )
            case_data = pad_data(case_data)

            for lesion_nr in self.lesions_to_process:
                print("Selecting lesion number", lesion_nr)
                self.lesion_nr = lesion_nr

                img_out_path = FolderLayoutULS(
                    case_id=case_id,
                    output_dir=self.output_path + "/imagesTr",
                    postfix=f"_lesion_{self.lesion_nr:02d}_sample_",
                    extension=".nii.gz",
                )
                seg_out_path = FolderLayoutULS(
                    case_id=case_id,
                    output_dir=self.output_path + "/labelsTr",
                    postfix=f"_lesion_{self.lesion_nr:02d}_sample_",
                    extension=".nii.gz",
                )
                crop_lesion_volume = Compose(
                    [
                        Lambdad(keys=["seg"], func=self.lesion_selector),
                        RandCropByPosNegLabeld(
                            keys=["img", "seg"],
                            spatial_size=(
                                self.height_width,
                                self.height_width,
                                self.depth,
                            ),
                            label_key="seg",
                            neg=0,
                            num_samples=self.num_samples,
                        ),
                        SaveImaged(
                            keys=["img"],
                            resample=False,
                            squeeze_end_dims=False,
                            folder_layout=img_out_path,
                        ),
                        SaveImaged(
                            keys=["seg"],
                            resample=False,
                            squeeze_end_dims=False,
                            folder_layout=seg_out_path,
                        ),
                    ]
                )
                result = crop_lesion_volume(deepcopy(case_data))
                # print(result)

    def instance_segmentation_converter(self, lbl_data):
        meta = lbl_data.meta
        affine = lbl_data.affine
        # We use monai.transforms.RandCropByPosNegLabeld to crop the lesion VOI's
        # since most datasets do semantic and not instance segmentation we first relabel
        if not self.is_instance_labeled:
            lbl_data[lbl_data != self.lesion_label] = 0  # Only keep lesion label
            lbl_data, num_features = ndimage.label(lbl_data)
        else:
            num_features = int(np.amax(lbl_data))

        print("Number of lesions found:", num_features)

        # Remove lesion that are too small
        self.lesions_to_process = []
        for lesion_nr in range(1, num_features + 1):
            data = np.zeros(lbl_data.shape)
            data[lbl_data == lesion_nr] = 1
            max_size = 0
            for z in range(data.shape[0]):
                regions = regionprops(data[z].astype(int))
                for props in regions:
                    try:
                        if self.exclude_using_short_axis_size:
                            if props.minor_axis_length > max_size:
                                max_size = props.minor_axis_length
                        else:
                            if props.major_axis_length > max_size:
                                max_size = props.major_axis_length
                    except ValueError:
                        print("Couldn't fit regionprops")
                        max_size = 0
            if max_size < self.min_lesion_size_pixel:
                print(
                    f"Lesion {lesion_nr} too small: {max_size} < {self.min_lesion_size_pixel}"
                )
                # Set to background
                lbl_data[lbl_data == lesion_nr] = 0
            else:
                self.lesions_to_process.append(lesion_nr)

        return MetaTensor(lbl_data, meta=meta)

    def lesion_selector(self, lbl_data):
        # Here we set all other instance masks to background leaving
        # only the lesion we are interested in for RandCropByPosNegLabeld
        meta = lbl_data.meta
        affine = lbl_data.affine
        lbl_data[lbl_data != self.lesion_nr] = 0
        lbl_data[lbl_data == self.lesion_nr] = 1
        return MetaTensor(lbl_data, meta=meta)
