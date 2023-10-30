import os
import monai
from monai.transforms import (
    Compose,
    SaveImaged,
    LoadImaged,
    RandCropByPosNegLabeld,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    CenterSpatialCropd,
)
from folder_layout import FolderLayoutULS

full_data_path = ".../Dataset400_ULS_twd_model"
half_output_path = ".../Dataset401_ULS_twd_model_half_res"
core_output_path = ".../Dataset402_ULS_twd_model_full_res_core"

for output_path in [half_output_path, core_output_path]:
    os.makedirs(f"{output_path}/imagesTr", exist_ok=True)
    os.makedirs(f"{output_path}/labelsTr", exist_ok=True)
    os.makedirs(f"{output_path}/imagesTs", exist_ok=True)
    os.makedirs(f"{output_path}/labelsTs", exist_ok=True)

for split in ["/imagesTr", "/imagesTs"]:
    for file in os.listdir(full_data_path+split):
        image = full_data_path + split + "/" + file
        label = full_data_path + split.replace("images", "labels") + "/" + file.replace("_0000.nii.gz", ".nii.gz")
        print(image)

        data = {"img":image, "seg":label}
        half_img_out_path = FolderLayoutULS(
            case_id=file.replace(".nii.gz", ""),
            output_dir=half_output_path + split,
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )
        half_seg_out_path = FolderLayoutULS(
            case_id=file.replace("_0000.nii.gz", ""),
            output_dir=half_output_path + split.replace("images", "labels") + "/",
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )

        half = Compose(
            [
                LoadImaged(keys=["img", "seg"], image_only=False),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    spatial_size=(
                        256,
                        256,
                        1,
                    ),
                    label_key="seg",
                    neg=0,
                    num_samples=1,
                ),
                Resized(keys=["img"],
                        spatial_size=[128, 128, 1],
                        mode="trilinear"),
                Resized(keys=["seg"],
                        spatial_size=[128, 128, 1],
                        mode="nearest"),
                SaveImaged(
                    keys=["img"],
                    resample=False,
                    squeeze_end_dims=False,
                    folder_layout=half_img_out_path,
                ),
                SaveImaged(
                    keys=["seg"],
                    resample=False,
                    squeeze_end_dims=False,
                    folder_layout=half_seg_out_path,
                ),
            ]
        )
        half(data)

        core_img_out_path = FolderLayoutULS(
            case_id=file.replace(".nii.gz", ""),
            output_dir=output_path + split,
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )
        core_seg_out_path = FolderLayoutULS(
            case_id=file.replace("_0000.nii.gz", ""),
            output_dir=output_path + split.replace("images", "labels") + "/",
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )

        core = Compose(
            [
                LoadImaged(keys=["img", "seg"], image_only=False),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                RandCropByPosNegLabeld(
                    keys=["img", "seg"],
                    spatial_size=(
                        256,
                        256,
                        1,
                    ),
                    label_key="seg",
                    neg=0,
                    num_samples=1,
                ),
                CenterSpatialCropd(keys=["img", "seg"], roi_size=(128, 128, 1)),
                SaveImaged(
                    keys=["img"],
                    resample=False,
                    squeeze_end_dims=False,
                    folder_layout=core_img_out_path,
                ),
                SaveImaged(
                    keys=["seg"],
                    resample=False,
                    squeeze_end_dims=False,
                    folder_layout=core_seg_out_path,
                ),
            ]
        )
        core(data)
