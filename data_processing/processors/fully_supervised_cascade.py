import os
import monai
from monai.transforms import (
    Compose,
    SaveImaged,
    LoadImaged,
    EnsureChannelFirstd,
    Orientationd,
    Resized,
    CenterSpatialCropd,
)
from folder_layout import FolderLayoutULS

full_data_path = ".../Dataset500_ULS_thd"
half_output_path = ".../Dataset501_ULS_thd_half_res"
core_output_path = ".../Dataset502_ULS_thd_full_res_core"

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
                Resized(keys=["img"],
                        spatial_size=[128, 128, 64],
                        mode="trilinear"),
                Resized(keys=["seg"],
                        spatial_size=[128, 128, 64],
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
            output_dir=core_output_path + split,
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )
        core_seg_out_path = FolderLayoutULS(
            case_id=file.replace("_0000.nii.gz", ""),
            output_dir=core_output_path + split.replace("images", "labels") + "/",
            postfix="",
            extension=".nii.gz",
            use_idx=False,
        )
        core = Compose(
            [
                LoadImaged(keys=["img", "seg"], image_only=False),
                EnsureChannelFirstd(keys=["img", "seg"]),
                Orientationd(keys=["img", "seg"], axcodes="RAS"),
                CenterSpatialCropd(keys=["img","seg"], roi_size=(128, 128, 64)),
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
