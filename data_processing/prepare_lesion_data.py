import os
import pandas as pd
from pydicom import dcmread
from monai.utils import set_determinism
from monai.data import NibabelReader
from processors.supervised_processor import FullyAnnotatedLesionExtractor
from processors.semi_supervised_processor import PartiallyAnnotatedLesionExtractor
from processors.dl_preprocessing import DeepLesionPreprocessor
from processors.ccc_preprocessing import CrowdsCureCancerPreprocessor

set_determinism(seed=42)


def create_supervised_lesion_data(
    archives_folder, processed_data_path, fully_supervised_archives
):
    """
    Process archives where we have fully 3D annotated lesions. These datasets need to be
    converted to nifti beforehand and should contain an imagesTr/labelsTr folder with the data.
    """
    for archive, lesion_mask in fully_supervised_archives:
        print("Initializing lesion extractor for dataset:", archive)
        img_path = os.path.join(archives_folder, archive, "imagesTr")
        lbl_path = os.path.join(archives_folder, archive, "labelsTr")

        data = []
        for file in [fn for fn in os.listdir(img_path) if fn[0] != "."]:
            data.append(
                {
                    "img": os.path.join(img_path, file),
                    "seg": os.path.join(lbl_path, file),
                }
            )

        short_axis = False
        instance = False
        cca = False
        if archive in ["NIH_LN/MED", "NIH_LN/ABD"]:
            print("Using short-axis to determine minimum size (for lymph nodes).")
            short_axis = True
            instance = (
                True  # The LN datasets are already instance segmentation annotated
            )
        if archive in ["DeepLesion3D"]:
            print("Using only largest annotated area for lesion.")
            cca = True

        le = FullyAnnotatedLesionExtractor(
            output_path=os.path.join(processed_data_path, archive),
            depth=128,
            height_width=256,
            resample=False,
            figs=False,
            min_lesion_size_pixel=5,
            exclude_using_short_axis_size=short_axis,
            is_instance_labeled=instance,
            lesion_label=lesion_mask,
            num_samples=1,
            one_lesion_per_scan=cca,
        )
        print("Running lesion extractor...")
        le.process_dataset(data)
        print("Finished lesion extraction")


def create_semi_supervised_lesion_data(
    archives_folder, processed_data_path, partially_supervised_archives, preprocess=True
):
    """
    Two-step process for preparing the semi-supervised data. First we create labels from the measurements,
    by fitting an ellipse to the long/short-axis and adding (DeepLesion) or computing a bounding box (CCC18).
    Then we use GrabCut to try and fit a better mask using the created labels, if the basic ellipse corresponds
    better to the measurements made by the original annotators we stick to that.
    """

    if preprocess:
        # Step 1A: convert DeepLesion .png's to .nii.gz and write annotations to label
        print("Preparing DeepLesion masks based on measurements")
        dlp = DeepLesionPreprocessor(
            archive_path=os.path.join(archives_folder, "DeepLesion/"),
            output_path=os.path.join(archives_folder, "DeepLesion_preprocessed/"),
            min_context=3,
        )
        dlp.create_volumes()

        # Step 1B: prepare CCC18 annotations as labels
        print("Preparing CCC18 masks based on measurements")
        cccp = CrowdsCureCancerPreprocessor(
            archive_path=os.path.join(archives_folder, "CCC18/"),
            output_path=os.path.join(archives_folder, "CCC18_preprocessed/"),
        )
        cccp.create_volumes()

    # Step 2: create 2D GrabCut masks
    for archive, metadata_path in partially_supervised_archives:
        print("Creating GrabCut masks for dataset:", archive)
        img_path = os.path.join(archives_folder, archive, "images")
        lbl_path = os.path.join(archives_folder, archive, "labels")

        # We need to prepare: img/label from step 2, dicom window, the original measurements
        data, windows, measurements = [], {}, {}
        metadata = pd.read_csv(metadata_path, engine="python")
        for file in [fn for fn in os.listdir(img_path) if fn[0] != "."]:
            data.append(
                {
                    "img": os.path.join(img_path, file),
                    "seg": os.path.join(lbl_path, file),
                }
            )
            if archive == "DeepLesion_preprocessed":
                # DL metadata contains measurements in pixels
                measurement_in_pixel = True

                # Look up actual measured diameters
                lesion_dia_pix = (
                    metadata.loc[
                        (metadata["File_name"] == file.replace(".nii.gz", ".png")),
                        "Lesion_diameters_Pixel_",
                    ]
                    .iat[0]
                    .split(",")
                )
                dicom_window = (
                    metadata.loc[
                        (metadata["File_name"] == file.replace(".nii.gz", ".png")),
                        "DICOM_windows",
                    ]
                    .iat[0]
                    .split(",")
                )

                # Long/Short axis diameters of lesion
                dia1, dia2 = (
                    float(lesion_dia_pix[0]),
                    float(lesion_dia_pix[1]),
                )
                measurements[file.replace(".nii.gz", "")] = (dia1, dia2)
                # Window of lesion
                windows[file.replace(".nii.gz", "")] = (
                    [round(float(dicom_window[0]))],
                    [round(float(dicom_window[1]))],
                )
            elif archive == "CCC18_preprocessed":
                # CCC18 metadata contains measurements in mm
                measurement_in_pixel = False

                # Extract window level from DICOM metadata
                zslice = dcmread(
                    "/".join(metadata_path.split("/")[:-1])
                    + "/dicoms/"
                    + file.replace(".nii.gz", ".dcm")
                )

                # Window Center (0028,1050) and Window Width (0028,1051)
                try:
                    # Multiple windows
                    windows[file.replace(".nii.gz", "")] = (
                        [int(x) for x in zslice[0x00281050].value],
                        [int(x) for x in zslice[0x00281051].value],
                    )
                except TypeError:
                    # Single window
                    windows[file.replace(".nii.gz", "")] = (
                        [int(zslice[0x00281050].value)],
                        [int(zslice[0x00281051].value)],
                    )

                # Long/Short axis diameters of lesion
                ll = metadata.loc[
                    metadata["name"] == file.replace(".nii.gz", "/"),
                    "LongAxis.length",
                ].iat[0]
                sl = metadata.loc[
                    metadata["name"] == file.replace(".nii.gz", "/"),
                    "ShortAxis.length",
                ].iat[0]
                measurements[file.replace(".nii.gz", "")] = (float(ll), float(sl))
            else:
                raise NotImplementedError

        le = PartiallyAnnotatedLesionExtractor(
            output_path=os.path.join(processed_data_path, archive.replace("_preprocessed", "_grabcut")),
            depth=128,
            height_width=256,
            num_samples=1,
            pixel_measurements=measurement_in_pixel
        )
        print("Running lesion extractor...")
        le.process_dataset(data, windows, measurements)
        print("Finished lesion extraction")


def main():
    archives_folder = "..."
    output_path = "..."
    os.makedirs(output_path, exist_ok=True)

    ####################################################################################################################
    # Preparing the supervised lesion data from e.g. LiTS, KiTS, LIDC-IDRI
    # This function expects all supervised datasets to be stored under 'archives_folder' with an imagesTr and labelsTr
    # folder. The scans and annotations should have been converted to nifti beforehand.
    ####################################################################################################################
    # [Archive name, lesion mask], -1 for instance segmented datasets like NIH_LN
    fully_supervised_archives = [
        ["NIH_LN/MED", -1],
        ["NIH_LN/ABD", -1],
        ["DeepLesion3D", 1],
        ["diag_boneCT", 1],
        ["LiTS", 2],
        ["diag_pancreasCT", 1],
        ["kits21-master", 2],
        ["LIDC-IDRI", 1],
        ["LNDb", 1],
        ["MDSC/Task06_Lung", 1],
        ["MDSC/Task07_Pancreas", 2],
        ["MDSC/Task10_Colon", 1],
    ]
    create_supervised_lesion_data(archives_folder, output_path, fully_supervised_archives)

    ####################################################################################################################
    # Preparing the semi-supervised lesion data from long/short-axis measurements e.g. DeepLesion
    # For DeepLesion the png's should be stored under /images and the archive root should contain the metadata
    # (DL_info.csv). For CCC18 3D nifti volumes should be created from the DICOMS and the z-slice containing the
    # measurement should be determined using the dicom metadata provided. It's a hassle though, so I can't recommend
    # rerunning the data generation for it.
    ####################################################################################################################
    # [Archive name, metadata_location]
    partially_supervised_archives = [
        ["CCC18_preprocessed", archives_folder + "/CCC18/combined_anno_ccc18_with_z.csv"],
        ["DeepLesion_preprocessed", archives_folder + "/DeepLesion/DL_info.csv"],
    ]
    create_semi_supervised_lesion_data(
        archives_folder, output_path, partially_supervised_archives, False
    )


if __name__ == "__main__":
    main()
