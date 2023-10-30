import os
import numpy as np
import nibabel as nib
import cv2
from skimage.measure import EllipseModel
import skimage.draw as draw
from PIL import Image
import pandas as pd
from scipy import ndimage


class DeepLesionPreprocessor(object):
    def __init__(self, archive_path, output_path, min_context=3):
        self.archive_path = archive_path
        self.output_path = output_path
        self.min_context = min_context

        os.makedirs(self.output_path + "images/", exist_ok=True)
        os.makedirs(self.output_path + "labels/", exist_ok=True)

    def create_volumes(self):
        df = pd.read_csv(self.archive_path + "DL_info.csv", engine="python")
        for root, dirs, files in os.walk(self.archive_path + "/images"):
            files.sort()
            ############################################################################################
            # Create a 3D volume from the 2D slices and the metadata in DL_info.csv
            ############################################################################################

            # Set pixel spacing in header
            sx, sy, sz = 0, 0, 0
            for file in files:
                try:
                    ffn = file.split(".")[0]
                    spacing = df.loc[
                        (df["File_name"] == root.split("/")[-1] + "_" + file),
                        "Spacing_mm_px_",
                    ]
                    # Check if scan has valid spacing and not already processed
                    if len(spacing) > 0:
                        # Check for possibly noisy tag
                        pn = df.loc[
                            (df["File_name"] == root.split("/")[-1] + "_" + file),
                            "Possibly_noisy",
                        ].iat[0]

                        if int(pn) != "1":
                            # Look up actual measured diameters
                            lesion_dia_pix = (
                                df.loc[
                                    (
                                        df["File_name"]
                                        == root.split("/")[-1] + "_" + file
                                    ),
                                    "Lesion_diameters_Pixel_",
                                ]
                                .iat[0]
                                .split(",")
                            )
                            lm_cords = df.loc[
                                (df["File_name"] == root.split("/")[-1] + "_" + file),
                                "Measurement_coordinates",
                            ]
                            bb_cords = df.loc[
                                (df["File_name"] == root.split("/")[-1] + "_" + file),
                                "Bounding_boxes",
                            ]
                            slice_range = df.loc[
                                (df["File_name"] == root.split("/")[-1] + "_" + file),
                                "Slice_range",
                            ]

                            sx, sy, sz = [float(x) for x in spacing.iat[0].split(",")]
                            lm_cords = [int(x) for x in lm_cords.iat[0].split(",")]
                            bb_cords = [float(x) for x in bb_cords.iat[0].split(",")]

                            start_s, stop_s = [
                                int(x) for x in slice_range.iat[0].split(",")
                            ]

                            # Long/Short axis diameters of lesion
                            dia1, dia2 = (
                                float(lesion_dia_pix[0]),
                                float(lesion_dia_pix[1]),
                            )
                            self.lesion_sizes = [dia1, dia2]

                            # Load image
                            complete_img = np.array(
                                [
                                    self.read_DL_slice(root + "/" + img)
                                    for img in files
                                    if img.split(".")[-1] == "png"
                                    and img.split(".")[0]
                                    in [
                                        format(i, "03")
                                        for i in range(start_s, stop_s + 1)
                                    ]
                                ]
                            )

                            max_up = stop_s - int(ffn)
                            max_down = int(ffn) - start_s

                            # Some cases have the key slice on the extreme edge of the volume...
                            # Use minimum context above and below, drop other cases
                            if max_up >= self.min_context and self.min_context >= 2:
                                centered_image = complete_img[
                                    int(ffn)
                                    - start_s
                                    - min(max_up, max_down) : int(ffn)
                                    - start_s
                                    + min(max_up, max_down)
                                    + 1
                                ]

                                # Create Nifti image
                                img_volume = nib.Nifti1Image(
                                    np.flip(centered_image.T), np.eye(4)
                                )
                                img_volume.header.set_zooms(np.array([sx, sy, sz]))
                                # img_volume.header.set_data_shape(
                                #     centered_image.shape
                                # )
                                # img_volume.header.set_data_dtype(np.float64)

                                ############################################################################################
                                # Prepare label from lesion measurements, fitted elipse and bounding box
                                ############################################################################################
                                # Calculate lines based on cords
                                lines = cv2.line(
                                    np.zeros(centered_image[0].shape),
                                    (lm_cords[0], lm_cords[1]),
                                    (lm_cords[2], lm_cords[3]),
                                    color=1,
                                    thickness=1,
                                )

                                lines = cv2.line(
                                    lines,
                                    (lm_cords[4], lm_cords[5]),
                                    (lm_cords[6], lm_cords[7]),
                                    color=1,
                                    thickness=1,
                                )
                                lines[lines > 1] = 1

                                # Calculate an ellipse based on line annotations
                                points = np.argwhere(lines)
                                ell = EllipseModel()
                                ell.estimate(points)

                                xc, yc, a, b, theta = ell.params

                                rr, cc = draw.ellipse(
                                    xc, yc, 2 * a, 2 * b, rotation=theta
                                )
                                elipse = np.zeros(centered_image[0].shape)
                                elipse[rr, cc] = 2

                                # Adaptively erode ellipse until it better fits measurements
                                combined_annotations = np.zeros(centered_image[0].shape)
                                combined_annotations += lines
                                combined_annotations += elipse
                                while not np.any(combined_annotations == 1):
                                    elipse = ndimage.binary_erosion(elipse) * 2
                                    combined_annotations = np.zeros(
                                        centered_image[0].shape
                                    )
                                    combined_annotations += lines
                                    combined_annotations += elipse
                                # Dilate once so the ellipse covers the measurements again
                                elipse = ndimage.binary_dilation(elipse) * 2

                                # Add bb from cords
                                combined_annotations = np.zeros(centered_image.shape)
                                combined_annotations[
                                    combined_annotations.shape[0] // 2,
                                    int(bb_cords[1]) : int(bb_cords[3]) + 1,
                                    int(bb_cords[0]) : int(bb_cords[2]) + 1,
                                ] = 1

                                # Combine annotation: bg = 0, bb = 1, elipse = 2, measurements = 3
                                combined_annotations[
                                    combined_annotations.shape[0] // 2
                                ] += elipse
                                combined_annotations[
                                    combined_annotations == 2
                                ] = 0  # Set ellipse outside of bb to bg
                                combined_annotations[combined_annotations > 2] = 2
                                combined_annotations[
                                    combined_annotations.shape[0] // 2
                                ] += lines  # Add line annotations

                                # Create Nifti image
                                label_volume = nib.Nifti1Image(
                                    np.flip(combined_annotations.T), np.eye(4)
                                )
                                label_volume.header.set_zooms(np.array([sx, sy, sz]))

                                # Save label & image
                                nib.save(
                                    img_volume,
                                    f"{self.output_path}images/{root.split('/')[-1]}_{ffn}.nii.gz",
                                )
                                nib.save(
                                    label_volume,
                                    f"{self.output_path}labels/{root.split('/')[-1]}_{ffn}.nii.gz",
                                )
                except Exception as e:
                    print(
                        f"Processing {root.split('/')[-1]}_{ffn}.nii.gz resulted in {e}"
                    )

    @staticmethod
    def read_DL_slice(path):
        """
        For DeepLesion we must subtract 32768 from pixel intensities to obtain HUs
        """
        img = Image.open(path)
        np_img = np.array(img)
        np_img = np_img - 32768

        return np_img
