import numpy as np
import nibabel as nib
import os
import pandas as pd
import cv2
from skimage.measure import EllipseModel
import skimage.draw as draw
import math
from scipy import ndimage
from pydicom import dcmread


class CrowdsCureCancerPreprocessor(object):
    def __init__(
        self,
        archive_path,
        output_path,
    ):
        self.archive_path = archive_path
        self.output_path = output_path
        os.makedirs(self.output_path + "images/", exist_ok=True)
        os.makedirs(self.output_path + "labels/", exist_ok=True)

    def create_volumes(self):
        # Load csv
        self.df = pd.read_csv(
            self.archive_path + "/combined_anno_ccc18_with_z.csv", engine="python"
        )

        for image_path in [
            self.archive_path + "imagesTr/" + f
            for f in os.listdir(self.archive_path + "imagesTr/")
            if not f.startswith(".")
        ]:
            try:
                print("scan:", image_path)
                image_path = image_path

                image = nib.load(image_path)

                # Normalize orientation
                orig_ornt = nib.io_orientation(image.affine)
                targ_ornt = nib.orientations.axcodes2ornt("IPL")
                transform = nib.orientations.ornt_transform(orig_ornt, targ_ornt)

                image = image.as_reoriented(transform)

                print(image_path.split("/")[-1].split(".")[0] + "/")
                # Look up cords
                lx1 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "LongAxis.point1.x",
                ].iat[0]
                lx2 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "LongAxis.point2.x",
                ].iat[0]
                ly1 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "LongAxis.point1.y",
                ].iat[0]
                ly2 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "LongAxis.point2.y",
                ].iat[0]
                ll = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "LongAxis.length",
                ].iat[0]

                sx1 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "ShortAxis.point1.x",
                ].iat[0]
                sx2 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "ShortAxis.point2.x",
                ].iat[0]
                sy1 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "ShortAxis.point1.y",
                ].iat[0]
                sy2 = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "ShortAxis.point2.y",
                ].iat[0]
                sl = self.df.loc[
                    self.df["name"] == image_path.split("/")[-1].split(".")[0] + "/",
                    "ShortAxis.length",
                ].iat[0]

                # Match lesion slice
                img_data = image.get_fdata()
                print(
                    "loading dicom z-slice from",
                    self.archive_path
                    + "dicoms/"
                    + image_path.split("/")[-1].split(".")[0]
                    + ".dcm",
                )
                zslice = dcmread(
                    self.archive_path
                    + "dicoms/"
                    + image_path.split("/")[-1].split(".")[0]
                    + ".dcm"
                ).pixel_array
                z = -1
                for i in range(img_data.shape[0]):
                    if np.allclose(
                        zslice.astype("uint8"), img_data[i].astype("uint8"), rtol=1e-2
                    ):
                        print("found z coordinate", i)
                        z = i
                        break
                if z != -1:
                    print("lesion cords", lx1, ly1, lx2, ly2, sx1, sy1, sx2, sy2)
                    lm_cords = [
                        int(x) for x in [lx1, ly1, lx2, ly2, sx1, sy1, sx2, sy2]
                    ]
                    print("mm measurements", ll, sl)

                    # Minimum of 5mm long-axis to include
                    if ll > 5:
                        # Long/Short axis diameters of lesion
                        dia1, dia2 = (
                            math.dist([lx1, ly1], [lx2, ly2]),
                            math.dist([sx1, sy1], [sx2, sy2]),
                        )
                        self.lesion_sizes = [dia1, dia2]
                        print("pixel measurements", ll, sl)

                        # Calculate lines based on cords
                        lines = cv2.line(
                            np.zeros(img_data[z].shape),
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

                        # Calculate ellipse based on line annotations
                        points = np.argwhere(lines)
                        ell = EllipseModel()
                        ell.estimate(points)
                        xc, yc, a, b, theta = ell.params

                        rr, cc = draw.ellipse(xc, yc, 2 * a, 2 * b, rotation=theta)
                        elipse = np.zeros(img_data[z].shape)
                        elipse[rr, cc] = 2

                        # Adaptively erode ellipse until it better fits measurements
                        combined_annotations = np.zeros(img_data[z].shape)
                        combined_annotations += lines
                        combined_annotations += elipse
                        while not np.any(combined_annotations == 1):
                            elipse = ndimage.binary_erosion(elipse) * 2
                            combined_annotations = np.zeros(img_data[z].shape)
                            combined_annotations += lines
                            combined_annotations += elipse
                        # Dilate once so the ellipse covers the measurements again
                        elipse = ndimage.binary_dilation(elipse) * 2

                        # Create bounding box from measurements
                        bb_cords = self.bounding_box([[x, y] for x, y in zip(rr, cc)])

                        # Combine annotation: bg = 0, bb = 1, elipse = 2, measurements = 3
                        combined_annotations = np.zeros(img_data[z].shape)
                        combined_annotations[
                            bb_cords[0][0] : bb_cords[1][0],
                            bb_cords[0][1] : bb_cords[1][1],
                        ] = 1
                        combined_annotations += elipse
                        combined_annotations[
                            combined_annotations == 2
                        ] = 0  # Set ellipse outside of bb to bg
                        combined_annotations[combined_annotations > 2] = 2
                        combined_annotations += lines  # Add line annotations
                        label = np.zeros(img_data.shape)
                        label[z] = combined_annotations

                        # Create Nifti image
                        label_volume = nib.Nifti1Image(label, image.affine)
                        label_volume.header.set_zooms(image.header.get_zooms())

                        # Save label & image
                        nib.save(
                            image,
                            f"{self.output_path}images/{image_path.split('/')[-1]}",
                        )
                        nib.save(
                            label_volume,
                            f"{self.output_path}labels/{image_path.split('/')[-1]}",
                        )
            except Exception as e:
                print(f"Processing {image_path} resulted in {e}")

    @staticmethod
    def bounding_box(points):
        x_coordinates, y_coordinates = zip(*points)
        return [
            (min(x_coordinates), min(y_coordinates)),
            (max(x_coordinates), max(y_coordinates)),
        ]
