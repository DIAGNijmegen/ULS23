import collections
import os

import numpy as np
import cv2
from monai.data import NibabelReader, MetaTensor
from monai.transforms import (
    Orientationd,
    EnsureChannelFirstd,
    Compose,
    SaveImaged,
    LoadImaged,
    BorderPadd,
    RandCropByPosNegLabeld,
    MapTransform,
    NormalizeIntensityd,
)
from scipy import ndimage
from skimage.measure import regionprops
from .folder_layout import FolderLayoutULS
from torch import Tensor

class GrabCutMaskSelectiond(MapTransform):

    def __init__(self, keys, window, measurement, morphop_iters, grabcut_iters, naive_shape_penalty, pixel_measurements):
        super().__init__(keys)
        self.window = window
        self.measurement = measurement
        self.morphop_iters = morphop_iters
        self.grabcut_iters = grabcut_iters
        self.naive_shape_penalty = naive_shape_penalty
        self.pixel_measurements = pixel_measurements

    def __call__(self, data):
        img_data, lbl_data = data[self.keys[0]], data[self.keys[1]]
        img_meta = img_data.meta
        lbl_meta = lbl_data.meta

        # Create GrabCut mask from annotations
        img, seg = img_data[0].numpy().T, lbl_data[0].numpy().T
        gc_mask = self.create_grabcut_mask(seg, self.morphop_iters)
        if type(gc_mask) == IndexError:
            return gc_mask

        # We compare the GC masks with the regular ellipse fitted to the measurements
        ellipse_mask = np.zeros(seg.shape)
        ellipse_mask[seg >= 2] = 1
        masks = [ellipse_mask]

        # Run GrabCut for each window level from metadata to get 2D lesion masks
        for i in range(len(self.window[0])):
            print(f"Running GC with metadata window #{i + 1}: {[self.window[0][i], self.window[1][i]]}")
            masks.append(
                self.grabcut(img, gc_mask, [self.window[0][i], self.window[1][i]], self.grabcut_iters))

        # Run GC with windows based on the ellipse mask +/- 50/100HU as a backup
        ellipse_window = np.zeros(gc_mask.shape)
        ellipse_window[gc_mask == 3] = 1
        ellipse_window = ellipse_window * img
        print(f"Running GC with ellipse windows: {[np.amin(ellipse_window) - 50, np.amax(ellipse_window) + 50]}, "
              f"{[np.amin(ellipse_window) - 100, np.amax(ellipse_window) + 100]}")
        masks.append(self.grabcut(img, gc_mask, [np.amin(ellipse_window) - 50, np.amax(ellipse_window) + 50], self.grabcut_iters))
        masks.append(self.grabcut(img, gc_mask, [np.amin(ellipse_window) - 100, np.amax(ellipse_window) + 100], self.grabcut_iters))

        # Substitute lesion mask with GrabCut mask if closer to measured diameters
        if self.pixel_measurements:
            lbl_data = Tensor(np.array([self.determine_closest_mask(masks, self.measurement, self.naive_shape_penalty, 1).T]))
        else:
            lbl_data = Tensor(np.array([self.determine_closest_mask(masks, self.measurement, self.naive_shape_penalty, img_meta["pixdim"][2]).T]))

        return {"img":MetaTensor(img_data, meta=img_meta), "seg":MetaTensor(lbl_data, meta=lbl_meta)}

    def create_grabcut_mask(self, anno, morphop_iters):
        ###############################################################################
        # Create GrabCut mask:
        # in the mask 0 is definite background, 1 definite foreground,
        # 2 probable background, and 3 probable foreground
        ###############################################################################
        z_slice = np.argwhere(anno)[0][0]
        el = np.zeros(anno.shape)  # Ellipse
        bb = np.zeros(anno.shape)  # Bounding box
        lc = np.zeros(anno.shape)  # Lesion core

        el[anno >= 2] = 1
        bb[anno == 1] = 1
        lc[anno >= 2] = 1

        try:
            lesion_props = regionprops(lc[z_slice].astype(int))
            min_ax_l = lesion_props[0].minor_axis_length
        except IndexError:
            return IndexError("Couldn't fit ellipse to lesion core for GC mask")

        # To get the lesion core we erode the ellipse by short-axis length // erosion_factor
        lc[z_slice] = ndimage.binary_erosion(
            lc[z_slice], iterations=int(min_ax_l // morphop_iters)
        )
        # Ellipse and Bounding box are dilated to provide more room for GC to adjust the contour
        el[z_slice] = ndimage.binary_dilation(el[z_slice])
        bb[z_slice] = ndimage.binary_dilation(bb[z_slice], iterations=int(min_ax_l // morphop_iters))

        gc_mask = np.zeros(anno.shape)
        # Inside bb is probable background
        gc_mask[bb == 1] = 2
        # Ellipse is probable foreground
        gc_mask[el == 1] = 3
        # Lesion core is definite foreground
        gc_mask[lc == 1] = 1

        return gc_mask

    def grabcut(self, img, gc_mask, window, iters):
        z_slice = np.argwhere(gc_mask)[0][0]
        # Apply window level
        img = self.window_and_normalize_ct(img, [window[1], window[0]], 0, 255)

        predictions = np.zeros(gc_mask.shape)
        if cv2.GC_FGD in gc_mask[z_slice, :, :]:
            temp_img = cv2.cvtColor(
                img[z_slice, :, :].astype(np.uint8), cv2.COLOR_GRAY2BGR
            )

            fgdModel = np.zeros((1, 65), dtype="float64")
            bgdModel = np.zeros((1, 65), dtype="float64")

            (outputmask, bgModel, fgModel) = cv2.grabCut(
                temp_img,
                np.ascontiguousarray(gc_mask[z_slice, :, :], dtype=np.uint8),
                None,
                bgdModel,
                fgdModel,
                iterCount=iters,
                mode=cv2.GC_INIT_WITH_MASK,
            )
            outputmask = np.where((outputmask == 2) | (outputmask == 0), 0, 1).astype(
                "uint8"
            )
            out_img = temp_img * outputmask[:, :, np.newaxis]

            out_img[out_img > 0] = 1
            out_img = cv2.cvtColor(out_img, cv2.COLOR_BGR2GRAY)

            predictions[z_slice, :, :] = out_img

            # Closing operation to smooth mask
            predictions[z_slice] = ndimage.binary_closing(predictions[z_slice])

        labeled_seg, num_features = ndimage.label(predictions)
        if num_features > 1:
            # Remove all but largest component
            largest_component = collections.Counter(x for x in labeled_seg.flatten() if x != 0).most_common(1)[0][0]
            labeled_seg[labeled_seg != largest_component] = 0
            labeled_seg[labeled_seg == largest_component] = 1

        return labeled_seg

    @staticmethod
    def determine_closest_mask(masks, measurements, naive_shape_penalty, axial_spacing):
        differences = []
        for idx, mask in enumerate(masks):
            try:
                z_slice = np.argwhere(mask)[0][0]
                lesion_props = regionprops(mask[z_slice].astype(int))
                maj_ax_l = lesion_props[0].major_axis_length * axial_spacing
                min_ax_l = lesion_props[0].minor_axis_length * axial_spacing
                dif = (abs(100 - measurements[0]/(maj_ax_l/100)) + abs(100 - measurements[0]/(min_ax_l/100)))/2
                if idx == 0:
                    # The naive shape penalty penalizes the 'naive' ellipse mask by the specified percentage of error.
                    # This is useful for if you want to prefer a GrabCut mask, even though it might result in a larger
                    # measurement error. The GC mask might fit the overal 2D shape of the lesion better since it at least
                    # tries to determine the contour of the lesion. We use a 2.5% handicap.
                    dif += naive_shape_penalty
            except:
                dif = 9999 # Grabcut mask failure

            differences.append(dif)

        print("Average percentage long/short-axis measurement difference per mask:", differences)
        return masks[differences.index(min(differences))]

    @staticmethod
    def window_and_normalize_ct(array, window, min_val, max_val):
        arr_copy = array.copy()
        arr_copy[arr_copy > window[0]] = window[0]
        arr_copy[arr_copy < window[1]] = window[1]
        return min_val + (
                (arr_copy - np.amin(arr_copy))
                * (max_val - min_val)
                / (np.amax(arr_copy) - np.amin(arr_copy))
        )


class PartiallyAnnotatedLesionExtractor(object):
    def __init__(
            self,
            output_path,
            depth=32,
            height_width=64,
            num_samples=1,
            pixel_measurements=True,
            morphop_iters=5,
            grabcut_iters=25,
            naive_shape_penalty=2.5,
    ):
        self.output_path = output_path
        self.depth = depth
        self.height_width = height_width
        self.num_samples = num_samples
        self.morphop_iters = morphop_iters
        self.grabcut_iters = grabcut_iters
        self.naive_shape_penalty = naive_shape_penalty
        self.pixel_measurements = pixel_measurements

        os.makedirs(self.output_path + "/imagesTr", exist_ok=True)
        os.makedirs(self.output_path + "/labelsTr", exist_ok=True)

    def process_dataset(self, data, windows, measurements):
        already_finished = os.listdir(self.output_path + "/imagesTr")
        for case in data:
            print("Case:", case["img"])
            case_id = case["img"].split("/")[-1].split(".")[0]
            if f"{case_id}_sample_{self.num_samples-1}.nii.gz" not in already_finished:
                load_data = Compose(
                    [
                        LoadImaged(keys=["img", "seg"], image_only=False),
                        EnsureChannelFirstd(keys=["img", "seg"], channel_dim="no_channel"),
                        # NormalizeIntensityd(keys="img"), # Uncomment if you want to normalize based on the full volumes
                        Orientationd(keys=["img", "seg"], axcodes="RAS"),
                        GrabCutMaskSelectiond(keys=["img", "seg"],
                                              window=windows[case_id],
                                              measurement=measurements[case_id],
                                              morphop_iters=self.morphop_iters,
                                              grabcut_iters=self.grabcut_iters,
                                              naive_shape_penalty=self.naive_shape_penalty,
                                              pixel_measurements=self.pixel_measurements),
                    ]
                )

                lesion_data = load_data(case)
                if isinstance(lesion_data, Exception):
                    print(f"Skipping case {case_id}, reason: {lesion_data}")
                    continue

                img_out_path = FolderLayoutULS(
                    case_id=case_id,
                    output_dir=self.output_path + "/imagesTr",
                    postfix=f"_sample_",
                    extension=".nii.gz",
                )
                seg_out_path = FolderLayoutULS(
                    case_id=case_id,
                    output_dir=self.output_path + "/labelsTr",
                    postfix=f"_sample_",
                    extension=".nii.gz",
                )

                extract_lesion = Compose(
                    [
                        BorderPadd(
                            keys=["img"],
                            spatial_border=(
                                self.height_width // 2,
                                self.height_width // 2,
                                self.depth // 2,
                            ),
                            mode="constant",
                            constant_values=np.amin(lesion_data["img"])-1, # -1 so we can re-identify padded area later
                            value=np.amin(lesion_data["img"]),
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
                lesion_data = extract_lesion(lesion_data)