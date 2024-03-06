import os
import time
import json
import torch
from scipy import ndimage
import SimpleITK as sitk
import numpy as np
from pathlib import Path
from evalutils import SegmentationAlgorithm
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.utilities.helpers import empty_cache


class Uls23(SegmentationAlgorithm):
    def __init__(self):
        self.image_metadata = None  # Keep track of the metadata of the input volume
        self.id = None  # Keep track of batched volume file name for export
        self.z_size = 128  # Number of voxels in the z-dimension for each VOI
        self.xy_size = 256  # Number of voxels in the xy-dimensions for each VOI
        self.device = torch.device("cuda")
        self.predictor = None # nnUnet predictor

    def start_pipeline(self):
        """
        Starts inference algorithm
        """
        start_time = time.time()

        # We need to create the correct output folder, determined by the interface, ourselves
        os.makedirs("/output/images/ct-binary-uls/", exist_ok=True)

        self.load_model()
        spacings = self.load_data()
        predictions = self.predict(spacings)
        self.postprocess(predictions)

        end_time = time.time()
        print(f"Total job runtime: {end_time - start_time}s")

    def load_model(self):
        start_model_load_time = time.time()
        # Set up the nnUNetPredictor
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False, # False is faster but less accurate
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        # Initialize the network architecture, loads the checkpoint
        self.predictor.initialize_from_trained_model_folder(
            "/opt/algorithm/nnunet/nnUNet_results/Dataset901_Filtered_FSUP/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc",
            use_folds=("all"),
            checkpoint_name="checkpoint_best.pth",
        )
        end_model_load_time = time.time()
        print(f"Model loading runtime: {end_model_load_time - start_model_load_time}s")

    def load_data(self):
        """
        1) Loads the .mha files containing the VOI stacks in the input directory
        2) Unstacks them into individual lesion VOI's
        3) Optional: preprocess volumes
        4) Predict per VOI
        """
        start_load_time = time.time()
        # Input directory is determined by the algorithm interface on GC
        input_dir = Path("/input/images/stacked-3d-ct-lesion-volumes/")

        # Load the spacings per VOI
        with open(Path("/input/stacked-3d-volumetric-spacings.json"), 'r') as json_file:
            spacings = json.load(json_file)

        for input_file in input_dir.glob("*.mha"):
            self.id = input_file

            # Load and keep track of the image metadata
            self.image_metadata = sitk.ReadImage(input_dir / input_file)

            # Now get the image data
            image_data = sitk.GetArrayFromImage(self.image_metadata)
            for i in range(int(image_data.shape[0] / self.z_size)):
                voi = image_data[self.z_size * i:self.z_size * (i + 1), :, :]
                # Note: spacings[i] contains the scan spacing for this VOI

                # Unstack the VOI's, perform optional preprocessing and save
                # them to individual binary files for memory-efficient access
                np.save(f"/tmp/voi_{i}.npy", np.array([voi])) # Add dummy batch dimension for nnUnet

        end_load_time = time.time()
        print(f"Data pre-processing runtime: {end_load_time - start_load_time}s")

        return spacings

    def predict(self, spacings):
        """
        Runs nnUnet inference on the images, then moves to post-processing
        :param spacings: list containing the spacing per VOI
        :return: list of numpy arrays containing the predicted lesion masks per VOI
        """
        start_inference_time = time.time()
        predictions = []
        for i, voi_spacing in enumerate(spacings):
            # Load the 3D array from the binary file
            voi = torch.from_numpy(np.load(f"/tmp/voi_{i}.npy"))
            voi = voi.to(dtype=torch.float32)

            print(f'\nPredicting image of shape: {voi.shape}, spacing: {voi_spacing}')
            predictions.append(self.predictor.predict_single_npy_array(voi, {'spacing': voi_spacing}, None, None, False))

        end_inference_time = time.time()
        print(f"Total inference runtime: {end_inference_time - start_inference_time}s")
        return predictions

    def postprocess(self, predictions):
        """
        Runs post-processing and saves stacked predictions.
        :param predictions: list of numpy arrays containing the predicted lesion masks per VOI
        """
        start_postprocessing_time = time.time()
        # Run postprocessing code here, for the baseline we only remove any
        # segmentation outputs not connected to the center lesion prediction
        for i, segmentation in enumerate(predictions):
            print(f"Post-processing prediction {i}")
            instance_mask, num_features = ndimage.label(segmentation)
            if num_features > 1:
                print("Found multiple lesion predictions")
                segmentation[instance_mask != instance_mask[
                    int(self.z_size / 2), int(self.xy_size / 2), int(self.xy_size / 2)]] = 0
                segmentation[segmentation != 0] = 1

            predictions[i] = segmentation

        predictions = np.concatenate(predictions, axis=0) # Stack predictions

        # Create mask image and copy over metadata
        mask = sitk.GetImageFromArray(predictions)
        mask.CopyInformation(self.image_metadata)

        sitk.WriteImage(mask, f"/output/images/ct-binary-uls/{self.id.name}")
        print("Output dir contents:", os.listdir("/output/images/ct-binary-uls/"))
        print("Output batched image shape:", predictions.shape)
        end_postprocessing_time = time.time()
        print(f"Postprocessing & saving runtime: {end_postprocessing_time - start_postprocessing_time}s")


if __name__ == "__main__":
    Uls23().start_pipeline()
