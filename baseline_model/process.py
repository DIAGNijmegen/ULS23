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
        self.image_metadata = None # Keep track of the metadata of the input volume
        self.id = None # Keep track of batched volume file name for export
        self.z_size = 128 # Number of voxels in the z-dimension for each VOI
        self.xy_size = 256  # Number of voxels in the xy-dimensions for each VOI
        self.device = torch.device("cuda")

    def start_pipeline(self):
        """
        Starts inference algorithm
        """
        start_time = time.time()

        # We need to create the correct output folder, determined by the interface, ourselves
        os.makedirs("/output/images/ct-binary-uls/", exist_ok=True)

        # Start general inference pipeline
        spacings = self.load_data()
        self.preprocess(spacings)
        predictions = self.predict(spacings)
        self.postprocess(predictions)

        end_time = time.time()
        print(f"Total job runtime: {end_time - start_time}s")

    def load_data(self):
        """
        Loads the .mha files and saves them to binary files for memory-efficient access of individual VOI's
        """
        start_load_time = time.time()
        # Determined by interface on GC
        input_dir = Path("/input/images/stacked-3d-ct-lesion-volumes/")

        for input_file in input_dir.glob("*.mha"):
            self.id = input_file

            img_path = input_dir / input_file

            # Load and track the image metadata (this does not load the image data into memory)
            self.image_metadata = sitk.ReadImage(img_path)

            # Now get the image data
            image_data = sitk.GetArrayFromImage(self.image_metadata)

            # Unstack the VOI's and save them to individual binary files
            for i in range(int(image_data.shape[0] / self.z_size)):
                img = image_data[self.z_size * i:self.z_size * (i + 1), :, :]
                np.save(f"/tmp/voi{i}.npy", img)

        # Load the spacings per VOI
        with open(Path("/input/stacked-3d-volumetric-spacings.json"), 'r') as json_file:
            spacings = json.load(json_file)

        end_load_time = time.time()
        print(f"Data loading runtime: {end_load_time - start_load_time}s")

        return spacings

    def preprocess(self, spacings):
        """
        Run preprocessing on individual VOI's
        :param spacings: list containing the spacing per VOI
        """
        start_prep_time = time.time()

        for i, voi_spacing in enumerate(spacings):
            # Load the 3D array from the binary file
            voi_data = np.load(f"/tmp/voi{i}.npy")

            # If you want to run preprocessing on individual images do that here:
            # ...

            # We only add a dummy batch dimension for nnUnet
            voi_data = np.array([voi_data])

            np.save(f"/tmp/pp_voi{i}.npy", voi_data)

        end_prep_time = time.time()
        print(f"Preprocessing runtime: {end_prep_time - start_prep_time}s")

    def predict(self, spacings):
        """
        Runs nnUnet inference on the images, then moves to post-processing
        :param spacings: list containing the spacing per VOI
        :return: list of numpy arrays containing the predicted lesion masks per VOI
        """
        # Instantiate the nnUNetPredictor
        start_model_load_time = time.time()

        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=False, # False is faster but less accurate
            device=self.device,
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=False
        )
        # Initialize the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            "/opt/algorithm/nnunet/nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc",
            use_folds=("all"),
            checkpoint_name="checkpoint_final.pth",
        )
        end_model_load_time = time.time()
        print(f"Model loading runtime: {end_model_load_time - start_model_load_time}s")

        # Run inference
        start_inference_time = time.time()
        predictions = self.inference_on_single_process(predictor, spacings)

        end_inference_time = time.time()
        print(f"Inference runtime: {end_inference_time - start_inference_time}s")
        return predictions

    def inference_on_single_process(self, predictor, spacings):
        """
        Removed all nnUnet multiprocessing code as that is not fully supported on GC.
        Also no resampling since we don't need that for the baseline model.

        Note 31/01/24: with the increase of shared memory of GC containers,
        regular multi-processing based nnUnet inference may now be feasible.
        See https://grand-challenge.org/blogs/january-2024-cycle-report/
        
        The challenge organizers will investigate and provide an updated
        version of the inference pipeline if this works.

        :param predictor: initialized nnUnet predictor
        :param spacings: list containing the spacing per VOI
        """
        predictions = []
        for i, voi_spacing in enumerate(spacings):
            # Load the 3D array from the binary file
            data = torch.from_numpy(np.load(f"/tmp/pp_voi{i}.npy"))
            data = data.to(dtype=torch.float32)

            print(f'\nPredicting image of shape: {data.shape}, spacing: {voi_spacing}')
            predicted_logits = predictor.predict_logits_from_preprocessed_data(data).cpu()

            predicted_probabilities = predictor.label_manager.apply_inference_nonlin(predicted_logits)
            del predicted_logits
            segmentation = predictor.label_manager.convert_probabilities_to_segmentation(predicted_probabilities)

            # segmentation may be torch.Tensor but we continue with numpy
            if isinstance(segmentation, torch.Tensor):
                segmentation = segmentation.cpu().numpy()

            # revert transpose
            segmentation = segmentation.transpose(predictor.plans_manager.transpose_backward)

            print(f'\nDone with image, prediction shape: {segmentation.shape}')
            
            # Convert mask to UInt8, default pixel datatype inferred from nnUnet
            # sitkInt64: "MET_LONG_LONG", not supported by GC
            predictions.append(segmentation.astype(np.uint8))

        # clear device cache
        empty_cache(self.device)
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
                segmentation[instance_mask != instance_mask[int(self.z_size/2), int(self.xy_size/2), int(self.xy_size/2)]] = 0
                segmentation[segmentation != 0] = 1

        predictions = np.concatenate(predictions, axis=0)

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
