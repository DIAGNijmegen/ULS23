import os
import torch
from torch import onnx as onnx
import numpy as np
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

# Build the algorithm docker container first using '. build.sh' then run this command
# docker run -it --entrypoint bash -v $(pwd):/output uls23:latest
# Now start this script from the container: python3.9 export2onnx.py

if __name__ == "__main__":
    print("loading model...")
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=torch.device("cpu"),
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )
    # Initialize the network architecture, loads the checkpoint of the model you want to export
    predictor.initialize_from_trained_model_folder(
        "/opt/algorithm/nnunet/nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc",
        use_folds=("all"),
        checkpoint_name="checkpoint_final.pth",
    )

    # Retrieve the model and set it to eval mode
    model = predictor.network
    print(model)

    # We add a (1, 1,) to the shape here to function as a channel dimension and batch size
    patch_size = (1, 1, 128, 256, 256)

    # Generate some random input data
    dummy_input = np.random.randn(*patch_size)

    # Export the model
    print("Saving model...")
    os.makedirs("/output/onnx_model/")
    torch.onnx.export(model, torch.from_numpy(dummy_input).type(torch.float), "/output/onnx_model/ULS_nnUnet.onnx", export_params=True, verbose=True)
    print("Finished :)")