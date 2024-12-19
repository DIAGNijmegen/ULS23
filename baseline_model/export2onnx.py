import argparse
import torch
from pathlib import Path
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

def convert_nnunet_model_to_onnx(predictor):
    net = predictor.network

    # IMPORTANT: Actually load the data from the loaded checkpoint
    # in list_of_parameters (only one fold used, so first element)
    print("Loading weights...", flush=True)
    net.load_state_dict(predictor.list_of_parameters[0])

    dummy_input = torch.randn(1, 1, 128, 256, 256, requires_grad=True)

    # Set to eval mode
    net.eval()

    print("Export start...", flush=True)
    onnx_out = torch.onnx.dynamo_export(net, dummy_input)
    onnx_out.save("ULS23_test.onnx")
    print("Export done", flush=True)

if __name__ == "__main__":
    # Note: ensure upgraded libraries in docker image:
    # pip3 install --upgrade torch onnx onnxruntime onnxscript torchvision numpy scipy protobuf

    device = torch.device('cpu')
    print("Using", device)

    # Initialize the predictor
    predictor = nnUNetPredictor(
        tile_step_size=0.5,
        use_gaussian=True,
        use_mirroring=False,
        device=device,
        verbose=False,
        verbose_preprocessing=False,
        allow_tqdm=False
    )

    print("Loading checkpoint...", flush=True)
    # Initialize the network architecture, loads the checkpoint of the model you want to export, but doesn't load it to the model yet!
    predictor.initialize_from_trained_model_folder(
        r"/input/challenge/experiments/nnUNet_results/Dataset400_FSUP_ULS/nnUNetTrainer_ULS_500_QuarterLR__nnUNetPlansNoRs__3d_fullres_resenc",
        use_folds=("all"),
        checkpoint_name="checkpoint_final.pth",
    )

    convert_nnunet_model_to_onnx(predictor)
