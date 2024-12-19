import torch
import time
import numpy as np
import onnxruntime as ort
import onnx
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
import nibabel as nib
import os
from onnxruntime.quantization import quantize_dynamic, QuantType

def apply_threshold(prediction, threshold=0.5):
    """Apply a threshold to the prediction to obtain a binary mask."""
    return (prediction >= threshold).astype(np.uint8)

def postprocess_output(binary_mask: np.ndarray, original_shape: tuple) -> np.ndarray:
    """Post-process the output to match the original shape."""
    if binary_mask.shape[1] > 1:
        binary_mask = np.argmax(binary_mask, axis=1)
    binary_mask = np.squeeze(binary_mask, axis=0)
    binary_mask = np.transpose(binary_mask, (1, 2, 0))
    binary_mask = np.expand_dims(binary_mask, axis=-1)
    final_shape = original_shape[:-1] + (1,)
    assert binary_mask.shape == final_shape, f"Output shape {binary_mask.shape} does not match expected shape {final_shape}"
    return binary_mask

def run_onnx_inference(session, input_data):
    """Run inference using an ONNX model on CPU and return the output."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    y_pred = session.run([output_name], {input_name: input_data})[0]
    binary_mask = apply_threshold(y_pred)
    return postprocess_output(binary_mask, (256, 256, 128, 1))

def benchmark_onnx_inference(session, nifti_data, num_runs=10):
    """Benchmark ONNX inference on CPU."""
    print("Running ONNX inference benchmark...")
    times = []
    for i in range(num_runs):
        start_time = time.time()
        _ = run_onnx_inference(session, nifti_data)
        end_time = time.time()
        times.append(end_time - start_time)
        print(f"Run {i + 1}: {times[-1]:.4f} seconds")
    return {"average": np.mean(times), "min": np.min(times), "max": np.max(times), "std": np.std(times)}

def quantize_onnx_model(input_model_path, output_model_path):
    """Apply dynamic quantization to the ONNX model and time the process."""
    start_time = time.time()
    quantize_dynamic(input_model_path, output_model_path, weight_type=QuantType.QInt8)
    end_time = time.time()
    quantization_time = end_time - start_time
    print(f"Quantized model saved to {output_model_path}")
    print(f"Quantization took {quantization_time:.4f} seconds")
    return quantization_time

def setup_onnx_sessions(onnx_model_path, quantized_model_path):
    """Setup ONNX sessions with different optimization levels."""
    # Check if the CPU supports BF16 (required for optimal performance)
    supported_providers = ort.get_available_providers()
    print("Available providers:", supported_providers)

    sess_options = ort.SessionOptions()
    sess_options.execution_mode = ort.ExecutionMode.ORT_PARALLEL # To use inter op threads
    sess_options.use_deterministic_compute = False
    available_cpus = len(os.sched_getaffinity(0))
    print("Using available CPUs: ", available_cpus)
    # To approximate GrandChallenge provided AWS instance configuration with 4vCPU try 2/1, 2/2, 4/1, 4/1
    sess_options.inter_op_num_threads = 4 # available_cpus
    sess_options.intra_op_num_threads = 2 # available_cpus

    # Session with no optimizations
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    session_no_optim = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    print("session_no_optim: ")
    print(f"  Graph Optimization Level: {sess_options.graph_optimization_level}")
    print(f"  Enable Profiling: {sess_options.enable_profiling}")
    print(f"  Intra-op Num Threads: {sess_options.intra_op_num_threads}")
    print(f"  Inter-op Num Threads: {sess_options.inter_op_num_threads}")
    print(f"  Log Severity Level: {sess_options.log_severity_level}")
    print(f"  Use Deterministic Compute: {sess_options.use_deterministic_compute}")

    # Session with full ONNX runtime optimizations (default?)
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL # Offline Mode may save time by saving to disk...
    session_optim = ort.InferenceSession(onnx_model_path, sess_options=sess_options, providers=["CPUExecutionProvider"])
    print("session_optim: ")
    print(f"  Graph Optimization Level: {sess_options.graph_optimization_level}")
    print(f"  Enable Profiling: {sess_options.enable_profiling}")
    print(f"  Intra-op Num Threads: {sess_options.intra_op_num_threads}")
    print(f"  Inter-op Num Threads: {sess_options.inter_op_num_threads}")
    print(f"  Log Severity Level: {sess_options.log_severity_level}")
    print(f"  Use Deterministic Compute: {sess_options.use_deterministic_compute}")

    # Set session options for BF16
    sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED # Note: level lower than ORT_ENABLE_ALL
    sess_options.enable_profiling = True  # Optional, to profile BF16 usage

    # Enable BF16 for activations if supported
    providers = [("CPUExecutionProvider", {"arena_extend_strategy": "kNextPowerOfTwo",
                                           "intra_op_num_threads": 1,
                                           "enable_bf16": True})]

    # Create an ONNX Runtime session with BF16 enabled
    session_bf16 = ort.InferenceSession(quantized_model_path, sess_options=sess_options, providers=providers)
    print("session_bf16: ")
    print(f"  Graph Optimization Level: {sess_options.graph_optimization_level}")
    print(f"  Enable Profiling: {sess_options.enable_profiling}")
    print(f"  Intra-op Num Threads: {sess_options.intra_op_num_threads}")
    print(f"  Inter-op Num Threads: {sess_options.inter_op_num_threads}")
    print(f"  Log Severity Level: {sess_options.log_severity_level}")
    print(f"  Use Deterministic Compute: {sess_options.use_deterministic_compute}")
    return session_no_optim, session_optim, session_bf16

def validate_onnx_model(onnx_model_path):
    try:
        model = onnx.load(onnx_model_path)
        onnx.checker.check_model(model)  # Validates the model's structure
        print(f"ONNX model {onnx_model_path} is valid.")
    except onnx.onnx_cpp2py_export.checker.ValidationError as e:
        print(f"ONNX model validation failed for {onnx_model_path}: {e}")
        return False
    return True

def main():
    # File paths and model details
    case_path = r"/.../nnUNet_raw/Dataset400_FSUP_ULS/imagesTr/uls_00000_0000.nii.gz" # Test case
    onnx_model_path = "ULS23_test.onnx"
    quantized_model_path = "ULS23_test_quantized.onnx"
    num_runs = 10

    # Load the NIfTI file
    nii = nib.load(case_path).get_fdata()
    nifti_data = np.expand_dims(nii, axis=-1).astype(np.float32)
    nifti_data = np.transpose(nifti_data, (4, 3, 2, 1, 0))

    # Validate original ONNX model
    if not validate_onnx_model(onnx_model_path):
        print("ONNX model validation failed. Exiting.")
        return

    # Apply dynamic quantization and time it
    quantization_time = quantize_onnx_model(onnx_model_path, quantized_model_path)

    # Setup ONNX sessions
    session_no_optim, session_optim, session_bf16 = setup_onnx_sessions(onnx_model_path, quantized_model_path)

    # Benchmark ONNX models
    print("\nBenchmarking non-optimized model:")
    onnx_results_no_optim = benchmark_onnx_inference(session_no_optim, nifti_data, num_runs)

    print("\nBenchmarking optimized model:")
    onnx_results_optim = benchmark_onnx_inference(session_optim, nifti_data, num_runs)

    print("\nBenchmarking quantized+bf16 model:")
    onnx_results_quantized = benchmark_onnx_inference(session_bf16, nifti_data, num_runs)

    # Print summary
    print("\nBenchmark Summary:")
    print(f"Quantization Time: {quantization_time:.4f} s")
    print(f"ONNX No Optimization - Average: {onnx_results_no_optim['average']:.4f} s, Min: {onnx_results_no_optim['min']:.4f} s, Max: {onnx_results_no_optim['max']:.4f} s, Std: {onnx_results_no_optim['std']:.4f} s")
    print(f"ONNX with Graph Optimizations - Average: {onnx_results_optim['average']:.4f} s, Min: {onnx_results_optim['min']:.4f} s, Max: {onnx_results_optim['max']:.4f} s, Std: {onnx_results_optim['std']:.4f} s")
    print(f"Quantized ONNX Model - Average: {onnx_results_quantized['average']:.4f} s, Min: {onnx_results_quantized['min']:.4f} s, Max: {onnx_results_quantized['max']:.4f} s, Std: {onnx_results_quantized['std']:.4f} s")

if __name__ == "__main__":
    main()