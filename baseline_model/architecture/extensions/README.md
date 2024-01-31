## Extending nnUNetv2

These folders are copied into the nnunetv2 library when building the docker image, allowing for extensions to the nnunetv2 codebase. 

#### No resampling:
In order to train without resampling your data you can use the 'no_resampling_data_or_seg_to_shape' function in your plans file to be used with 'resampling_fn_data', 'resampling_fn_seg' and 'resampling_fn_probabilities'.
