No, we are not going to give you the ground truth for the challenge here :)

However, if you want to test the evaluation container, this folder will need to contain:
- `single_sample.json` a list with the indices of the unique lesion VOI's in the stack
- `multi_sample.json` a nested list with for each unique lesion all possible pairs of comparisons to the different samples and how to translate from the second sample in the pair back to the first. We provide an example file: [multi_sample.json](multi_sample.json).
- `stacked_spacing.json` a list containing the xyz spacing for each stacked VOI
- `X.mha` the stack of ground truth segmentations for the image file of the job. This file should have the same name (not identifier) as the image file from the algorithm job.