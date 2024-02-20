import os
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ["OMP_NUM_THREADS"] = "1"
import SimpleITK as sitk
sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(1)
import json
import statistics
import numpy as np
from pathlib import Path
from misc import load_predictions_json, long_and_short_axis_diameters, dice_coefficient, create_scores_dict, align_images, sape

class ULS23_evaluator():
    def __init__(self):
        os.makedirs("/output", exist_ok=True)
        self._ground_truth_path = Path("/opt/app/ground-truth/")
        self._predictions_path = Path("/input/")
        self._output_file = Path("/output/metrics.json")

        self.stack_size = 100
        self.z = 128
        self.consistency_samples = 3

    def run(self):
        print("Loading ground-truth")
        with open(self._ground_truth_path / "single_sample.json", 'r') as json_file:
            self.ss_cases = json.load(json_file)

        with open(self._ground_truth_path / "multi_sample.json", 'r') as json_file:
            self.ms_cases = json.load(json_file)

        with open(self._ground_truth_path /"stacked_spacing.json", 'r') as json_file:
            self.spacings = json.load(json_file)

        self.mapping_dict = load_predictions_json(self._predictions_path / "predictions.json")

        print("Starting evaluation")
        for job in self.mapping_dict.keys():
            print(f"Evaluating job: {job}")
            self._input_path = self._predictions_path / job / "output/images/ct-binary-uls"
            self.input_dict = self.mapping_dict[job]
            self.evaluate_stacks()

    def evaluate_stacks(self):
        scores = create_scores_dict()

        for file_id, (pred_file, gt_file) in enumerate(self.input_dict.items()):
            gt_path = self._ground_truth_path / gt_file
            pred_path = self._input_path / pred_file

            # Load the images for this case
            gt = sitk.ReadImage(gt_path)
            pred = sitk.ReadImage(pred_path)

            # Get array
            gt = sitk.GetArrayFromImage(gt)
            pred = sitk.GetArrayFromImage(pred)

            # Unstack VOI's and evaluate
            for idx, i in enumerate(self.ss_cases):
                gt_voi = gt[self.z*i:self.z*(i+1), :, :]
                pred_voi = pred[self.z*i:self.z*(i+1), :, :]

                gt_long, gt_short, _, _ = long_and_short_axis_diameters(gt_voi)
                pred_long, pred_short, _, _ = long_and_short_axis_diameters(pred_voi)

                scores["case"]["SegmentationDice"][idx] = dice_coefficient(gt_voi, pred_voi)

                scores["case"]["LongAxisErrorPercentage"][idx] = sape(gt_long, pred_long)
                scores["case"]["ShortAxisErrorPercentage"][idx] = sape(gt_short, pred_short)
                    
                scores["case"]["fn"][idx] = f"file_{file_id}_lesion_{idx}"

                print(f"VOI {idx}, DICE: {scores['case']['SegmentationDice'][idx]}")

            # Calculate segmentation consistency for the N samples of this lesions VOI
            for idx, case in enumerate(self.ms_cases):
                average_consistency_dice = 0
                for comparison in case:
                    a = pred[128 * comparison[0]:128 * (comparison[0] + 1), :, :]
                    b = pred[128 * comparison[1]:128 * (comparison[1] + 1), :, :]
                    # Check if there are predictions to compare
                    if np.amax(a) > 0 and np.amax(b) > 0:
                        # Align prediction b to prediction a
                        b_aligned = align_images(b.T, self.spacings[comparison[0]], comparison[2][0], comparison[2][1],
                                                 comparison[2][2]).T
                        average_consistency_dice += dice_coefficient(a, b_aligned)
                    else:
                        average_consistency_dice += 0

                scores["consistency_check"]["ConsistencyDice"][idx] = average_consistency_dice / len(case)

        for metric in ["SegmentationDice", "LongAxisErrorPercentage", "ShortAxisErrorPercentage"]:
            values = scores["case"][metric].values()
            scores["aggregates"][metric]["min"] = min(values)
            scores["aggregates"][metric]["max"] = max(values)
            scores["aggregates"][metric]["25pc"] = statistics.quantiles(values)[0]
            scores["aggregates"][metric]["median"] = statistics.quantiles(values)[1]
            scores["aggregates"][metric]["75pc"] = statistics.quantiles(values)[2]
            scores["aggregates"][metric]["mean"] = statistics.mean(values)
            scores["aggregates"][metric]["std"] = statistics.stdev(values)
            scores["aggregates"][metric]["count"] = len(values)

        scores["aggregates"]["ConsistencyDice"]["min"] = min(scores["consistency_check"]["ConsistencyDice"].values())
        scores["aggregates"]["ConsistencyDice"]["max"] = max(scores["consistency_check"]["ConsistencyDice"].values())
        scores["aggregates"]["ConsistencyDice"]["25pc"] = statistics.quantiles(scores["consistency_check"]["ConsistencyDice"].values())[0]
        scores["aggregates"]["ConsistencyDice"]["median"] = statistics.quantiles(scores["consistency_check"]["ConsistencyDice"].values())[1]
        scores["aggregates"]["ConsistencyDice"]["75pc"] = statistics.quantiles(scores["consistency_check"]["ConsistencyDice"].values())[2]
        scores["aggregates"]["ConsistencyDice"]["mean"] = statistics.mean(scores["consistency_check"]["ConsistencyDice"].values())
        scores["aggregates"]["ConsistencyDice"]["std"] = statistics.stdev(scores["consistency_check"]["ConsistencyDice"].values())
        scores["aggregates"]["ConsistencyDice"]["count"] = len(scores["consistency_check"]["ConsistencyDice"].values())
        scores["aggregates"]["ConsistencyDice"]["samples"] = self.consistency_samples

        scores["aggregates"]["ChallengeScore"] = (0.8 * scores["aggregates"]["SegmentationDice"]["mean"]
                                    + 0.05 * (1 - min(1, scores["aggregates"]["LongAxisErrorPercentage"]["mean"]))
                                    + 0.05 * (1 - min(1, scores["aggregates"]["ShortAxisErrorPercentage"]["mean"]))
                                    + 0.1 * scores["aggregates"]["ConsistencyDice"]["mean"])

        with open(str(self._output_file), "w") as f:
            json.dump(scores, f)

if __name__ == "__main__":
    ULS23_evaluator().run()