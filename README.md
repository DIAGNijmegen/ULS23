![ULS23_banner.png](assets%2FULS23_banner.png)
### ULS23 Challenge Repository
Repository for the [Universal Lesion Segmentation Challenge '23](https://uls23.grand-challenge.org/datasets/)

### Labels
The annotations folder contains the labels for the training data of the ULS23 Challenge.

To download the associated imaging data, visit: 
- [Part 1](https://zenodo.org/records/10035161): Novel annotated data (ULS23_DeepLesion3D, ULS23_Radboudumc_Bone, ULS23_Radboudumc_Pancreas)
- [Part 2](https://zenodo.org/records/10050960): Processed data (kits21, LIDC-IDRI, LiTS)
- [Part 3](https://zenodo.org/records/10054306): Processed data (MDSC task 6/7/10, NIH-LN, CCC18)
- [Part 4](https://zenodo.org/records/10054702): Processed data (DeepLesion)
- [Part 5](https://zenodo.org/records/10055808): Processed data (DeepLesion)
- [Part 6](https://zenodo.org/records/10056235): Processed data (DeepLesion)

_Note: when using MONAI to work with the data please ensure you are on version >= 1.2.0. 
We have had reports of problems when loading the data using older versions._

#### Novel data annotation procedure:

`ULS23_DeepLesion3D`: Using reader studies on GrandChallenge, trained (bio-)medical students used the measurement information of the lesions in DeepLesion for 3D segmentation in the axial plane. 
Each lesion was segmented in triplicate and the majority mask was used as the final label. 
Lesions were selected using hard-negative mining with a standard 3D nnUnet trained on the fully annotated publicly available data.
We compared the axial diameters extracted from the prediction of this model to the reference measurements provided by DeepLesion and included the lesions with the worst performance.
We selected lesions to be included such that they were representative of the entire thorax-abdomen area.
This meant 200 abdominal lesions, 100 bone lesions, 50 kidney lesions, 50 liver lesions, 100 lung lesions, 100 mediastinal lesions and 150 assorted lesions.

`ULS23_Radboudumc_Bone` & `ULS23_Radboudumc_Pancreas`: VOI's in these datasets are from studies conducted at the Radboudumc hospital in Nijmegen, The Netherlands. 
Lesions were selected based on the radiological reports mentioning bone or pancreas disease. 
An experienced radiologist identified and then segmented the lesions in 3D. ULS23_Radboudumc_Bone contains both sclerotic & lytic bone lesions.

If you notice any problems with an images or mask, please launch an issue on the repo and we will try to correct it.

### Baseline Model
The `baseline_model` folder contains the minor code adaptations to the [nnUnetv2](https://github.com/MIC-DKFZ/nnUNet/tree/master) framework that are necessary to run the baseline model for the challenge.
Simply copy over the files in the nnunetv2 subfolder into your local nnunetv2 installation location.
To prevent resampling we have created a dummy resampling function 'no_resampling_data_or_seg_to_shape', which can be called in the plans files.
We also provide additional trainer classes to be able to train for more epochs and with a smaller starting learning rate. 
We used these when fine-tuning our baseline model pre-trained on the weakly annotated data.

Model weights and the algorithm container of the best performing baseline model for the weakly-annotated data are [stored on Zenodo](https://zenodo.org/doi/10.5281/zenodo.10665106). It can also be [used directly](https://grand-challenge.org/algorithms/universal-lesion-segmentation-uls23-baseline/) on novel data from GrandChallenge.

We also include the data split used for testing on the combined, fully-annotated training datasets and the plans files.

The `GC_algorithm` folder will contain the code for transforming the nnUnetv2 baseline into a GrandChallenge compatible algorithm.

### Data Processing Code

The `data_processing` folder contains the code used to prepare the source datasets into the ULS23 format: cropping VOI's around lesions and preparing the sem-supervised data using GrabCut.

### Contact Information
[max.degrauw@radboudumc.nl](mailto:max.degrauw@radboudumc.nl), [alessa.hering@radboudumc.nl](mailto:alessa.hering@radboudumc.nl) 

[Diagnostic Image Analysis Group,
Radboud University Medical Center,
Nijmegen, The Netherlands](https://www.diagnijmegen.nl/)

### References:
- Heller, N., Isensee, F., Trofimova, D., Tejpaul, R., Zhao, Z., Chen, H., ... & Weight, C. (2023). The KiTS21 Challenge: Automatic segmentation of kidneys, renal tumors, and renal cysts in corticomedullary-phase CT. arXiv preprint arXiv:2307.01984.
- Heller, N., Isensee, F., Maier-Hein, K. H., Hou, X., Xie, C., Li, F., ... & Weight, C. (2021). The state of the art in kidney and kidney tumor segmentation in contrast-enhanced CT imaging: Results of the KiTS19 challenge. Medical image analysis, 67, 101821. 
- Bilic, P., Christ, P., Li, H. B., Vorontsov, E., Ben-Cohen, A., Kaissis, G., ... & Menze, B. (2023). The liver tumor segmentation benchmark (lits). Medical Image Analysis, 84, 102680. 
- Roth, H., Lu, L., Seff, A., Cherry, K. M., Hoffman, J., Wang, S., Liu, J., Turkbey, E., & Summers, R. M. (2015). A new 2.5 D representation for lymph node detection in CT [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/K9/TCIA.2015.AQIIDCNM
- Pedrosa, J., Aresta, G., Ferreira, C., Atwal, G., Phoulady, H. A., Chen, X., ... & Campilho, A. (2021). LNDb challenge on automatic lung cancer patient management. Medical image analysis, 70, 102027. 
- Jacobs, C., van Rikxoort, E. M., Murphy, K., Prokop, M., Schaefer-Prokop, C. M., & van Ginneken, B. (2016). Computer-aided detection of pulmonary nodules: a comparative study using the public LIDC/IDRI database. European radiology, 26, 2139-2147. 
- Antonelli, M., Reinke, A., Bakas, S., Farahani, K., Kopp-Schneider, A., Landman, B. A., ... & Cardoso, M. J. (2022). The medical segmentation decathlon. Nature communications, 13(1), 4128. 
- Rother, C., Kolmogorov, V., & Blake, A. (2004). " GrabCut" interactive foreground extraction using iterated graph cuts. ACM transactions on graphics (TOG), 23(3), 309-314. 
- Yan, K., Wang, X., Lu, L., & Summers, R. M. (2018). DeepLesion: automated mining of large-scale lesion annotations and universal lesion detection with deep learning. Journal of medical imaging, 5(3), 036501-036501. 
- Urban, T., Ziegler, E., Pieper, S., Kirby, J., Rukas, D., Beardmore, B., Somarouthu, B., Ozkan, E., Lelis, G., Fevrier-Sullivan, B., Nandekar, S., Beers, A., Jaffe, C., Freymann, J., Clunie, D., Harris, G. J., & Kalpathy-Cramer, J. (2019). Crowds Cure Cancer: Crowdsourced data collected at the RSNA 2018 annual meeting [Data set]. The Cancer Imaging Archive. https://doi.org/10.7937/TCIA.2019.yk0gm1eb
- Isensee, F., Jaeger, P. F., Kohl, S. A., Petersen, J., & Maier-Hein, K. H. (2021). nnU-Net: a self-configuring 
method for deep learning-based biomedical image segmentation. Nature methods, 18(2), 203-211.
