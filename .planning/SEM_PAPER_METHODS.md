# SEM Paper Method Deep Dive

Checked: 2026-07-09

## 1. SAM-I-Am

Paper: https://doi.org/10.1016/j.commatsci.2024.113400  
Preprint: https://arxiv.org/abs/2404.06638

**Task:** zero-shot material-region segmentation in atomic-resolution STEM/TEM.

**Pipeline:**

1. Run SAM Automatic Mask Generator with a 16 x 16 prompt grid.
2. Remove geometrically invalid masks: tiny children, islands, compound
   parents, and duplicates. A child below 5% of its parent is removed; a
   parent whose children cover over 70% is removed.
3. Draw 60 x 60 texture crops from each surviving mask.
4. Embed crops with FENet using a ResNet-18 backbone fine-tuned on the DTD
   texture dataset, with the last two fully connected layers removed.
5. Cluster embeddings with KMeans using the known number of materials.
6. Assign each mask by majority vote and merge masks in the same cluster.

**Evaluation data:** 82 expert-labeled ADF/HAADF STEM images, mostly
1024 x 1024, containing 2-4 materials. The data were created by the authors
with SAM-assisted Label Studio annotation and were not verified as a public
download.

**Fit for this project:** useful as a post-processing design after SAM, but
not a ready electrode-particle model. It assumes atomic lattice textures,
cross-sectional TEM regions, and a known number of materials.

## 2. Segment Anything for Microscopy (micro-sam)

Paper: https://doi.org/10.1038/s41592-024-02580-4  
Code and models: https://github.com/computational-cell-analytics/micro-sam

**Task:** interactive and automatic instance segmentation for microscopy.

**Pipeline:**

1. Start from pretrained SAM and fine-tune the image encoder, prompt encoder,
   and mask decoder. Full-model fine-tuning performed best.
2. Simulate iterative prompts from ground truth for interactive training.
3. Train an additional UNETR-style automatic instance segmentation decoder.
4. Predict foreground probability, distance to object centers, and distance
   to object boundaries.
5. Convert these three maps to instances with seeded watershed.

**Training details:** 512 x 512 patches, batch size 2 (or 1 under constrained
hardware), Dice loss for masks, L2 loss for predicted IoU, Adam at 1e-5.
ViT-B or ViT-L is the recommended quality/runtime trade-off.

**Important limitation:** the EM generalist was trained mainly on
mitochondria and nuclei using MitoLab, MitoEM, and PlatyEM. The authors state
that it is reliable mainly for these and similarly round organelles; other
morphologies need a specialist model.

**Fit for this project:** best available annotation and fine-tuning framework.
For electrode SEM, use it to create masks quickly, then fine-tune and validate
an electrode-specific specialist. Running the biological EM checkpoint on an
electrode image is real inference, but its masks are not scientifically
validated electrode measurements.

## 3. Furat et al. 2022

Paper: https://doi.org/10.1038/s41524-022-00749-z

**Task:** 2.5x super-resolution of NMC532 SEM images before crack
segmentation.

**Pipeline:**

1. Register experimentally measured low- and high-resolution SEM images with
   OpenCV template matching.
2. Normalize grayscale intensities and form 33 matched image pairs.
3. Train a modified SRGAN on 24 pairs; use 5 for validation and 4 for test.
4. The generator is SRResNet with 16 residual blocks, ReLU instead of PReLU,
   no batch normalization, one PixelShuffle stage, and one-channel sigmoid
   output.
5. Train on batches of 32 paired 96 x 96 low-resolution and 240 x 240
   high-resolution crops with Adam at 1e-4, adversarial weight 2.0, and a
   VGG19 perceptual loss.
6. Segment cracks from the super-resolved images and compare against
   high-resolution ground truth.

**Result relevant here:** crack Jaccard improved from 0.556 with bilinear
upsampling to 0.679 with SRGAN; specific crack-density relative error fell
from 0.136 to 0.036.

**Reproducibility:** data are available only on request and no trained weights
or official implementation were published.

**Fit for this project:** this is an optional image-restoration stage, not a
SEM-to-performance predictor. It should not be reproduced without paired
low/high-resolution images from the same microscope protocol.

## 4. Badmos et al. 2020

Paper: https://doi.org/10.1007/s10845-019-01484-x

**Task:** detect microstructural defects in sectioned lithium-ion electrodes.

**Verified pipeline:** split large light-optical micrographs into patches,
classify patches with transfer-learned CNN features, and project patch
predictions back as a class heatmap for defect localization.

**Important limitation:** this is light microscopy, not SEM, and it predicts
quality defects rather than particle morphology or electrochemical
performance. A public image archive, code, and checkpoint were not verified.
The paywalled full method prevented independent verification of architecture
and training hyperparameters, so those details should not be inferred.

**Fit for this project:** reuse only the patch-to-heatmap user flow. It is not
a reproducible model source for the current demo.

## 5. Deeg et al. 2024

Paper: https://doi.org/10.3390/batteries10030099

**Task:** regress rate-dependent capacity directly from electrode
cross-section images.

**Data:** 314 light-microscopy cross-sections of NCM111 electrodes at
20%, 25%, 30%, 35%, and 50% porosity. Images were center-cropped to
224 x 672. Targets were capacity at 0.2C, 1C, 2C, 3C, and 5C.

**Model:**

1. Three VGG-like blocks, each with two 3 x 3 convolution layers; filter
   count rises from 32 to 96.
2. ReLU, L2 regularization, and batch normalization in convolution blocks.
3. Max pooling after the first two blocks and global average pooling after
   the third, yielding 128 image features.
4. Four dense layers of 64 units with leaky ReLU, batch normalization, L2,
   and dropout.
5. A linear five-neuron output layer, one capacity prediction per C-rate.
   The network has about 250,000 trainable parameters.

Two separate models predict gravimetric and volumetric capacity. Training
uses an 80/10/10 split, five-fold cross-validation, MAPE loss, Lookahead with
RAdam, a maximum of 5,000 steps, warm-up, and early stopping. Grad-RAM and
guided backpropagation provide explanations.

**Reproducibility:** data are available from the authors on request; no
public checkpoint was verified.

**Fit for this project:** this is the closest direct image-to-performance
method, but it cannot be applied to surface SEM images. Reproduction requires
paired cross-section images and electrochemical capacity labels acquired
under a consistent protocol.

## 6. Oh et al. 2024

Paper: https://doi.org/10.1038/s41524-024-01279-6  
Code/data: https://github.com/MIIMSEKAIST/CNN_for_NCM-composition-and-state-prediction

**Task:** classify four NCM compositions and three cycling states, producing
12 joint classes.

**Pipeline:**

1. Acquire 1,637 SEM images at 500x from NCM333, 523, 622, and 811 in
   pristine, formation, and 100-cycle states.
2. Generate class-balanced random crops; the best setup uses approximately
   300 x 300 crops resized to 224 x 224.
3. Fine-tune ImageNet-pretrained EfficientNet-B7 with 12 output classes.
4. Use cross-entropy and Adam; the released configuration uses learning rate
   3.5447e-4, batch size 4, and 40 epochs.
5. Do not normalize brightness in the best model because intensity
   distributions contained state information.
6. Explain predictions with guided Grad-CAM, which focuses on particle
   boundaries and gaps.

The paper reports 99.6% in-domain accuracy, but on unseen additive conditions
composition remained 96.0% while state accuracy fell to 34.17%, demonstrating
strong domain shift.

**Repository audit:** code is present, but no `.pth`, `.pt`, `.ckpt`, or ONNX
checkpoint is committed. The current `Dataset` directory contains 51 images
from only `NCM622_cycled`, not the complete 12-class training set. Therefore,
the published classifier cannot be honestly run or retrained from the
repository alone.

**Fit for this project:** the 51 public SEM images are valid demo inputs. They
cannot support a 12-class model or a capacity/retention claim.

## Implementation Decision

The papers support two distinct web flows:

1. **Available now: SEM to morphology.** Run an actual segmentation model,
   show its mask, then compute area fraction, connected-component size,
   equivalent diameter, circularity, and crack-like boundary statistics from
   that mask. Label results as model-derived measurements and expose the
   model/checkpoint identity.
2. **Requires new paired data: SEM to performance.** Train a Deeg-style
   multi-output regressor only after collecting images paired with capacity
   or retention. Until then, no capacity or retention number should be
   presented as inferred from the uploaded image.

For the first flow, micro-sam is the strongest framework, but an
electrode-specific specialist checkpoint is required for defensible
measurements. The practical sequence is:

1. Use micro-sam interactively to annotate representative electrode SEM
   images.
2. Split by original image or specimen before generating crops to prevent
   leakage.
3. Fine-tune a ViT-B specialist and validate masks with IoU, boundary IoU,
   precision, and recall.
4. Freeze and version the checkpoint.
5. Deploy inference in Streamlit and derive morphology only from the returned
   mask.

