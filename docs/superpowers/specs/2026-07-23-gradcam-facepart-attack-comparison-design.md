# Grad-CAM++ vs FacePart Attack-Mask Comparison

## Goal

Measure whether the paper's Grad-CAM++ mask is better than the FacePart
semantic mask when both masks constrain the same targeted pixel-space PGD
attack on the same generated image.

## Controlled Variables

- Task: remove `Smiling` using CelebA attribute index 31.
- Inputs: five existing A9 outputs, before final target correction.
- Classifier: `models/resnet50_multilabel_model.pth`.
- Attack start: the generated A9 image, never the original image.
- Attack: targeted BCE PGD with the same epsilon, step size, iteration limit,
  projection, and early-stop rule for both masks.
- FacePart mask: hard union of mouth, upper lip, and lower lip.
- Saliency mask: Grad-CAM++ from the original image at ResNet-50
  `layer4[-1]`, thresholded at 0.4 as specified by the paper.

## Measurements

Record source and generated classifier probabilities, attacked probability,
target pass, mask area, changed-pixel area, perturbation L1/L2/L-infinity,
change outside each mask, change outside the FacePart union, and identity
cosine when the existing identity evaluator is available.

## Outputs

The experiment writes one folder per sample containing the common input,
both masks, both attacked outputs, and a comparison sheet. It also writes
per-image CSV/JSON results and an aggregate summary. Results must not be
described as a general superiority claim from five images.

