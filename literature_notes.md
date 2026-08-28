# Literature Notes

These notes cover the non-LLM papers in the source plan plus the microscopy
papers investigated for the SEM demo. They justify design choices; they do not
turn the synthetic MVP into scientific validation.

## Recent progress in Si/Ti3C2Tx MXene anode materials for lithium-ion batteries

Link: https://doi.org/10.1016/j.isci.2024.111217  
Year: 2024  
Material/system: Silicon/Ti3C2Tx MXene composite anodes.  
Main method: Review of composite architectures, synthesis routes, and lithium-storage behavior.  
Data used: Published electrochemical and structural results from Si/MXene studies.  
Target/output: Capacity, cycling retention, rate capability, and structural stability.  
What we can reuse: Si/MXene ratio, composite route, morphology, capacity, and retention as schema fields.  
How it supports our pipeline: Si supplies high capacity while conductive MXene networks can improve transport and accommodate structural change.  
Difference from our project: The review synthesizes prior materials evidence; our code ranks lab recipes from structured process and characterization data.

## Application and Development of Silicon Anode Binders for Lithium-Ion Batteries

Link: https://doi.org/10.3390/ma16124266  
Year: 2023  
Material/system: Polymer binders for silicon and silicon-based anodes.  
Main method: Review of binder chemistry, mechanical behavior, and electrochemical performance.  
Data used: Published binder formulations and cycling measurements.  
Target/output: Electrode integrity, capacity retention, Coulombic efficiency, and cycle life.  
What we can reuse: Binder type/content, cross-linking strategy, capacity, retention, and efficiency fields.  
How it supports our pipeline: Sodium alginate has distributed carboxyl groups and can form strong interactions or cross-linked networks that buffer silicon expansion.  
Difference from our project: The paper compares binder chemistry; our MVP currently varies alginate content without molecular descriptors.

## Rational design of alginate-derived network binder for high-performance silicon-based anodes

Link: https://doi.org/10.1016/j.jpowsour.2024.235745  
Year: 2025  
Material/system: SAH alginate-derived network binder for silicon anodes.  
Main method: One-pot polymerization and in-situ cross-linking of sodium alginate with acrylic monomers.  
Data used: Binder characterization and electrochemical cycling of silicon electrodes.  
Target/output: Structural integrity, SEI stability, and capacity retention.  
What we can reuse: Binder formulation, cross-link density proxy, processing conditions, and retention.  
How it supports our pipeline: A rigid/flexible three-dimensional binder network can buffer silicon volume change; the paper reports 87.3% retention after 100 cycles at 0.5C.  
Difference from our project: Our synthetic schema models alginate amount only and does not represent SAH chemistry.

## Carbon Additive-Free Crumpled Ti3C2Tx MXene-Encapsulated Silicon Nanoparticle Anodes

Link: https://doi.org/10.1021/acsaem.1c01736  
Year: 2021  
Material/system: Crumpled Ti3C2Tx MXene sheets encapsulating silicon nanoparticles.  
Main method: One-step spray drying with varied MXene/silicon composition.  
Data used: Morphology, composition, cycling, rate, and Coulombic-efficiency measurements.  
Target/output: Stable conductive composite architecture and cycling capacity.  
What we can reuse: Si/MXene fraction, carbon content, spray-drying conditions, morphology, and cycling outputs.  
How it supports our pipeline: MXene surface groups and the crumpled conductive framework help preserve contact around expanding silicon.  
Difference from our project: The reported best architecture is carbon-free and spray-dried; our search grid is a generic slurry-electrode demo.

## Image-Guided Microstructure Optimization using Diffusion Models

Link: https://arxiv.org/abs/2505.07906  
Year: 2025  
Material/system: Li- and Mn-rich layered oxide cathode precursors.  
Main method: Diffusion image model, quantitative SEM analysis, and particle swarm optimization.  
Data used: SEM images paired with coprecipitation time, concentration, and pH.  
Target/output: User-defined morphology represented by texture, sphericity, and D50.  
What we can reuse: Image-derived morphology descriptors and a forward/inverse process optimization pattern.  
How it supports our pipeline: It experimentally demonstrates image-guided selection of synthesis conditions.  
Difference from our project: Choi et al. optimize morphology itself; we use SEM/EDX as intermediate features and optimize electrochemical retention.

## Deep learning-based segmentation of lithium-ion battery microstructures

Link: https://doi.org/10.1038/s41467-021-26480-9  
Year: 2021  
Material/system: Graphite-silicon composite negative electrodes imaged by X-ray tomography.  
Main method: 3D U-Net trained with real and artificially generated electrode volumes.  
Data used: XTM/PXCT volumes and labels for pore, graphite, silicon, and carbon-black/binder phases.  
Target/output: Four-phase segmentation and quantitative microstructure evolution.  
What we can reuse: Porosity, phase fraction, particle, and binder-domain descriptors after segmentation.  
How it supports our pipeline: It shows that segmented battery microstructure can produce model-ready quantitative features and that synthetic training data can help.  
Difference from our project: Our current prototype uses lightweight 2D threshold segmentation; 3D U-Net is a later upgrade for labeled volumetric data.

## Super-resolving microscopy images of Li-ion electrodes for fine-feature quantification

Link: https://doi.org/10.1038/s41524-022-00749-z  
Year: 2022  
Material/system: SEM images of cracked, aged NMC cathode particles.  
Main method: Super-resolution GANs trained on measured low/high-resolution SEM pairs.  
Data used: Thirty-three registered SEM pairs, split into training, validation, and test sets.  
Target/output: Higher-resolution images and improved crack segmentation.  
What we can reuse: Crack-density and fine-feature measurements, with image-resolution provenance.  
How it supports our pipeline: Super-resolution can improve quantification when field of view and fine detail trade off.  
Difference from our project: The source data are available only on reasonable request, so no GAN is trained in this local demo.

## A comprehensive and quantitative SEM-EDS analytical process applied to lithium-ion battery electrodes

Link: https://doi.org/10.1038/s41598-025-89362-w  
Year: 2025  
Material/system: Graphite electrodes from commercial cells at three degradation levels.  
Main method: Automated SEM capture, stitching, aligned EDS maps, texture features, clustering, and distribution similarity.  
Data used: 30 samples, 3,000 SEM images, and 436,800 aligned SEM/EDS patch pairs.  
Target/output: Quantified morphology and F/O/C elemental-distribution changes with degradation.  
What we can reuse: Aligned morphology/composition records, elemental percentages, texture, and distribution-similarity features.  
How it supports our pipeline: SEM and EDS can be integrated quantitatively rather than treated as disconnected qualitative figures.  
Difference from our project: Our CSV uses Si/Ti/C/O/impurity proxies and does not claim reliable lithium measurement by EDX.

## On-the-fly closed-loop materials discovery via Bayesian active learning

Link: https://doi.org/10.1038/s41467-020-19597-w  
Year: 2020  
Material/system: Ge-Sb-Te composition spread for phase and optical-property discovery.  
Main method: CAMEO closed-loop Bayesian active learning with phase-map knowledge and Gaussian-process regression.  
Data used: Iterative XRD, ellipsometry, theory, and expert-derived property information.  
Target/output: Next measurement choice and discovery of a property optimum.  
What we can reuse: Surrogate mean/uncertainty, acquisition-driven experiment selection, and human guidance.  
How it supports our pipeline: It establishes the pattern of updating knowledge after each measured experiment and selecting the next informative candidate.  
Difference from our project: Our GP/UCB implementation ranks a fixed recipe grid and does not yet control instruments or retrain after a real experiment.

## Toward High-Performance Energy and Power Battery Cells with ML-based Optimization

Link: https://arxiv.org/abs/2307.05521  
Year: 2023  
Material/system: Simulated NMC111 electrode manufacturing and cell performance.  
Main method: Physics-based manufacturing/electrochemical simulation, deterministic ML surrogates, and bi-objective Bayesian optimization.  
Data used: Synthetic process, microstructure, and pseudo-two-dimensional electrochemical simulation outputs.  
Target/output: Inverse-designed active-material fraction, slurry solid content, and calendering degree for energy/power objectives.  
What we can reuse: Process-to-performance surrogate modeling and inverse recipe selection.  
How it supports our pipeline: It demonstrates that manufacturing parameters can be optimized against electrochemical objectives through ML surrogates.  
Difference from our project: Their workflow is simulation-rich and multi-objective; ours is a small experiment-facing Si/MXene demonstration.

## ML-assisted multi-objective optimization of battery manufacturing from synthetic data

Link: https://doi.org/10.1016/j.ensm.2022.12.040  
Year: 2023  
Material/system: NMC111 electrode slurry, drying, and calendering process.  
Main method: Physics-based simulations, low-discrepancy sampling, ML surrogates, and multi-objective Bayesian optimization.  
Data used: Synthetic manufacturing-process and electrode-property records with experimental validation of the selected condition.  
Target/output: Low tortuosity with high conductivity, active surface area, and density.  
What we can reuse: Structured synthetic data for pipeline development, constrained search, and later multi-objective acquisition.  
How it supports our pipeline: It shows that synthetic data can be a legitimate engineering scaffold when its provenance and validation limits are explicit.  
Difference from our project: Our synthetic data are illustrative rather than generated by calibrated physics simulations.

## Composition and state prediction of lithium-ion cathode from SEM images

Link: https://doi.org/10.1038/s41524-024-01279-6  
Year: 2024  
Material/system: NCM333/523/622/811 cathodes in pristine, formation, and cycled states.  
Main method: EfficientNet-based CNN classification with cropped SEM images and Grad-CAM.  
Data used: 1,637 original SEM images before augmentation. The current public repository contains code and 51 NCM622-cycled images, but no trained checkpoint or complete 12-class dataset.  
Target/output: NCM composition and electrochemical-state class.  
What we can reuse: Public battery SEM images, class labels, and image-ingestion conventions.  
How it supports our pipeline: It confirms that SEM morphology carries composition/state signal usable by machine-learning workflows.  
Difference from our project: We use three NCM622 images only to exercise feature extraction; the published classifier cannot be reproduced from the currently released subset.

## Swift Prediction of Battery Performance from Microstructural Electrode Images

Link: https://doi.org/10.3390/batteries10030099  
Year: 2024  
Material/system: NCM111 electrodes with five porosities.  
Main method: A roughly 250k-parameter, VGG-like multi-output CNN with Grad-RAM and guided-backpropagation explanations.  
Data used: 314 light-microscopy cross-section images paired with capacities at 0.2C, 1C, 2C, 3C, and 5C.  
Target/output: Rate-dependent specific capacity.  
What we can reuse: The direct image-to-performance modeling pattern and explainability checks.  
How it supports our pipeline: It demonstrates that microstructure images can predict rate capability rather than only morphology classes.  
Difference from our project: It uses light-microscopy cross-sections, not surface SEM; its paired dataset is available on request, so its capacity model cannot be applied to our current images.

## Image-based defect detection in lithium-ion battery electrodes

Link: https://doi.org/10.1007/s10845-019-01484-x  
Year: 2020  
Material/system: Sectioned lithium-ion battery electrode micrographs.  
Main method: Patch-based transfer-learning CNN classification and class-heatmap localization.  
Data used: Optical micrographs labeled with deformation, contamination, non-uniform coating, and preparation artifacts.  
Target/output: Defect/no-defect classification and localized quality findings.  
What we can reuse: Defect labels and patch-based image quality-control workflow.  
How it supports our pipeline: It shows that electrode micrographs can support automated quality features at scale.  
Difference from our project: This is light microscopy rather than SEM. No public image archive, code, or checkpoint was verified, and the paywalled full method prevents independent verification of its exact architecture.

## Segment Anything for Microscopy

Link: https://doi.org/10.1038/s41592-024-02580-4  
Year: 2025  
Material/system: Diverse light- and electron-microscopy datasets.  
Main method: Full SAM fine-tuning plus a UNETR-style decoder that predicts foreground, center distance, and boundary distance for seeded-watershed instance segmentation.  
Data used: Multiple public LM and biological EM segmentation benchmarks.  
Target/output: Interactive and automatic object masks, tracking, and annotation.  
What we can reuse: Human-correctable masks and fine-tuning once representative labeled microscopy data exist.  
How it supports our pipeline: It provides a practical route from manual SEM annotation to reusable segmentation models.  
Difference from our project: Its generalist EM model was trained mainly for mitochondria and nuclei; the authors explicitly recommend specialist fine-tuning for other morphologies such as battery particles.

## SAM-I-Am: Semantic boosting for atomic-scale electron micrograph segmentation

Link: https://doi.org/10.1016/j.commatsci.2024.113400  
Year: 2025  
Material/system: Atomic-scale transmission electron micrographs.  
Main method: SAM masks are geometrically filtered, 60 x 60 mask crops are embedded by DTD-trained FENet/ResNet-18, then KMeans and majority voting merge masks with similar textures.  
Data used: 82 manually labeled ADF/HAADF STEM images with 2-4 materials; sharing restrictions are discussed.  
Target/output: Semantically coherent material-region masks without fine-tuning.  
What we can reuse: Lightweight post-processing of foundation-model masks when labels are scarce.  
How it supports our pipeline: It motivates combining generic segmentation with domain-specific morphology rules.  
Difference from our project: The cited code URL currently returns 404 and the validation data were not verified as publicly downloadable.
