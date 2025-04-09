# NextFace V2 With Batch Automated Setup
## Modified and Updated by: Risk

NextFace is a light-weight PyTorch library for high-fidelity 3D face reconstruction from monocular image(s) where scene attributes – 3D geometry, reflectance (diffuse, specular and roughness), pose, camera parameters, and scene illumination – are estimated. It is a first-order optimization method that uses the PyTorch autograd engine and ray tracing to fit a statistical morphable model to input image(s).

<p align="center">
  <img src="resources/emily.png" style="float: left; width: 23%; margin-right: 1%; margin-bottom: 0.5em;">
  <img src="resources/emily.gif" style="float: left; margin-right: 1%; margin-bottom: 0.5em;">
  <img src="resources/beard.png" style="float: left; width: 23%; margin-right: 1%; margin-bottom: 0.5em;">
  <img src="resources/beard.gif" style="float: left; margin-right: 1%; margin-bottom: 0.5em;">
  <img src="resources/visual.jpg">
</p>

<p align="center">
  <strong>A demo on YouTube from here:</strong>
</p>
<p align="center">
  <a href="http://www.youtube.com/watch?v=bPFp0oZ9plg" title="Practical Face Reconstruction via Differentiable Ray Tracing">
    <img src="http://img.youtube.com/vi/bPFp0oZ9plg/0.jpg" alt="Practical Face Reconstruction via Differentiable Ray Tracing" />
  </a>
</p>

## Table of Contents
- [News](#news)
- [Features](#features)
- [Installation](#-installation)
- [How to Use](#how-to-use)
- [Directory Structure](#directory-structure)
- [How it Works](#how-it-works)
- [Good Practice for Best Reconstruction](#good-practice-for-best-reconstruction)
- [Limitations](#limitations)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)
- [Citation](#citation)

# News
- **2025-04-08**: Fixed the MediaPipe installer and added a full app installation/setup batch.
- **19 March 2023**: Fixed a bug in the optimizer where the gradients were not activated for the camera position (rotation and translation). Also added a new optimization strategy for stages two and three which should improve overall performance. Please pull the latest update.
- **21 June 2022**: Many thanks to [Jack Saunders](https://researchportal.bath.ac.uk/en/persons/jack-saunders) for adding a new feature to NextFace: support for [MediaPipe](https://google.github.io/mediapipe/solutions/face_mesh.html#overview) as a replacement for the FAN landmarks detector. MediaPipe produces much more stable and accurate results than FAN. To try MediaPipe, pull the new version of the code and install it using:
  ```
  py -m pip install --user mediapipe
  ```
  By default, the landmarks detector is set to MediaPipe. To switch back to FAN, edit the **optimConfig.ini** file (set `landmarksDetectorType = 'fan'`).
- **01 May 2022**: To generate an animation (like the GIFs in the README) that rotates the reconstruction on the vertical axis, run the `replay.py` script with the path of the pickle file containing the optimized scene attributes (located in `checkpoints/stage3_output.pickle`).
- **26 April 2022**: Added export of the estimated light map (as an environment map). This is useful for rendering the face with other engines (Unreal, Unity, OpenGL). Pull the latest code. You can export the light map as PNG or EXR (see **optimConfig.ini**).
- **25 April 2022**: For high-resolution textures (1024x1024 or 2048x2048), download **uvParametrization.2048.pickle** and **uvParametrization.1024.pickle** from [here](https://github.com/abdallahdib/NextFace/releases). Place them in the `baselMorphableModel` directory and change `textureResolution` in **optimConfig.ini** accordingly. Note: larger UV maps require more CPU/GPU memory.
- **24 April 2022**: A Colab notebook is now available in `demo.ipynb`.
- **20 April 2022**: Updated landmarks association file for better reconstruction, especially along face contours. Please pull the latest changes.
- **20 April 2022**: Demonstrated NextFace on a challenging face with appealing reconstruction results:
  <p align="center">
    <img src="resources/results1.gif" style="float: left; width: 50%; margin-right: 1%; margin-bottom: 0.5em;">
  </p>

# Features
- Reconstructs face at high fidelity from single or multiple RGB images.
- Estimates face geometry.
- Estimates detailed face reflectance (diffuse, specular, and roughness).
- Estimates scene light with spherical harmonics.
- Estimates head pose and orientation.
- Runs on both CPU and CUDA-enabled GPU.

# 🔧 Installation
* Clone the repository.
* Run the included `setup_faceNext_env.bat` script found within the Main FaceNext Folder (right-click → **Run as Administrator**).  
  This script will:
  - Delete any previous `faceNext` environment.
  - Create a new conda environment with Python 3.9.
  - Install required packages including PyTorch 1.10.2, torchvision 0.11.3, CUDA 10.2, and others.
  - Install **MediaPipe** using the only working method on Windows:
    ```
    py -m pip install --user mediapipe
    ```
* Alternatively, use the provided `environment.yml` for manual setup (note that some packages like MediaPipe must still be installed separately).
* Activate the environment:
  ```bash
  conda activate faceNext
  ```
* Download the Basel Face Model from [here](https://faces.dmi.unibas.ch/bfm/bfm2017.html).  
  Fill out the form to receive a download link via email, then place the `model2017-1_face12_nomouth.h5` file into the `./baselMorphableModel` directory.
* Download the Albedo Face Model from [here](https://github.com/waps101/AlbedoMM/releases/download/v1.0/albedoModel2020_face12_albedoPart.h5).  
  Place the `albedoModel2020_face12_albedoPart.h5` file into the `./baselMorphableModel` directory.

# How to Use

## Reconstruction from a Single Image
- To reconstruct a face from a single image, run:
  ```
  python optimizer.py --input <path-to-your-input-image> --output <output-path-where-to-save-results>
  ```

## Reconstruction from Multiple Images (Batch Reconstruction)
- If you have multiple images with the same resolution, place all your images in a single directory and run:
  ```
  python optimizer.py --input <path-to-your-folder-containing-images> --output <output-path-where-to-save-results>
  ```

## Reconstruction from Multiple Images for the Same Person
- If you have multiple images for the same person, place them in a single folder and run:
  ```
  python optimizer.py --sharedIdentity --input <path-to-your-folder-containing-images> --output <output-path-where-to-save-results>
  ```
  The `--sharedIdentity` flag tells the optimizer that all images belong to the same person. In this case, the shape identity and face reflectance attributes are shared across all images, generally producing better estimation.

## Configuring NextFace
- The **optimConfig.ini** file allows you to control various aspects of NextFace such as:
  - Optimization parameters (regularizations, number of iterations, etc.)
  - Compute device (CPU or GPU)
  - Spherical harmonics (number of bands, environment map resolution)
  - Ray tracing (number of samples)
- The code is self-documented and easy to follow.

# Directory Structure
```
NextFace/
├── baselMorphableModel/
│   ├── model2017-1_face12_nomouth.h5
│   ├── albedoModel2020_face12_albedoPart.h5
│   └── (other model files)
├── checkpoints/
│   └── stage3_output.pickle
├── resources/
│   ├── emily.png
│   ├── emily.gif
│   ├── beard.png
│   ├── beard.gif
│   ├── visual.jpg
│   └── results1.gif
├── setup_faceNext_env.bat
├── environment.yml
├── optimConfig.ini
├── optimConfigShadows.ini
├── optimizer.py
├── replay.py
├── demo.ipynb
└── README.md
```

# How It Works
NextFace reproduces the optimization strategy of our [early work](https://arxiv.org/abs/2101.05356). The optimization is composed of three stages:
- **Stage 1**: A coarse stage where face expression and head pose are estimated by minimizing the geometric loss between 2D landmarks and their corresponding face vertices, producing a good starting point.
- **Stage 2**: The face shape identity/expression, statistical diffuse and specular albedos, head pose, and scene light are estimated by minimizing the photometric consistency loss between the ray-traced image and the real one.
- **Stage 3**: To refine the statistical albedos from the previous stage, per-pixel optimization is performed to capture more detailed albedo information. Consistency, symmetry, and smoothness regularizers (similar to [this work](https://arxiv.org/abs/2101.05356)) are used to prevent overfitting and account for lighting conditions.

By default, NextFace uses 9-order spherical harmonics (as in [this work](https://openaccess.thecvf.com/content/ICCV2021/papers/Dib_Towards_High_Fidelity_Monocular_Face_Reconstruction_With_Rich_Reflectance_Using_ICCV_2021_paper.pdf)) for scene illumination. You can modify the number of bands in **optimConfig.ini** to improve shadow recovery.

# Good Practice for Best Reconstruction
- For optimal results, ensure images are taken in well-lit conditions (minimal shadows).
- For a single image, ensure the face is frontal to obtain a complete reconstruction (only visible parts are recovered).
- Avoid extreme facial expressions to prevent failures in reconstruction.

# Limitations
- The method relies on landmarks for initialization (Stage 1). Inaccurate landmarks may lead to suboptimal reconstructions. NextFace uses landmarks from [face_alignment](https://github.com/1adrianb/face-alignment), which are robust but not perfect.
- Reconstruction speed decreases with higher resolution images. Lower the `maxResolution` value in **optimConfig.ini** if needed.
- Fine geometry details (e.g., wrinkles, pores) might not be fully captured and can be baked into the final albedos. Our recent work [here](https://arxiv.org/abs/2203.07732) addresses fine detail capture.
- Spherical harmonics model distant lights; under strong directional shadows, residual shadows may appear. Adjust regularizer weights in **optimConfig.ini** if necessary:
  - For diffuse map: `weightDiffuseSymmetryReg` and `weightDiffuseConsistencyReg`
  - For specular map: `weightSpecularSymmetryReg` and `weightSpecularConsistencyReg`
  - For roughness map: `weightRoughnessSymmetryReg` and `weightRoughnessConsistencyReg`
  A configuration file (**optimConfigShadows.ini**) with higher regularizer values is provided.
- Single-image reconstruction is an ill-posed problem; obtaining intrinsic reflectance maps requires multiple images per subject.

# Roadmap
If time permits:
- Expression tracking from video by optimizing head pose and expression on a per-frame basis.
- Adding a virtual light stage as proposed in [this work](https://arxiv.org/abs/2101.05356) for high-frequency point lights.
- Support for the [FLAME](https://github.com/Rubikplayer/flame-fitting) morphable model.
- A GUI interface for loading images, editing landmarks, running optimization, and visualizing results.

# License
NextFace is available for free, under the GPL license, for research and educational purposes only. Please refer to the LICENSE file.

# Acknowledgements
- The UV map is taken from [here](https://github.com/unibas-gravis/parametric-face-image-generator/blob/master/data/regions/face12.json).
- Landmarks association file is taken from [here](https://github.com/kimoktm/Face2face/blob/master/data/custom_mapping.txt).
- [redner](https://github.com/BachiLi/redner/) is used for ray tracing.
- The albedo model is taken from [here](https://github.com/waps101/AlbedoMM/).

# Contact
- Email: deeb.abdallah @at gmail  
- Twitter: [abdallah_dib](https://twitter.com/abdallah_dib)

# Citation
If you use NextFace and find it helpful, please cite our work:

```
@inproceedings{dib2021practical,
  title={Practical face reconstruction via differentiable ray tracing},
  author={Dib, Abdallah and Bharaj, Gaurav and Ahn, Junghyun and Th{\'e}bault, C{\'e}dric and Gosselin, Philippe and Romeo, Marco and Chevallier, Louis},
  booktitle={Computer Graphics Forum},
  volume={40},
  number={2},
  pages={153--164},
  year={2021},
  organization={Wiley Online Library}
}

@inproceedings{dib2021towards,
  title={Towards High Fidelity Monocular Face Reconstruction with Rich Reflectance using Self-supervised Learning and Ray Tracing},
  author={Dib, Abdallah and Thebault, Cedric and Ahn, Junghyun and Gosselin, Philippe-Henri and Theobalt, Christian and Chevallier, Louis},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={12819--12829},
  year={2021}
}

@article{dib2022s2f2,
  title={S2F2: Self-Supervised High Fidelity Face Reconstruction from Monocular Image},
  author={Dib, Abdallah and Ahn, Junghyun and Thebault, Cedric and Gosselin, Philippe-Henri and Chevallier, Louis},
  journal={arXiv preprint arXiv:2203.07732},
  year={2022}
}
```
