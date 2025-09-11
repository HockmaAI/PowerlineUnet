# Powerline Unet <img width="696" height="62" alt="powerline3" src="https://github.com/user-attachments/assets/b1d9032a-c58b-45c8-a6b1-9646ced3e920" />
A U-Net model for Power Line image segmentation, generating precise masks for computer vision using drone-based inspections. Built for AI and CV enthusiasts.

## Features

- A partitioning approach to segmentation, to detect all powerline conductors
- Run with bash or command line instructions
- Fine trained model is supplied (hosted on huggingface)
- Dependencies are specified

This project addresses a critical automation step in powerline inspections

> Output masks define the power line pixels,
> serving as the crucial first step in visible 
> fault identification of power line images captured by drones. 
> Eventually, all inspections will be fully automated,
> using this technology to identify weakened 
> conductors caused by issues like
> broken strands, 
> corrosion
> untwisting (bird-caging).

This project is the result of several years of research and study
part of my honors thesis. 

<img width="800" height="1000" alt="detection" src="https://github.com/user-attachments/assets/9050aa07-df18-4966-86ed-ad127d5d2f50" />

## Installation Instructions:

- Python 3.10, CUDA 11.7, NVIDIA GPU
- CUDA/cuDNN setup NVIDIA CUDA 11.7 and cuDNN 8.9
- Use this command to install depdendencies:
```sh
    pip install -r requirements.txt
```

## Usage Instructions:
- This project uses the model hockmaAI/PowerlineUNet 
 (https://huggingface.co/hockmaAI/PowerlineUNet) hosted on Hugging Face.
- run inference using this command:
```sh
   python src/inference.py --input_path data/inference/in/1.jpg --output_path data/inference/out/1_mask.png
```
- optional commands (model_path; input_path; output_path)
- output directory is in data/inference/out

## Dependencies
- Reference requirements.txt and note any specific versions (eg. numpy<2.0 for compatability)

## Model Details
- A 4-layer UNet model with attention gates
- Trained on 32,000,000 parameters

## Sample Data
To get you started, the data/inference/in directory includes sample images. 
Simply place your own images in this directory to run the model.
For academic or research purposes, the full training dataset of 
over 2,000 images is available upon request.

## Author:
- GitHub: hockmaAI
- Email: hockma1985@gmail.com

Contributions are welcome! 
If you have ideas for improving the model, adding features, or enhancing the documentation, 
please feel free to open an issue or submit a pull request.

## License:

This project is licensed under the MIT License.
