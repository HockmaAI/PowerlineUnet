from setuptools import setup, find_packages

setup(
    name="powerline-unet",
    version="0.2.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.1.0",
        "torchvision>=0.16.0",
        "albumentations>=1.4.0",
        "opencv-python-headless>=4.8.0",
        "pillow>=10.0.0",
        "numpy>=1.24.0,<2.0.0",
        "pyyaml>=6.0.0",
    ],
    python_requires=">=3.10",
    author="Mark Hocking",
    author_email="hockma1985@gmail.com",
    description="U-Net for power line segmentation from drone imagery",
)
