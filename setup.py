"""Setup configuration"""
from setuptools import setup, find_packages

setup(
    name="face-recognition-system",
    version="2.0.0",
    description="Production-grade face recognition system",
    author="AI Engineer",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "facenet-pytorch>=2.5.3",
        "opencv-python>=4.8.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "flask>=3.0.0",
        "face-recognition>=1.3.0",
        "scikit-learn>=1.3.0",
        "matplotlib>=3.7.0",
        "seaborn>=0.12.0",
        "joblib>=1.3.2",
    ],
    python_requires=">=3.8",
)
