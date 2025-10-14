from setuptools import setup, find_packages

setup(
    name="sim2real",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "mujoco",
        "pyyaml",
        "scipy",
        "torch",
        "onnxruntime",
        "pynput",
        "ipdb",
        "termcolor",
    ]
)
