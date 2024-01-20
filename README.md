## Project Description
This project is forked from https://github.com/li-plus/chatglm.cpp

* The MANIFEST.in file is a configuration file used for building Python distribution packages (e.g., using setuptools and distutils). It specifies which files should be included in the distribution package and how to include them. Typically, the MANIFEST.in file is located in the project's root directory.

* The pyproject.toml file is a configuration file used for building Python distribution packages (e.g., using setuptools and distutils).

* The setup.py file is a configuration file used for building Python distribution packages using setuptools.

* .pyd files are Python Dynamic Link Library files used for extension modules (written in C/C++) on the Windows platform. They are binary files of Python extension modules, usually containing the compiled results of the written Python extension modules. These extension modules allow you to interact with the underlying C/C++ code by calling their functions in Python. Typically, .pyd files are used on the Windows platform to replace .dll (Dynamic Link Library) files, for better integration with the Python runtime.

## Environment Setup
Pulling submodule dependencies:

```bash
git submodule update --init --recursive

python3 -m pip install -U pip
python3 -m pip install torch tabulate tqdm transformers accelerate sentencepiece
```

Check if `cmake` is installed, if not, then install it from https://cmake.org/

Move the files from the directory below:

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\visual_studio_integration\MSBuildExtensions
```

to

```
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations
```

Modify the `setup.py` file:

```py
cmake_args = [
    # Add this to enable GPU usage
    f"-DGGML_CUBLAS=ON",
]
```

Trigger the build:

```bash
pip install .
```

## Usage

First, compile the binary files:

```bash
cmake -B build -DGGML_CUBLAS=ON 
cmake --build build -j --config Release

./build/bin/Release/main.exe -m .\chatglm3-ggml\chatglm3-ggml-q4_0.bin  -p Please help me generate a short text praising a baby's cuteness, needing 1800 words
```

Then install it into the Python environment:

```bash
# First create a virtual environment

pip install .
```

This will automatically trigger the build.