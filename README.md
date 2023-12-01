
## 项目地址
https://github.com/li-plus/chatglm.cpp

## 项目文件说明
MANIFEST.in 文件是用于构建 Python 发布包（例如，使用 setuptools 和 distutils）时的配置文件。它指定了哪些文件应该包含在发布包中，以及如何包含这些文件。通常，MANIFEST.in 文件位于项目的根目录中。

pyproject.toml 文件是用于构建 Python 发布包（例如，使用 setuptools 和 distutils）时的配置文件。

setup.py 文件则是用于构建 Python 发布包使用 setuptools 时的配置文件。

## 配置环境
拉取子模块的依赖

```bash
git submodule update --init --recursive

python3 -m pip install -U pip
python3 -m pip install torch tabulate tqdm transformers accelerate sentencepiece
```


检查有没有安装 `cmake`，如果没有则安装，则先安装 https://cmake.org/

把下面这个目录的文件

```
C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.1\extras\visual_studio_integration\MSBuildExtensions
```

都丢到

```
C:\Program Files\Microsoft Visual Studio\2022\Community\MSBuild\Microsoft\VC\v170\BuildCustomizations
```

修改 `setup.py` 文件

```py
cmake_args = [
    # 加上这个才能使用 GPU
    f"-DGGML_CUBLAS=ON",
]
```

触发构建

```bash
pip install .
```

## 使用方法

先编译出二进制文件

```bash
cmake -B build -DGGML_CUBLAS=ON 
cmake --build build -j --config Release

./build/bin/Release/main.exe -m .\chatglm3-ggml\chatglm3-ggml-q4_0.bin  -p 请帮我生成一个夸赞宝贝可爱的短文，需要1800字
```

再安装到 Python 环境中

```bash
# 先创建虚拟环境

pip install .
```

它会自动触发构建




