"""onnx-static-optimizer 打包脚本。

安装（开发模式，改代码即时生效）::

    pip install -e .

安装后即可在任意目录 ``from onnx_static_optimizer import optimize``，
不再依赖 ``tests/`` 里的 sys.path hack。
"""

import os
import re

from setuptools import find_packages, setup

HERE = os.path.abspath(os.path.dirname(__file__))


def read_version() -> str:
    """从包 ``__init__.py`` 读 ``__version__``，避免版本号两处维护。

    不直接 import 该模块：那会在安装前就触发 onnx/numpy 导入，
    而此时依赖可能还没装好。
    """
    init_py = os.path.join(HERE, "onnx_static_optimizer", "__init__.py")
    with open(init_py, encoding="utf-8") as f:
        m = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', f.read(), re.M)
    if not m:
        raise RuntimeError(
            "无法从 onnx_static_optimizer/__init__.py 解析 __version__")
    return m.group(1)


def read_long_description() -> str:
    readme = os.path.join(HERE, "README.md")
    if not os.path.exists(readme):
        return ""
    with open(readme, encoding="utf-8") as f:
        return f.read()


setup(
    name="onnx-static-optimizer",
    version=read_version(),
    description="把带动态维度的 ONNX 模型转换为 NPU 编译器友好的静态模型",
    long_description=read_long_description(),
    long_description_content_type="text/markdown",
    packages=find_packages(exclude=("tests", "tests.*")),
    python_requires=">=3.8",
    install_requires=[
        "numpy",
        "onnx",
    ],
    extras_require={
        # Pass 5（ORT 子图折叠）需要；缺失时自动跳过，不影响其他 Pass
        "ort": ["onnxruntime"],
        # 跑 tests/ 全部用例需要（test_yolov8.py 依赖 ultralytics）
        "test": ["onnxruntime"],
    },
    entry_points={
        "console_scripts": [
            "onnx-static-optimizer=onnx_static_optimizer.cli:main",
        ],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
