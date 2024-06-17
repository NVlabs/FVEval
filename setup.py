import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="fv-eval",
    py_modules=["fv-eval"],
    version="0.1",
    description="",
    author="Nvidia",
    packages=find_packages(),
    # install_requires=[
    #     str(r)
    #     for r in pkg_resources.parse_requirements(
    #         open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    #     )
    # ],
    entry_points={
        "console_scripts": [
            "evaluate_functional_correctness = verilog_eval.evaluate_functional_correctness",
        ]
    }
)
