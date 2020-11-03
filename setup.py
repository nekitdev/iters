from pathlib import Path
import re

from setuptools import setup  # type: ignore

root = Path(__file__).parent

text = (root / "iters" / "__init__.py").read_text("utf-8")

result = re.search(r'^__version__\s*=\s*[\'"]([^\'"]*)[\'"]', text, re.MULTILINE)

if result is None:
    raise RuntimeError("Failed to find version.")

version = result.group(1)

readme = (root / "README.rst").read_text("utf-8")


setup(
    name="iters.py",
    author="nekitdev",
    author_email="nekitdevofficial@gmail.com",
    url="https://github.com/nekitdev/iters.py",
    project_urls={"Issue tracker": "https://github.com/nekitdev/iters.py/issues"},
    version=version,
    packages=["iters"],
    license="MIT",
    description="Rich Iterators for Python",
    long_description=readme,
    long_description_content_type="text/x-rst",
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Natural Language :: English",
        "Operating System :: OS Independent",
    ],
)
