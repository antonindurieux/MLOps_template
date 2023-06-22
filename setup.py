from pathlib import Path

from setuptools import find_namespace_packages, setup

# Load packages from requirements.txt
BASE_DIR = Path(__file__).parent
with open(Path(BASE_DIR, "requirements.txt"), "r") as file:
    required_packages = [ln.strip() for ln in file.readlines()]

# Doc
docs_packages = ["mkdocs==1.3.0", "mkdocstrings==0.18.1"]

# Style
style_packages = ["black==22.3.0", "flake8==3.9.2", "isort==5.10.1"]

# Tests
test_packages = ["pytest==7.1.2", "pytest-cov==2.10.1", "great-expectations==0.15.15"]

# Define our package
setup(
    name="tagifai",
    version=0.1,
    description="Classify machine learning projects.",
    python_requires=">=3.7",
    packages=find_namespace_packages(),
    install_requires=[required_packages],
    extra_require={"dev": docs_packages + style_packages + test_packages, "docs": docs_packages},
)
