from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="kaio",
    version="0.1.4",
    author="Kaio Platform",
    author_email="danielb@kaion5.com",
    description="Python client for the Kaio multi-tenant machine learning platform",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Kaion5-Compute/kaio",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.8",
    install_requires=[
        "requests>=2.25.0",
        "packaging>=21.0",
    ],
)
