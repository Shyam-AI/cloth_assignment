from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

setup(
    name="src",
    version="0.0.1",
    author="Shyam-AI",
    description="Clothing detection",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/Shyam-AI/cloth",
    author_email="shyamdl2803@gmail.com",
    packages=["src"],
    python_requires=">=3.6"
   
)