"""Setup script for The Hobbit Scholar CLI application."""

from setuptools import setup
from pathlib import Path

# Read README for long description
readme_file = Path(__file__).parent / "README.md"
try:
    if readme_file.exists():
        long_description = readme_file.read_text(encoding="utf-8-sig")  # Handle BOM
    else:
        long_description = ""
except (UnicodeDecodeError, OSError):
    long_description = "RAG-powered chatbot for answering questions about The Hobbit"

INSTALL_REQUIRES = [
    # Runtime dependencies required by `main.py` and `vector.py`.
    "langchain>=1.0,<2.0",
    "langchain-core>=1.0,<2.0",
    "langchain-classic>=1.0,<2.0",
    "langchain-community>=0.4,<1.0",
    "langchain-chroma>=1.0,<2.0",
    "langchain-ollama>=1.0,<2.0",
    "langchain-text-splitters>=1.0,<2.0",
    "chromadb>=1.0,<2.0",
    "flashrank>=0.2,<1.0",
]

setup(
    name="hobbit-scholar",
    version="1.0.0",
    description="RAG-powered chatbot for answering questions about The Hobbit",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Your Name",
    author_email="your.email@example.com",
    url="https://github.com/Chams15/RagAgentHobbit",
    py_modules=["main", "vector"],
    install_requires=INSTALL_REQUIRES,
    extras_require={
        "dev": [
            "pytest>=8.0",
            "build>=1.0",
            "twine>=5.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "hobbit-scholar=main:main",
            "hobbit-init=vector:main_cli",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    python_requires=">=3.9",
    include_package_data=True,
    package_data={
        "": ["TheHobbit.md", "*.txt"],
    },
)
