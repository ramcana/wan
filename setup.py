import pytest
#!/usr/bin/env python3
"""Setup script for WAN CLI toolkit"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README for long description
readme_path = Path(__file__).parent / "docs" / "GETTING_STARTED.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="wan-cli",
    version="1.0.0",
    description="WAN Project Quality & Maintenance Toolkit - Unified CLI for development workflows",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="WAN Development Team",
    python_requires=">=3.8",
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        "typer>=0.9.0",
        "rich>=13.0.0",
        "pyyaml>=6.0",
        "pytest>=7.0.0",
        "black>=23.0.0",
        "ruff>=0.1.0",
        "coverage>=7.0.0",
    ],
    extras_require={
        "dev": [
            "pre-commit>=3.0.0",
            "pytest-cov>=4.0.0",
            "pytest-xdist>=3.0.0",
        ],
        "docs": [
            "mkdocs>=1.5.0",
            "mkdocs-material>=9.0.0",
        ],
        "monitoring": [
            "psutil>=5.9.0",
            "prometheus-client>=0.17.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "wan-cli=cli.main:app",
        ],
    },
    scripts=["wan-cli"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Topic :: Software Development :: Quality Assurance",
        "Topic :: Software Development :: Testing",
        "Topic :: System :: Systems Administration",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
    keywords="cli development quality testing maintenance automation",
    project_urls={
        "Documentation": "https://github.com/your-org/wan-project/docs",
        "Source": "https://github.com/your-org/wan-project",
        "Tracker": "https://github.com/your-org/wan-project/issues",
    },
)