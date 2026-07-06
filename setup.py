"""
Setup script for StyleSense AI, enabling the package to be installed in
editable mode (`pip install -e .`) so `src.*` imports work cleanly across
environments, including test runners and CI pipelines.
"""

from setuptools import find_packages, setup

setup(
    name="stylesense-ai",
    version="1.0.0",
    description="A multi-modal conversational shopping agent powered by Google Gemini and LangGraph.",
    author="StyleSense AI Contributors",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
    install_requires=[
        "streamlit>=1.38.0",
        "google-generativeai>=0.8.3",
        "langchain>=0.3.7",
        "langchain-core>=0.3.15",
        "langchain-google-genai>=2.0.4",
        "langgraph>=0.2.45",
        "chromadb>=0.5.18",
        "faiss-cpu>=1.9.0",
        "beautifulsoup4>=4.12.3",
        "requests>=2.32.3",
        "lxml>=5.3.0",
        "Pillow>=11.0.0",
        "pandas>=2.2.3",
        "numpy>=1.26.4",
        "python-dotenv>=1.0.1",
        "pydantic>=2.9.2",
        "pydantic-settings>=2.6.1",
        "tenacity>=9.0.0",
        "loguru>=0.7.2",
    ],
    extras_require={
        "dev": [
            "pytest>=8.3.3",
            "pytest-mock>=3.14.0",
            "pytest-cov>=6.0.0",
        ]
    },
)
