from setuptools import setup

setup(
    name="codegen_backend",
    version="0.0.1",
    packages=["codegen_backend"],
    install_requires=[
        "openai",
        "fastapi",
        "uvicorn[standard]",
        "torch",
        "transformers",
    ],
)
