from setuptools import setup

setup(
    name="timetrak",
    version="0.1",
    py_modules=["timetrak"],
    install_requires=[],  
    entry_points={
        "console_scripts": [
            "timetrak = timetrak:main",
        ],
    },
)