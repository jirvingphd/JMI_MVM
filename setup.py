import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="JMI_MVM",
    version="0.2.1",
    author="James M. Irving, Michael V. Moravetz",
    author_email="james.irving.phd@outlook.com",
    description="A collection of our functions and classes from bootcamp. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jirvingphd/JMI_MVM",
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=['numpy','pandas','seaborn','matplotlib','sklearn','pydotplus','scipy','xgboost'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent"
    ],
)