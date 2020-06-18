from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

if __name__ == "__main__":
    setup(name="goes16ci",
          version="0.1.3.b0",
          description="Machine learning benchmark for lightning prediction with GOES16",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          license="MIT",
          long_description=long_description,
          long_description_content_type="text/markdown",
          url="https://github.com/NCAR/goes16ci",
          packages=["goes16ci"],
          install_requires=["numpy",
                            "scipy",
                            "pandas",
                            "tensorflow>=1.15.2",
                            "xarray",
                            "dask",
                            "pyyaml",
                            "s3fs"]
          )
