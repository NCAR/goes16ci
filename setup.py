from setuptools import setup

if __name__ == "__main__":
    setup(name="goes16ci",
          version="0.1.2",
          description="Machine learning benchmark for lightning prediction with GOES16",
          author="David John Gagne",
          author_email="dgagne@ucar.edu",
          license="MIT",
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
