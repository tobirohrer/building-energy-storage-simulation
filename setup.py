from distutils.core import setup
from pathlib import Path
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

setup(name='building_energy_storage_simulation',
      version='0.7.0',
      description='A simulation of a building to optimize energy storage utilization.',
      long_description=long_description,
      long_description_content_type='text/markdown',
      author='Tobias Rohrer',
      author_email='tobias.rohrer@outlook.com',
      url="https://github.com/tobirohrer/building-energy-storage-simulation",
      packages=['building_energy_storage_simulation'],
      package_dir={'building_energy_storage_simulation': 'building_energy_storage_simulation'},
      # Required to include profiles which are stored as .csv files
      package_data={
          "building_energy_storage_simulation": [
              "data/preprocessed/*.csv",
          ]
      },
      include_package_data=True,
      install_requires=[
          "gymnasium",
          "pandas",
          "numpy"
      ],
      extras_require={
          "docs": [
              "sphinx"
          ],
          "tests": [
              "pytest"
          ]
      }
      )
