.. building-energy-storage-simulation documentation master file, created by
   sphinx-quickstart on Wed Dec 21 13:41:14 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to building-energy-storage-simulation's documentation!
==============================================================

The Building Energy Storage Simulation serves as open Source OpenAI gym (now [gymnasium](https://github.com/Farama-Foundation/Gymnasium)) environment for Reinforcement Learning. The environment represents a building with an energy storage (in form of a battery) and a solar energy system. The aim is to control the energy storage in such a way that the energy of the solar system can be used optimally.

The inspiration of this project and the data profiles come from the [CityLearn](https://github.com/intelligent-environments-lab/CityLearn) environment. Anyhow, this project focuses on the ease of usage and the simplicity of its implementation.

Check out the :doc:`usage` section for further information, including how to
:ref:`install <installation>` the project.


Contents
--------

.. toctree::
   :maxdepth: 1
   :Caption: User Guide

   usage

.. toctree::
   :maxdepth: 1
   :Caption: API

   modules/environment
   modules/simulation
   modules/building
   modules/battery


