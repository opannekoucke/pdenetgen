[PDE-NetGen: from symbolic PDE representations of physical processes to trainable neural network architectures/representations.](https://github.com/opannekoucke/pdenetgen)
====================================================================


---
<h2><center>Contents</center></h2>

  1. [Introduction](#introduction)
  1. [Installation](#install)
  1. [Citation](#cite)
---


Introduction <a id='introduction'/>
-----------------------------------


Bridging physics and deep learning is a topical challenge. While deep learning frameworks open avenues in physical science, the design of physically-consistent deep neural network architectures is an open issue. In the spirit of physics-informed NNs, PDE-NetGen package provides new means to automatically translate physical equations, given as PDEs, into neural network architectures. PDE-NetGen combines symbolic calculus and a neural network generator. The later exploits NN-based implementations of PDE solvers using Keras. With some knowledge of a problem, PDE-NetGen is a plug-and-play tool to generate physics-informed NN architectures. They provide computationally-efficient yet compact representations to address a variety of issues, including among others adjoint derivation, model calibration, forecasting, data assimilation as well as uncertainty quantification. As an illustration, the workflow is first presented for the 2D diffusion equation, then applied to the data-driven and physics-informed identification of uncertainty dynamics for the Burgers equation.


  - Olivier Pannekoucke and Ronan Fablet. "[PDE-NetGen: from symbolic PDE representations of physical processes to trainable neural network architectures/representations.](https://)." arXiv preprint arXiv:XXXX.XXXX (2020).


Install <a id='install'/>
-------------------------

 1. Clone the repository `git clone https://github.com/opannekoucke/pdenetgen.git`
 1. Install the package `make install` (or `python setup.py install`)


Citation <a id='cite'/>
-----------------------

    @article{Pannekoucke2020A,
      title={PDE-NetGen: from symbolic PDE representations of physical processes to trainable neural network architectures/representations.},
      author={Olivier Pannekoucke and Ronan Fablet},
      journal={arXiv preprint arXiv:},
      year={2020}
    }

