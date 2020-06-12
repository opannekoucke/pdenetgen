[PDE-NetGen : from symbolic PDE representations of physical processes to trainable neural network representations.](https://github.com/opannekoucke/pdenetgen)<!-- omit in toc -->
====================================================================


---
<h2><center>Contents</center></h2>

- [- Citation](#ullicitationliul)
- [Introduction](#introduction)
- [Example](#example)
  - [Implementation of a simple PDE: the 2D diffusion equation](#implementation-of-a-simple-pde-the-2d-diffusion-equation)
  - [Estimation of a unknown physical term](#estimation-of-a-unknown-physical-term)
    - [1. Implementation using `TrainableScalar`](#1-implementation-using-trainablescalar)
    - [2. Implementation using an external neural network (the exogenous case)](#2-implementation-using-an-external-neural-network-the-exogenous-case)
- [Install](#install)
- [Citation](#citation)
---


Introduction
------------

Bridging physics and deep learning is a topical challenge. While deep learning frameworks open avenues in physical science, the design of physically-consistent deep neural network architectures is an open issue. In the spirit of physics-informed NNs, PDE-NetGen package provides new means to automatically translate physical equations, given as PDEs, into neural network architectures. PDE-NetGen combines symbolic calculus and a neural network generator. The later exploits NN-based implementations of PDE solvers using Keras. With some knowledge of a problem, PDE-NetGen is a plug-and-play tool to generate physics-informed NN architectures. They provide computationally-efficient yet compact representations to address a variety of issues, including among others adjoint derivation, model calibration, forecasting, data assimilation as well as uncertainty quantification. As an illustration, the workflow is first presented for the 2D diffusion equation, then applied to the data-driven and physics-informed identification of uncertainty dynamics for the Burgers equation.


  - Olivier Pannekoucke and Ronan Fablet. "[PDE-NetGen 1.0: from symbolic PDE representations of physical processes to trainable neural network representations]( https://doi.org/10.5194/gmd-2020-35)." accepted in Geoscientific Model Development (2020).


Example
-------

### Implementation of a simple PDE: the 2D diffusion equation

A diffusion over a 2D domain can be implemented as

![Implementation of a 2D diffusion equation](./figures/Fig1.png)

(see the notebook ./example/pdenetgen-diffusion2D.ipynb)

### Estimation of a unknown physical term

As an example, we consider a situation that can be encountered in physics where the 
dynamics makes appear an unkown term to determine from a dataset. For the illustration, we consider the dynamics 

$$
(1)\qquad 
\left\{
\begin{array}{ccl}
\frac{\partial}{\partial t} u &=& \kappa \frac{\partial^{2}}{\partial x^{2}} u - u \frac{\partial}{\partial x} u - \frac{  \frac{\partial}{\partial x} \operatorname{{V_{u}}} }{2},\\
%%%%%
\frac{\partial}{\partial t} \operatorname{{V_{u}}} &=& - \frac{\kappa \operatorname{{V_{u}}}}{ \operatorname{{\nu_{u,xx}}} } + \kappa \frac{\partial^{2}}{\partial x^{2}} \operatorname{{V_{u}}} - \frac{\kappa \left(\frac{\partial}{\partial x} \operatorname{{V_{u}}}\right)^{2}}{2 \operatorname{{V_{u}}}}-  u  \frac{\partial}{\partial x} \operatorname{{V_{u}}} - 2 \operatorname{{V_{u}}} \frac{\partial}{\partial x} u,\\
%%%%%
\frac{\partial}{\partial t} \operatorname{{\nu_{u,xx}}} &=& 4 \kappa \operatorname{{\nu_{u,xx}}}^{2} \mathbb E\left[\operatorname{{\varepsilon_{u}}} \frac{\partial^{4}}{\partial x^{4}} \operatorname{{\varepsilon_{u}}}\right] - 3 \kappa \frac{\partial^{2}}{\partial x^{2}} \operatorname{{\nu_{u,xx}}}- \kappa + \\ 
& &\frac{6 \kappa \left(\frac{\partial}{\partial x} \operatorname{{\nu_{u,xx}}}\right)^{2}}{\operatorname{{\nu_{u,xx}}}} - \frac{2 \kappa \operatorname{{\nu_{u,xx}}} \frac{\partial^{2}}{\partial x^{2}} \operatorname{{V_{u}}} }{  \operatorname{{V_{u}}}  } +
\frac{\kappa  \frac{\partial}{\partial x} \operatorname{{V_{u}}}  \frac{\partial}{\partial x} \operatorname{{\nu_{u,xx}}}}{ \operatorname{{V_{u}}}  } +
	\frac{2 \kappa \operatorname{{\nu_{u,xx}}}  \left(\frac{\partial}{\partial x} \operatorname{{V_{u}}}\right)^{2}  }{  \operatorname{{V_{u}}}^{2}  } -  u  \frac{\partial}{\partial x} \operatorname{{\nu_{u,xx}}} +
	2 \operatorname{{\nu_{u,xx}}}  \frac{\partial}{\partial x} u 
\end{array}
\right.
$$
where the term $\mathbb E\left[\operatorname{{\varepsilon_{u}}} \frac{\partial^{4}}{\partial x^{4}} \operatorname{{\varepsilon_{u}}}\right]$ is unkown.


With PDE-NetGen, we can design a closure from the data. For the 
illustration we consider a candidate for the closure, given by

<img src="https://render.githubusercontent.com/render/math?math=(2)\qquad \mathbb E\left[\operatorname{{\varepsilon_{u}}} \frac{\partial^{4}}{\partial x^{4}} \operatorname{{\varepsilon_{u}}}\right]
\sim 
a \frac{\frac{\partial^{2}}{\partial x^{2}} \operatorname{{\nu_{u,xx}}}{\left(t,x \right)}}{\operatorname{{\nu_{u,xx}}}^{2}{\left(t,x \right)}} + 
b \frac{1}{ \operatorname{{\nu_{u,xx}}}^{2}{\left(t,x \right)}} +
c \frac{\left(\frac{\partial}{\partial x} \operatorname{{\nu_{u,xx}}}{\left(t,x \right)}\right)^{2}}{\operatorname{{\nu_{u,xx}}}^{3}{\left(t,x \right)}},">
where $(a,b,c)$ are unkowns.

Two implementations can be considered. 

#### 1. Implementation using `TrainableScalar`

This is implemented by using `TrainableScalar` as follows:
 
 1. the candidate for the closure is defined as a symbolic level
 2. a neural network that implement the full dynamics (including the closure) is then generated and ready for the training

![Implementation of a closure](./figures/Fig5.png)

The use of `TrainableScalar` is the simplest way to try a closure designed from partial derivatives.

(see the notebook ./example/pdenetgen-NN-PKF_burgers_learn-TrainableScalar-closure.ipynb)

#### 2. Implementation using an external neural network (the exogenous case)

Another implemention is possible that relies on the introduction
of an external neural network, for instance a deep neural network of your choice that you have to build by yourself and
that can be plugged to the neural network generated from Eq.(1)
. 

(see the notebook ./example/pdenetgen-NN-PKF_burgers_learn-exogenous-closure.ipynb where the closure is provided as the implementation of Eq.(2) -- you can try your own NN that can be different from the candidate Eq.(2))

Install
-------

 1. Clone the repository `git clone https://github.com/opannekoucke/pdenetgen.git`
 1. Install the package `make install` (or `python setup.py install`)
 1. Examples are given as jupyter notebooks (see ./example/) 


Citation
--------

@Article{gmd-2020-35,
AUTHOR = {Pannekoucke, O. and Fablet, R.},
TITLE = {PDE-NetGen 1.0: from symbolic PDE representations of physical
processes to trainable neural network representations},
JOURNAL = {Geoscientific Model Development},
VOLUME = {2020},
YEAR = {2020},
PAGES = {1--14},
DOI = {10.5194/gmd-2020-35}
}
