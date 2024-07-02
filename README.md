# Bilinear Generalized Vector Approximate Message Passing (BiG-VAMP)
##### Mohamed Akrout, Anis Housseini, Faouzi Bellili, Amine Mezghani
This repository contains the Matlab code of the algorithm proposed in the paper: [Bilinear Generalized Vector Approximate Message Passing](https://arxiv.org/abs/2009.06854).


## Abstract
We introduce the bilinear generalized vector approximate message passing (BiG-VAMP) algorithm which jointly recovers two matrices $\boldsymbol{U}$ and $\boldsymbol{V}$ from their noisy product through a probabilistic  observation model. BiG-VAMP provides computationally efficient approximate implementations of both max-sum and sum-product loopy belief propagation (BP). We show how the proposed BiG-VAMP recovers different types of structured matrices and overcomes the fundamental limitations of other state-of-the-art techniques to the bilinear recovery problem, such as BiG-AMP, BAd-VAMP and LowRAMP. In essence, BiG-VAMP applies to a broader class of practical applications which involve  a general form of structured matrices. For the sake of theoretical performance prediction, we also conduct a state evolution (SE) analysis of the proposed algorithm  and show its consistency with the asymptotic empirical  mean-squared error (MSE). Numerical results on various applications such as matrix factorization, dictionary learning, and matrix completion demonstrate unambiguously the effectiveness of the proposed BiG-VAMP algorithm and its superiority over state-of-the-art algorithms. Using the developed SE framework, we also examine the phase transition diagrams of the matrix completion problem, thereby unveiling a low detectability region corresponding to the low signal-to-noise ratio (SNR) regime.

## Repository Structure
This repository contains four folders:
  - **Bi-VAMP**: it contains the code of Bi-VAMP algorithm solving the noisy matrix factorization problem whose observation model is $\boldsymbol{Y}~ =~ \boldsymbol{U}\boldsymbol{V}^{\top} + \boldsymbol{W}$. This folder has two files:
    * *Bi-VAMP.m*: the file containing the algorithmic steps of BiG-VAMP.
    * *Parameters.m*: the file defining the parameters of the BiG-VAMP simulation.
  - **BiG-VAMP**: it contains the code of the BiG-VAMP solving the noisy real-valued matrix completion (MC) problem whose observation model is $\boldsymbol{Y}~ =~ \boldsymbol{S}\cdot(\boldsymbol{U}\boldsymbol{V}^{\top} + \boldsymbol{W})$. Here, $\boldsymbol{S}$ is the binary selection matrix masking a specific percentage of the entries in the matrix $\boldsymbol{U}\boldsymbol{V}^{\top} + \boldsymbol{W}$. This folder has two files
    * *Bi-VAMP-MC.m*: the file containing the algorithmic steps of BiG-VAMP for matrix completion (BiG-VAMP-MC).
    * *Parameters.m*: the file defining the parameters of the BiG-VAMP-MC simulation.
- **functions**: it contains specific functions called by Bi-VAMP and/or BiG-VAMP-MC.
- **bigamp-package**: it contains the code of the [BiG-AMP](https://arxiv.org/abs/1310.2632) algorithm from the [GAMPMATLAB](https://sourceforge.net/projects/gampmatlab/) package.

## Running Experiments
#### Key Parameters
The parameters of Bi-VAMP and BiG-VAMP-MC are summarized in the object `Parameters` and are summarized below:
| Name &nbsp; &nbsp; &nbsp; &nbsp; | Description | 
| :---         |             :--- |
| nb_iter         |     max number of iterations      |
| conv_criterion         |     convergence criterion      |
| damping     | damping coefficient   | 
| prior_u     | prior on the matrix U   | 
| prior_v   | prior on the matrix V     | 
| prior_u_option     | prior parameters for U   |
| prior_v_option     | prior parameters for V   |
| selection_percentage     | parameter to control the # of 0s in Y  (only for BiG-VAMP-MC) |
| beta    | temperature parameter  |
| seed    | seed for reproducibility  |
#### Entry Points
- To run Bi-VAMP, execute the file `main_linear.m`
- To run BiG-VAMP for matrix completion, execute the file `main_MC.m`
- To run BiG-VAMP vs. BiG-AMP for matrix completion, execute the file `main_BiGAMP_vs_BiGVAMP_MC_binary_prior.m`. The output in the Matlab console after 100 Monte Carlo simulations should be:
```
Running BiG-AMP and BiG-VAMP with:
A binary {-1, 1} and X Gaussian(0,1)

=============== Results of BiG-AMP ===============
nrmse BiG-AMP = 0.601415 (+/- 0.013788)
Running time BiG-AMP = 788.602854 (+/- 112.390760)
=============== Results of BiG-VAMP ===============
nrmse BiG-VAMP = 0.122554 (+/- 0.045154)
Running time BiG-VAMP = 29.316404 (+/- 37.978760)
```
## Important facts
- If either $\boldsymbol{U}$ or $\boldsymbol{V}$ is known, BiG-VAMP coincides with the multiple measurement vectors setting of the [VAMP](https://arxiv.org/abs/1610.03082) algorithm.
- The BiG-VAMP algorithm outperforms the [BiG-AMP](https://arxiv.org/abs/1310.2632) algorithm for discrete-valued priors (e.g., binary priors).
- For benchmarking purposes, the value of the damping parameter should be tuned depending on the problem dimension (i.e., n, m, and r), the priors on $\boldsymbol{U}$ and $\boldsymbol{V}$, and the non-linearity parameters (e.g., selection percentage in matrix completion).
  
## Citing the paper (bib)

If you make use of our code, please make sure to cite our paper:
```
@inproceedings{akrout2022big,
  title={BiG-VAMP: The Bilinear Generalized Vector Approximate Message Algorithm},
  author={Akrout, Mohamed and Housseini, Anis and Bellili, Faouzi and Mezghani, Amine},
  booktitle={2022 56th Asilomar Conference on Signals, Systems, and Computers},
  pages={1377--1384},
  year={2022},
  organization={IEEE}
}
```
