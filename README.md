# CRISPR_DeepEnsemble

Leveraging uncertainty quantification to optimise CRISPR guide RNA selection

## Preamble

> CRISPR-based genome editing relies on guide RNA sequences to target specific regions of interest. A large number of methods have been developed to predict how efficient different guides are at inducing indels. As more experimental data becomes available, methods based on machine learning have become more prominent. Here, we explore whether quantifying the uncertainty around these predictions can be used to design better guide selection strategies. We demonstrate how using a deep ensemble approach achieves better performance than utilising a single model. This approach can also provide uncertainty quantification. This allows to design, for the first time, strategies that consider uncertainty in guide RNA selection. These strategies achieve precision over 91\% and can identify suitable guides for more than 93\% of genes in the mouse genome.Our deep ensemble model is available at https://github.com/bmds-lab/CRISPR_DeepEnsemble.

## Dependencies
- [Python 3.8+](https://www.python.org/)
- [Jupyter](https://jupyter.org/)

## Getting started

- `CRISPR_DeepEnsemble.ipynb`: A notebook containing instruction on how to use the deep ensemble. Utilises the sample data from provided in the `data` directory.

- `CRISPR_DeepEnsemble.py`: A python file containing the implementation of the deep ensemble. See `CRISPR_DeepEnsemble.ipynb` for an example implementation.

## References

Bradford, J., Chappel, T., & Perrin, D. (2022). Rapid Whole-Genome Identification of High Quality CRISPR Guide RNAs with the Crackling Method. The CRISPR Journal, 5(3), 410-421.

Bradford, J., & Perrin, D. (2019). A benchmark of computational CRISPR-Cas9 guide design methods. PLoS computational biology, 15(8), e1007274.

Bradford, J., & Perrin, D. (2019). Improving CRISPR guide design with consensus approaches. BMC genomics, 20(9), 931.

Chari, R., Yeo, N. C., Chavez, A., & Church, G. M. (2017). sgRNA Scorer 2.0: a species-independent model to predict CRISPR/Cas9 activity. ACS synthetic biology, 6(5), 902-904.

Lorenz, R., Bernhart, S. H., Zu Siederdissen, C. H., Tafer, H., Flamm, C., Stadler, P. F., & Hofacker, I. L. (2011). ViennaRNA Package 2.0. Algorithms for molecular biology, 6(1), 1-14.

Montague, T. G., Cruz, J. M., Gagnon, J. A., Church, G. M., & Valen, E. (2014). CHOPCHOP: a CRISPR/Cas9 and TALEN web tool for genome editing. Nucleic acids research, 42(W1), W401-W407.

Sunagawa, G. A., Sumiyama, K., Ukai-Tadenuma, M., Perrin, D., Fujishima, H., Ukai, H., ... & Shimizu, Y. (2016). Mammalian reverse genetics without crossing reveals Nr3a as a short-sleeper gene. Cell reports, 14(3), 662-677.