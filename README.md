# CRISPR_DeepEnsemble

Leveraging uncertainty quantification to optimise CRISPR guide RNA selection

## Preamble

> CRISPR-based genome editing relies on guide RNA sequences to target specific regions of interest. A large number of methods have been developed to predict how efficient different guides are at inducing indels. As more experimental data becomes available, methods based on machine learning have become more prominent. Here, we explore whether quantifying the uncertainty around these predictions can be used to design better guide selection strategies. We demonstrate how using a deep ensemble approach achieves better performance than utilising a single model. This approach can also provide uncertainty quantification. This allows to design, for the first time, strategies that consider uncertainty in guide RNA selection. These strategies achieve precision over 91\% and can identify suitable guides for more than 93\% of genes in the mouse genome.Our deep ensemble model is available at https://github.com/bmds-lab/CRISPR_DeepEnsemble.

## Dependencies
- [Python 3.8+](https://www.python.org/)
- [Jupyter](https://jupyter.org/)

Optional (Only required if you intend to run the data preprocessing again. We have already provided the final dataset, `merged_X.pt`, `merged_Features.pt` and `merged_Y.pt`)
- [Bowtie2](https://bowtie-bio.sourceforge.net/bowtie2/index.shtml)
- [SAMtools](https://www.htslib.org/)

## Getting started

- `CRISPR_DeepEnsemble.ipynb`: A notebook containing instruction on how to use the deep ensemble. Utilises the sample data from provided in the `data` directory.

- `CRISPR_DeepEnsemble.py`: A python file containing the implementation of the deep ensemble. See `CRISPR_DeepEnsemble.ipynb` for an example implementation.

## Data
Data was obtained from Kim et al. (2019), Wang et al. (2019), Kim et al. (2020) and Xiang et al. (2021). All datasets have a corresponding folder in the `data` directory. Each folder contains a notebook that has details on how the data was obtained and processed.

## References

Bradford, J., Chappel, T., & Perrin, D. (2022). Rapid Whole-Genome Identification of High Quality CRISPR Guide RNAs with the Crackling Method. The CRISPR Journal, 5(3), 410-421.

Bradford, J., & Perrin, D. (2019). A benchmark of computational CRISPR-Cas9 guide design methods. PLoS computational biology, 15(8), e1007274.

Bradford, J., & Perrin, D. (2019). Improving CRISPR guide design with consensus approaches. BMC genomics, 20(9), 931.

Chari, R., Yeo, N. C., Chavez, A., & Church, G. M. (2017). sgRNA Scorer 2.0: a species-independent model to predict CRISPR/Cas9 activity. ACS synthetic biology, 6(5), 902-904.

Kim, H. K., Kim, Y., Lee, S., Min, S., Bae, J. Y., Choi, J. W., Park, J., Jung, D., Yoon, S., & Kim, H. H. (2019). SpCas9 activity prediction by DeepSpCas9, a deep learning–based model with high generalization performance. Science Advances, 5(11), eaax9249. 

Kim, N., Kim, H. K., Lee, S., Seo, J. H., Choi, J. W., Park, J., Min, S., Yoon, S., Cho, S.-R., & Kim, H. H. (2020). Prediction of the sequence-specific cleavage activity of Cas9 variants. Nature Biotechnology, 38(11), 1328-1336.

Wang, D., Zhang, C., Wang, B., Li, B., Wang, Q., Liu, D., Wang, H., Zhou, Y., Shi, L., Lan, F., & Wang, Y. (2019). Optimized CRISPR guide RNA design for two high-fidelity Cas9 variants by deep learning. Nature Communications, 10(1), 4284.

Xiang, X., Corsi, G. I., Anthon, C., Qu, K., Pan, X., Liang, X., Han, P., Dong, Z., Liu, L., Zhong, J., Ma, T., Wang, J., Zhang, X., Jiang, H., Xu, F., Liu, X., Xu, X., Wang, J., Yang, H., Bolund, L., Church, G. M., Lin, L., Gorodkin, J., & Luo, Y. (2021). Enhancing CRISPR-Cas9 gRNA efficiency prediction by data integration and deep learning. Nature Communications, 12(1), 3238.