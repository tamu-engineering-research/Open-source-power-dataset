# Synthetic Data Generation
Here we describe how to reproduce the benchmark results for synthetic time-series generation given PMU data.

## Relevant Packages Install
- Create and activate anaconda virtual environment for each method
```angular2html
conda create -n TSGeneration python=3.x.x
conda activate TSGeneration
```
- Install required packages
```angular2html
pip install -r requirements.txt
```

## Benchmark Results Reproduction
You can find all codes of different models in folders above and their respective configuration files within each folder. You can directly run `main.py` from the models above to produce `generated_samples.npz`.

## References

1. **DoppelGANger**
   
   <em>Lin, Zinan, et al. "Using GANs for sharing networked time series data: Challenges, initial promise, and open questions." Proceedings of the ACM Internet Measurement Conference (2020).</em>
   
   https://github.com/fjxmlzn/DoppelGANger 
   
1. **COT-GAN**
   
   <em>Xu, Tianlin, et al. "Cot-gan: Generating sequential data via causal optimal transport." arXiv preprint arXiv:2006.08571 (2020).</em>
   
   https://github.com/tianlinxu312/cot-gan
   
1. **TimeGAN**
   
   <em>Yoon, Jinsung, Daniel Jarrett, and Mihaela Van der Schaar. "Time-series generative adversarial networks." (2019).</em>
   
   https://github.com/jsyoon0823/TimeGAN
   
1. **RCGAN**
      
   <em>Esteban, Cristóbal, Stephanie L. Hyland, and Gunnar Rätsch. "Real-valued (medical) time series generation with recurrent conditional gans." arXiv preprint arXiv:1706.02633 (2017).</em>
   
   https://github.com/ratschlab/RGAN
   
1. **WGAN-GP (NaiveGAN)**
   
   <em>Gulrajani, Ishaan, et al. "Improved training of wasserstein gans." arXiv preprint arXiv:1704.00028 (2017).</em>
  