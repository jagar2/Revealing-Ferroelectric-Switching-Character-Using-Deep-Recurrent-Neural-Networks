# Revealing-Ferroelectric-Switching-Character-Using-Deep-Recurrent-Neural-Networks
The ability to manipulate domains and domain walls underpins function in a range of next-generation applications of ferroelectrics. While there have been demonstrations of controlled nanoscale manipulation of domain structures to drive emergent properties, such approaches lack an internal feedback loop required for automation. Here, using a deep sequence-to-sequence autoencoder we automate the extraction of features of nanoscale ferroelectric switching from multichannel hyperspectral band-excitation piezoresponse force microscopy of tensile-strained PbZr<sub>0.2</sub>Ti<sub>0.8</sub>O<sub>3</sub> with a hierarchical domain structure. Using this approach, we identify characteristic behavior in the piezoresponse and cantilever resonance hysteresis loops, which allows for the classification and quantification of nanoscale-switching mechanisms. Specifically, we are able to identify elastic hardening events which are associated with the nucleation and growth of charged domain walls. This work demonstrates the efficacy of unsupervised neural networks in learning features of the physical response of a material from nanoscale multichannel hyperspectral imagery and provides new capabilities in leveraging multimodal *in operando* spectroscopies and automated control for the manipulation of nanoscale structures in materials.

Note only the most important portions of the data are contained in the git repository, as some of the data sets (phase field, and training checkpoints) are very large (>30 gb). If these files are desired they can be dowloaded from: 

Phase field
https://zenodo.org/record/1482091/files/Phase_field.zip?download=1

Training data
https://zenodo.org/record/1482091/files/Trained_models.zip?download=1

[![DOI](https://zenodo.org/badge/124429041.svg)](https://zenodo.org/badge/latestdoi/124429041)

1. Go to https://colab.research.google.com/
2. Select load from GitHub: https://github.com/jagar2/Revealing-Ferroelectric-Switching-Character-Using-Deep-Recurrent-Neural-Networks.git
![](Readme_images/Image1.png)
3. Open Jupyter notebook Revealing Ferroelectric Switching Character Using Deep Recurrent Neural Networks-Collaboratory.ipynb
![](Readme_images/Image2.png)
4. Change runtime type to include a GPU (this is only necessary to train new networks)
![](Readme_images/Image3.png)
5. Run code by pressing `shift + enter` on cell blocks until prompted to provide authorization code. Follow onscreen instructions to link to google drive.
![](Readme_images/Image4.png)
6. You can run scripts using `shift + enter`
 
