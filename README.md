<div align="center">
  
  <div>
  <h1>Weighted Ensemble Models Are Strong Continual Learners</h1>
  </div>

</div>

In this work, we study the problem of continual learning (CL) where the goal is to learn a model on a 
sequence of tasks, with the data from previous tasks becoming unavailable while learning on the current
task data. CL is essentially a balancing act between learning new tasks (plasticity) and maintaining 
performance on previously learned concepts (stability). To address the stability-plasticity trade-off, 
we propose performing weight-ensembling of the model parameters of the previous and current task. 
This weight-ensembled model, which we call Continual Model Averaging (or CoMA), achieves high accuracy 
on the current task without deviating significantly from the previous weight configuration, ensuring stability. 
We also propose an improved variant of CoMA, named Continual 
Fisher-weighted Model Averaging (or CoFiMA), that selectively weighs each parameter in the weight ensemble 
by leveraging the Fisher information of the model's weights. Both variants are conceptually simple, easy to 
implement, and effective in achieving state-of-the-art performance on several standard CL benchmarks.
## Requirement
install the conda environment using the environment.yml file

## Pre-trained Models
Please download pre-trained ViT-Base models from [MoCo v3](https://drive.google.com/file/d/1bshDu4jEKztZZvwpTVXSAuCsDoXwCkfy/view?usp=share_link) and [ImageNet-21K](https://drive.google.com/file/d/1PcAOf0tJYs1FVDpj-7lrkSuwXTJXVmuk/view?usp=share_link) and 
then put or link the pre-trained models to ```CoFiMA/pretrained```


## Log file 

The default log file for CoFiMA evaluated on the main benchmarks in Tab.1 are in ```CoFiMA/log.txt```


## Training
to launch the training of CoMA or CoFiMA on CIFAR-100, run the following command:

```bash train_all.sh```

## Acknolegment
This repo is heavily based on [PyCIL](https://github.com/G-U-N/PyCIL), many thanks.
