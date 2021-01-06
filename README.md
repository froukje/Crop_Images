# Analysing Crop Images

* Use the Dataset from https://www.kaggle.com/aman2000jaiswal/agriculture-crop-images
* Build a simple model for classifying images of 5 different crop sorts
* The purpose of this repository is to compare the workflow of PyTorch and PyTorch Lightning
* The following blog posts have been used:
  * https://towardsdatascience.com/from-pytorch-to-pytorch-lightning-a-gentle-introduction-b371b7caaf09
  * https://towardsdatascience.com/pytorch-lightning-machine-learning-zero-to-hero-in-75-lines-of-code-7892f3ba83c0
* Use nni (https://nni.readthedocs.io/) for hyperparamter tuning
* Use additionally Ray Tune (https://docs.ray.io/en/master/tune/) for hyperparamter tuning in pyTorch Lightning
* model.py uses pyTorch
* model_lightning.py uses pyTorch lightning
* model_nni.py is analogue to model.py, but uses nni for hyperparameter tuning with search_space.json and config.yml
* model_lightning_nni.py is analogue to model_lightning.py, but uses nni for hyperparameter tuning with search_space.json and config_lightning.yml.
* model_lightning_raytune.py uses Ray Tune for hyperparamter tuning.

