# Learning to Generate Soft-Labels from Noisy Labels
Official code for paper [Learning to Generate Soft-Labels from Noisy Labels]().

![](metalabelnet.png)
*Illustration of the proposed MetaLabelNet algorithm*

Requirements:
* torch
* torchvision
* scikit-learn
* matplotlib

## Running Proposed Algorithm

Code can be run as follows:

```
python main.py -d dataset_name -n noise_type -r noise_ratio -s batch_size -a alpha -b beta -s1 stage1 -s2 stage2 -m meta_data_num -u unlabeled_data_num -v verbose
```

where options for input arguments are as follows

* **dataset_name:** cifar10, clothing1M, food101N
* **noise_type:** feature-dependent, symmetric (valid only for cifar10 dataset for synthetic noise)
* **noise_ratio:** integer value between 0-100 representing noise percentage (valid only for cifar10 dataset for synthetic noise)
* **batch_size:** any integer value
* **alpha:** float alpha value 
* **beta:** float beta value
* **stage1:** integer epoch value for stage1
* **stage2:** integer epoch value for stage1
* **meta_data_num:** number of meta-data
* **unlabeled_data_num:** number of unlabeled-data
* **verbose:** integer value of: 0 (silent), 1(print at each epoch), 2(print at each batch)

Any of the input parameters can be skipped to use the default value. For example, to run with default values for all parameters:

```
python main.py -d clothing1M
```
## Running Baseline Methods

Baseline methods can be run as follows:

```
python baselines.py -d dataset_name -n noise_type -r noise_ratio -m model_name 
```

where baseline model can be one of the followings:

* **model_name:** cross_entropy, symmetric_crossentropy, generalized_crossentropy, bootstrap_soft, forwardloss, joint_optimization, pencil, coteaching, mwnet, mlnt