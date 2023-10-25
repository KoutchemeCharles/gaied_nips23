# Benchmarking Educational Program Repair

Repository for the paper Benchmarking Educational Program Repair published at the Generative AI for EDucation (GAIED) workshop at NeurIPS.
In this work, we propose to benchmark multiple language models for fixing students incorrect programs, and report preliminary results using finetuned LLMs. 

You need to have the Refactory repair tool somewhere on your disk to run the evaluation.

## Coming soon... installation and running setup 




## Installation

The requirements.txt file can be used with pip to get the packages needed to run the codes.

```
pip install -r requirements.txt
```

You should complete the configuration files to specify the path towards your version of the [FalconCode dataset](https://falconcode.dfcs-cloud.net/), as well as the path towards the [singapore dataset](https://github.com/githubhuyang/refactory), as well as the path towards your installation of the [Refactory automated repair tool]((https://github.com/githubhuyang/refactory)). 

## Experiments

Before running the experiments, make sure to adapt the configurations files to
specify at least a saving directory, and, potentially, the path towards the location
of your installation of the [Refactory](https://github.com/githubhuyang/refactory) automated repair tool. 

Training the model on the FalconCode dataset

```python
python scripts/run.py --config  $config_path --experiment FinetunedRepair --train
```

where config_path is one of the configurations in the configs/neurips folder. Alternatively you could create
your own configuration following the same style.

Evaluating the trained model on the Singapore dataset

```python
python scripts/run.py --config  $config_path --experiment FinetunedRepair --evaluate --evaluate_config PATH_TO_REPO/configs/repair/dataset/falconcode_skill.json
python scripts/run.py --config  $config_path --experiment FinetunedRepair --evaluate --evaluate_config PATH_TO_REPO/configs/repair/dataset/falconcode_lab.json
python scripts/run.py --config  $config_path --experiment FinetunedRepair --evaluate --evaluate_config PATH_TO_REPO/configs/repair/dataset/singapore.json
```

The final results can be obtained by running the results.ipynb jupyter notebooks