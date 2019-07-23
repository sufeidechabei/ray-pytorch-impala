import ray
import torch
from impala import DEFAULT_CONFIG
from impala import ImpalaTrainer
from ray.tune.logger import pretty_print
from model import VisionNetwork
from ray.rllib.models import ModelCatalog

ray.init()
config = DEFAULT_CONFIG.copy()
ModelCatalog.register_custom_model("my_model", VisionNetwork)
config['model'] = {"custom_model": "my_model", "custom_options": {}}
trainer = ImpalaTrainer(config=config, env="BreakoutNoFrameskip-v4")
for i in range(3000):
    result = trainer.train()
    print(pretty_print(result))
    if i % 100 == 0:
        checkpoint = trainer.save()
        print("checkpoint saved at", checkpoint)
