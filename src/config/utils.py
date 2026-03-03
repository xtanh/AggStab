from omegaconf import  DictConfig, OmegaConf



def compose_config(**kwds):
    return OmegaConf.create(kwds)

def merge_config(default_cfg, override_cfg):
    return OmegaConf.merge(default_cfg, override_cfg)

def load_yaml_config(fpath: str) -> OmegaConf:
    return OmegaConf.load(fpath)