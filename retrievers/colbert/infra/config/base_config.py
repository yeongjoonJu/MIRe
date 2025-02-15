import os
import torch
import ujson, json
import dataclasses

from typing import Any
from collections import defaultdict
from dataclasses import dataclass, fields
from retrievers.colbert.utils.utils import timestamp, torch_load_dnn

from .core_config import *
from retrievers.colbert.utilities.save_metadata import get_metadata_only
from omegaconf import DictConfig, OmegaConf

@dataclass
class BaseConfig(CoreConfig):
    @classmethod
    def from_existing(cls, *sources):
        kw_args = {}

        for source in sources:
            if source is None:
                continue
            local_kw_args = dataclasses.asdict(source)
            local_kw_args = {k: local_kw_args[k] for k in source.assigned if k in local_kw_args}
            kw_args = {**kw_args, **local_kw_args}

        obj = cls(**kw_args)

        return obj

    @classmethod
    def from_deprecated_args(cls, args):
        obj = cls()
        ignored = obj.configure(ignore_unrecognized=True, **args)

        return obj, ignored

    @classmethod
    def from_path(cls, name):
        with open(name) as f:
            args = ujson.load(f)

            if 'config' in args:
                args = args['config']

        return cls.from_deprecated_args(args)  # the new, non-deprecated version functions the same at this level.
    
    @classmethod
    def load_from_checkpoint(cls, checkpoint_path):
        if checkpoint_path.endswith('.dnn'):
            dnn = torch_load_dnn(checkpoint_path)
            config, _ = cls.from_deprecated_args(dnn.get('arguments', {}))

            # TODO: FIXME: Decide if the line below will have any unintended consequences. We don't want to overwrite those!
            config.set('checkpoint', checkpoint_path)

            return config

        loaded_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        if os.path.exists(loaded_config_path):
            loaded_config, _ = cls.from_path(loaded_config_path)
            loaded_config.set('checkpoint', checkpoint_path)

            return loaded_config

        return None  # can happen if checkpoint_path is something like 'bert-base-uncased'

    @classmethod
    def load_from_index(cls, index_path):
        # FIXME: We should start here with initial_config = ColBERTConfig(config, Run().config).
        # This should allow us to say initial_config.index_root. Then, below, set config = Config(..., initial_c)

        # default_index_root = os.path.join(Run().root, Run().experiment, 'indexes/')
        # index_path = os.path.join(default_index_root, index_path)

        # CONSIDER: No more plan/metadata.json. Only metadata.json to avoid weird issues when loading an index.

        try:
            metadata_path = os.path.join(index_path, 'metadata.json')
            loaded_config, _ = cls.from_path(metadata_path)
        except:
            metadata_path = os.path.join(index_path, 'plan.json')
            loaded_config, _ = cls.from_path(metadata_path)
        
        return loaded_config

    def save(self, path, overwrite=False):
        assert overwrite or not os.path.exists(path), path

        args = self.export()  # dict(self.__config)
        del args["assigned"]
        # if "wandb_config" in args:
        #     del args["wandb_config"]
        # if "logger_config" in args:
        #     del args["logger_config"]
        # if "dist_config" in args:
        #     del args["dist_config"]
        # if "ckpt_dir" in args:
        #     del args["ckpt_dir"]
        # args['meta'] = get_metadata_only()
        # args['meta']['version'] = 'colbert-v0.4'
        
        # if isinstance(args, DictConfig):
        #     args = OmegaConf.to_container(args, resolve=True)
        # args = json.JSONEncoder.default(self, args)
        
        with open(path, "w") as f:
            json.dump(args, f, ensure_ascii=False, indent=2)
        # with open(path, 'w') as f:
            # TODO: Add git_status details.. It can't be too large! It should be a path that Runs() saves on exit, maybe!
            # f.write(ujson.dumps(args, indent=4) + '\n')
            

    def save_for_checkpoint(self, checkpoint_path):
        assert not checkpoint_path.endswith('.dnn'), \
            f"{checkpoint_path}: We reserve *.dnn names for the deprecated checkpoint format."

        output_config_path = os.path.join(checkpoint_path, 'artifact.metadata')
        self.save(output_config_path, overwrite=True)
