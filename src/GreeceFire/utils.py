import os
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only
from pytorch_lightning.callbacks import ModelCheckpoint


class MyLightningCLI(LightningCLI):
    def add_arguments_to_parser(self, parser):
        parser.link_arguments("model.crop_side_length",
                              "data.crop_side_length")
        parser.add_lightning_class_args(ModelCheckpoint, 
                                        "my_model_checkpoint")
        parser.set_defaults({"my_model_checkpoint.monitor": "val/loss", 
                             "my_model_checkpoint.mode": "min",
                             "my_model_checkpoint.save_top_k": 1,
                             "my_model_checkpoint.save_last": True,
                             "my_model_checkpoint.verbose": False,
                             "my_model_checkpoint.auto_insert_metric_name": False})                                

    def before_instantiate_classes(self):  
        os.environ["WANDB_API_KEY"] = "7a9cbed74d12db3de9cef466bb7b7cf08bdf1ea4"
        os.environ["WANDB_MODE"] = "offline"        

        self.config.model.experiment = f"CC_seed{self.config.seed_everything}_{self.config.model.loss_function}_{self.config.data.crop_side_length}_{self.config.model.lr}_{self.config.model.pretrained_res}_range{self.config.data.predict_range}"
        self.config.model.pretrained_path = f"/home/aamer98/projects/def-ebrahimi/aamer98/repos/Wildfire_Bench/weights/{self.config.model.pretrained_res}.ckpt"

        self.config.trainer.logger.init_args.name = self.config.model.experiment

        self.config.my_model_checkpoint.dirpath = f"{self.config.trainer.default_root_dir}/checkpoints"
        self.config.my_model_checkpoint.filename = f"best_{self.config.model.experiment}"

    def before_fit(self):
        self.wandb_setup()

    def before_test(self):
        self.wandb_setup()

    def before_validate(self):
        self.wandb_setup()

    @rank_zero_only
    def wandb_setup(self):
        """
        Save the config used by LightningCLI to disk, then save that file to wandb.
        Using wandb.config adds some strange formating that means we'd have to do some 
        processing to be able to use it again as CLI input.

        Also define min and max metrics in wandb, because otherwise it just reports the 
        last known values, which is not what we want.
        """
        config_file_name = os.path.join(wandb.run.dir, "cli_config.yaml")
        cfg_string = self.parser.dump(self.config, skip_none=False)

        with open(config_file_name, "w") as f:
            f.write(cfg_string)
        
        wandb.save(config_file_name, policy="now", base_path=wandb.run.dir)
        wandb.define_metric("train/loss", summary="min")
        wandb.define_metric("val/loss", summary="min")
        wandb.define_metric("test/loss", summary="min")
        wandb.define_metric("train/f1", summary="max")
        wandb.define_metric("val/f1", summary="max")
        wandb.define_metric("test/f1", summary="max")
        wandb.define_metric("train/precision", summary="max")
        wandb.define_metric("val/precision", summary="max")
        wandb.define_metric("test/precision", summary="max")
        wandb.define_metric("train/avg_precision", summary="max")
        wandb.define_metric("val/avg_precision", summary="max")
        wandb.define_metric("test/avg_precision", summary="max")
        wandb.define_metric("train/recall", summary="max")
        wandb.define_metric("val/recall", summary="max")
        wandb.define_metric("test/recall", summary="max")
        wandb.define_metric("train/iou", summary="max")
        wandb.define_metric("test/iou", summary="max")