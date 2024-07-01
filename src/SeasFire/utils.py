import os
import wandb
from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.utilities import rank_zero_only


class MyLightningCLI(LightningCLI):
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
        wandb.init(
            # Set the project where this run will be logged
            project=f"ClimaX_SeasFire_{self.model.experiment}", 
            # We pass a run name (otherwise it’ll be randomly assigned, like sunshine-lollypop-10)
            # Track hyperparameters and run metadata
            config={
            "learning_rate": self.model.hparams.lr,
            "seed": self.config.seed_everything,
            "batch_size": self.datamodule.hparams.batch_size,
        })

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