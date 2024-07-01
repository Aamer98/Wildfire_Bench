import os
import wandb

from GreeceFire.utils import MyLightningCLI
from models.climax.module import ClimaXModule
from GreeceFire.datamodule import GreeceFireDataModule


def main():
    # Initialize Lightning with the model and data modules, and instruct it to parse the config yml
    cli = MyLightningCLI(
        model_class=ClimaXModule,
        datamodule_class=GreeceFireDataModule,
        seed_everything_default=42,
        save_config_overwrite=True,
        run=False,
        auto_registry=True,
        parser_kwargs={"parser_mode": "omegaconf", "error_handler": None},
    )
    cli.wandb_setup()

    # set lead time for prediction
    cli.model.set_pred_range(cli.datamodule.hparams.predict_range)

    # fit() runs the training
    cli.trainer.fit(cli.model, datamodule=cli.datamodule)
    
    # test the trained model
    cli.trainer.test(cli.model, datamodule=cli.datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()