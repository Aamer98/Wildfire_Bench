import os
from typing import List
import torch
import numpy as np
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
            project=f"ClimaX_WildfireSpreadTS_{self.model.experiment}", 
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


def get_means_stds_missing_values(training_years: List[int]):
    """_summary_ Returns mean and std values as tensor, computed on unaugmented and unstandardized 
    data of the indicated training years. We don't clip values, because min/max did not diverge 
    much from the 0.1 and 99.9 percentiles. Some variables are not standardized, indicated by mean=0, std=1. 
    These are specifically: All variables indicating a direction in degrees 
    (wind direction, aspect, forecast wind direction), and the categorical land cover type.

    Args:
        training_years (_type_): _description_

    Returns:
        _type_: _description_
    """
    stats_per_training_year_combo = {
        (2018, 2019): {
        'means': np.array([
            1.95905826e+03,  2.94404070e+03,  1.80315792e+03,  4.18785304e+03,
            2.20147914e+03,  4.75643503e-01,  3.40356332e+00,  5.20851084e-03,
            2.82528452e+02,  2.99264656e+02,  6.81236923e+01,  5.48062173e-03,
            6.72659479e+00,  2.51996541e-03,  1.53416361e+03, -1.13553723e+00,
            8.92881266e+00,  1.84884483e+01,  1.59374234e+00,  9.44323569e-02,
            1.84031356e+01,  5.57368450e-03,  1.48318570e+01], dtype=np.float32),
        'stds': np.array([
            1.13697378e+03, 1.63707682e+03, 1.89291266e+03, 2.21105881e+03,
            1.12833037e+03, 2.15286520e+00, 1.39548951e+00, 7.07312261e-01,
            6.87025186e+00, 7.96616357e+00, 1.89715391e+01, 2.10764239e-03,
            6.75470828e+00, 7.05844663e-01, 7.69800231e+02, 1.88290894e+00,
            3.14455922e+00, 4.32875914e+01, 1.07801499e+00, 5.67901367e-01,
            6.98550201e+00, 1.92365459e-03, 5.52618558e+00], dtype=np.float32),
        'missing_values': np.array([
            0.02567231, 0.0256701, 0.02566863, 0.02243902, 0.02243973,
            0.01035774, 0.01035774, 0.01035774, 0.01035774, 0.01035774,
            0.01035774, 0.01035774, 0.00783755, 0.00767712, 0.00767712,
            0.01404397, 0., 0., 0., 0.,
            0., 0., 0.99896296], dtype=np.float32)},
        (2018, 2020): {'means': np.array([
            1.92527729e+03,  2.90639462e+03,  1.79803198e+03,  4.17896804e+03,
            2.16614586e+03,  4.31776715e-01,  3.53074797e+00,  7.87432855e-03,
            2.82827927e+02,  2.99377397e+02,  7.06674571e+01,  5.32125263e-03,
            6.85055885e+00,  1.44364776e-03,  1.51493429e+03, -1.69149999e+00,
            8.85224324e+00,  1.14009785e+01,  1.64494010e+00,  1.19060594e-01,
            1.84249891e+01,  5.34158917e-03,  1.48273628e+01], dtype=np.float32),
        'stds': np.array([
            1.13771183e+03, 1.61051542e+03, 1.86898895e+03, 2.22301358e+03,
            1.09801944e+03, 2.21079146e+00, 1.52217285e+00, 7.07369022e-01,
            7.47886968e+00, 8.45205917e+00, 1.93343283e+01, 2.17756746e-03,
            6.68841515e+00, 7.06107475e-01, 7.96954308e+02, 1.83049817e+00,
            3.29888042e+00, 3.30864516e+01, 1.19640682e+00, 5.41434581e-01,
            7.37697721e+00, 1.90372161e-03, 5.50970354e+00], dtype=np.float32),
        'missing_values': np.array([
            0.020201, 0.02021083, 0.02020913, 0.04801165, 0.04801215,
            0.01282664, 0.01282664, 0.01282664, 0.01282664, 0.01282664,
            0.01282664, 0.01282664, 0.01241417, 0.01219699, 0.01219699,
            0.01726448, 0., 0., 0., 0.,
            0., 0., 0.99835655], dtype=np.float32)},
        (2018, 2021): {'means': np.array([
            1.89013022e+03,  2.96940208e+03,  1.80725363e+03,  4.50489531e+03,
            2.31480127e+03,  5.02559433e-01,  3.45041794e+00,  1.27139445e-02,
            2.82806380e+02,  2.99358849e+02,  6.84029260e+01,  5.54077946e-03,
            7.71963056e+00,  1.48636202e-03,  1.51864017e+03, -2.87844549e+00,
            8.39885028e+00,  9.27182969e+00,  1.50171912e+00,  6.32564205e-02,
            1.85506226e+01,  5.47539956e-03,  1.47133578e+01], dtype=np.float32),
        'stds': np.array([
            1.17527575e+03, 1.68449116e+03, 1.98719945e+03, 2.24481517e+03,
            1.12442274e+03, 2.26782957e+00, 1.36958104e+00, 7.07592235e-01,
            6.49979053e+00, 7.63064073e+00, 1.89506910e+01, 1.99923623e-03,
            7.10935154e+00, 7.06734174e-01, 7.28691576e+02, 1.61764224e+00,
            3.42863222e+00, 3.32588535e+01, 1.03051749e+00, 4.07503144e-01,
            6.56328211e+00, 1.67899092e-03, 5.49708016e+00], dtype=np.float32),
        'missing_values': np.array([
            0.06689008, 0.06688501, 0.06688439, 0.09876232, 0.09876266,
            0.00874745, 0.00874745, 0.00874745, 0.00874745, 0.00874745,
            0.00874745, 0.00874745, 0.00410403, 0.00401978, 0.00401978,
            0.01257587, 0., 0., 0., 0.,
            0., 0., 0.99811762], dtype=np.float32)},
        (2019, 2020): {'means': np.array([
            1.93210708e+03,  2.91370171e+03,  1.83706324e+03,  4.08953980e+03,
            2.10045986e+03,  4.94984735e-01,  3.53618060e+00,  4.07106577e-03,
            2.83407079e+02,  2.99610979e+02,  6.97037281e+01,  5.45841688e-03,
            6.50879203e+00,  1.75418619e-03,  1.44461686e+03, -8.59165131e-01,
            9.00657070e+00,  8.83954683e+00,  1.67522409e+00,  1.24404141e-01,
            1.86144498e+01,  5.38086557e-03,  1.48645348e+01], dtype=np.float32),
        'stds': np.array([
            1.12339811e+03, 1.63906793e+03, 1.88834784e+03, 2.14790877e+03,
            1.03149639e+03, 2.25637560e+00, 1.53103092e+00, 7.07038206e-01,
            8.01997924e+00, 8.95414067e+00, 2.04390438e+01, 2.30770483e-03,
            6.44091576e+00, 7.05879558e-01, 7.75084680e+02, 1.75367983e+00,
            3.19309720e+00, 1.77037349e+01, 1.21298008e+00, 5.50726106e-01,
            7.87039317e+00, 2.06457111e-03, 5.50900718e+00], dtype=np.float32),
        'missing_values': np.array([
            0.02618726, 0.02620377, 0.02620163, 0.0630706, 0.06307114,
            0.01363767, 0.01363767, 0.01363767, 0.01363767, 0.01363767,
            0.01363767, 0.01363767, 0.01766835, 0.01737764, 0.01737764,
            0.01816856, 0., 0., 0., 0.,
            0., 0., 0.99843441], dtype=np.float32)},
        (2019, 2021): {'means': np.array([
            1.88147144e+03,  3.00731959e+03,  1.85633292e+03,  4.56103236e+03,
            2.31152573e+03,  5.99824717e-01,  3.42193092e+00,  1.07399193e-02,
            2.83410161e+02,  2.99599435e+02,  6.64147507e+01,  5.77996581e-03,
            7.73302100e+00,  1.83252047e-03,  1.44600073e+03, -2.50143930e+00,
            8.36983739e+00,  5.62677766e+00,  1.47282063e+00,  4.51032926e-02,
            1.88067298e+01,  5.57455804e-03,  1.46989234e+01], dtype=np.float32),
        'stds': np.array([
            1.17682375e+03, 1.74877733e+03, 2.06062268e+03, 2.16947243e+03,
            1.06252645e+03, 2.33839504e+00, 1.31010010e+00, 7.07336368e-01,
            6.73165592e+00, 7.86586044e+00, 1.99876580e+01, 2.07340496e-03,
            7.04371793e+00, 7.06763509e-01, 6.70725589e+02, 1.41847911e+00,
            3.37572336e+00, 1.66397698e+01, 9.72962912e-01, 3.50080317e-01,
            6.80142805e+00, 1.77658924e-03, 5.49124080e+00], dtype=np.float32),
        'missing_values': np.array([
            0.09324284, 0.09323855, 0.0932379, 0.13653821, 0.13653853,
            0.00786934, 0.00786934, 0.00786934, 0.00786934, 0.00786934,
            0.00786934, 0.00786934, 0.00616083, 0.00605492, 0.00605492,
            0.01153656, 0., 0., 0., 0.,
            0., 0., 0.9980986], dtype=np.float32)},
        (2020, 2021): {'means': np.array([
            1.87555188e+03,  2.94560341e+03,  1.83152387e+03,  4.41762912e+03,
            2.23468240e+03,  5.14182028e-01,  3.53935693e+00,  1.14206561e-02,
            2.83397348e+02,  2.99592787e+02,  6.94810785e+01,  5.52133545e-03,
            7.50684449e+00,  1.02162830e-03,  1.45767808e+03, -2.57062415e+00,
            8.48965669e+00,  3.19498037e+00,  1.56481648e+00,  8.62800254e-02,
            1.86874840e+01,  5.34801398e-03,  1.47533915e+01], dtype=np.float32),
        'stds': np.array([
            1.16304777e+03, 1.68253524e+03, 1.97741772e+03, 2.19727741e+03,
            1.05329146e+03, 2.32933992e+00, 1.46831775e+00, 7.07383262e-01,
            7.38007815e+00, 8.37759275e+00, 1.99833888e+01, 2.15409146e-03,
            6.88269677e+00, 7.06702488e-01, 7.35189217e+02, 1.53372656e+00,
            3.44095035e+00, 8.56098760e+00, 1.13319484e+00, 4.04395714e-01,
            7.25420162e+00, 1.80867836e-03, 5.49436248e+00], dtype=np.float32),
        'missing_values': np.array([
            0.06439008, 0.06439824, 0.06439709, 0.1217506, 0.12175085,
            0.01114208, 0.01114208, 0.01114208, 0.01114208, 0.01114208,
            0.01114208, 0.01114208, 0.01120559, 0.01102537, 0.01102537,
            0.01554857, 0., 0., 0., 0.,
            0., 0., 0.99780835], dtype=np.float32)}}

    years_tuple = tuple(training_years)
    means = stats_per_training_year_combo[years_tuple]["means"]
    stds = stats_per_training_year_combo[years_tuple]["stds"]
    missing_values = stats_per_training_year_combo[years_tuple]["missing_values"]

    # Zero out means and stds for degree-based features and the categorical land cover type variable
    features_to_not_standardize = get_indices_of_degree_features() + [16]

    means[features_to_not_standardize] = 0
    stds[features_to_not_standardize] = 1

    return means, stds, missing_values


def get_indices_of_degree_features():
    """
    :return: Indices of features that take values in [0,360] and thus will be transformed via sin

    """
    return [7, 13, 19]
