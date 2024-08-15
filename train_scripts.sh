# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --data.root_dir=/home/aamer98/projects/def-ebrahimi/aamer98/data/wildfirespreadTS_unitTest

# wildfire
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --model.pretrained_res=5.625deg

# SeasFire
python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.devices=1 --model.pretrained_res=5.625deg --data.root_dir=/home/aamer98/projects/def-ebrahimi/aamer98/data/seasfire/SeasFireCube_coarse.zarr
python src/SeasFire/train.py --config configs/SeasFire_fine.yaml --trainer.devices=1 --model.pretrained_res=5.625deg --data.root_dir=/home/aamer98/projects/def-ebrahimi/aamer98/data/seasfire/SeasFireCube_v3.zarr

# GreekFire
python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.devices=1 --model.pretrained_res=5.625deg