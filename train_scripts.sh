# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_unitTest 

# wildfire
CUDA_VISIBLE_DEVICES=0 python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1

# SeasFire
CUDA_VISIBLE_DEVICES=0 python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.devices=1

# GreekFire
CUDA_VISIBLE_DEVICES=0 python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.devices=1