# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_unitTest 

# wildfire
CUDA_VISIBLE_DEVICES=1 python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --model.pretrained_res=5.625deg

# SeasFire
CUDA_VISIBLE_DEVICES=1 python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.devices=1 --model.pretrained_res=5.625deg
CUDA_VISIBLE_DEVICES=1 python src/SeasFire/train.py --config configs/SeasFire_fine.yaml --trainer.devices=1 --model.pretrained_res=5.625deg


# GreekFire
CUDA_VISIBLE_DEVICES=0 python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.devices=1 --model.pretrained_res=5.625deg