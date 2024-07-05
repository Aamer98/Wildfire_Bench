# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_unitTest 

# wildfire
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.devices=1

# SeasFire
python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.devices=1

# GreekFire
python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.devices=1