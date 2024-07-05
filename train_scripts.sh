# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_unitTest --data.predict_range=24 --data.batch_size=2

# wildfire
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=100 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_hdf5 --data.predict_range=24 --data.batch_size=2

# SeasFire
python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.devices=1

# GreekFire
python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.devices=1