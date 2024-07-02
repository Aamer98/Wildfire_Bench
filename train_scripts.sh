# wildfire unit test
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_unitTest --data.predict_range=24 --data.batch_size=2 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# wildfire
python src/WildfireSpreadTS/train.py --config configs/WildfireSpreadTS.yaml --trainer.strategy=ddp --trainer.devices=1 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/wildfirespreadTS_hdf5 --data.predict_range=24 --data.batch_size=2 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# SeasFire
CUDA_VISIBLE_DEVICES=1 python src/SeasFire/train.py --config configs/SeasFire.yaml --trainer.strategy=ddp --trainer.devices=1 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/seasfire/SeasFireCube_coarse.zarr --data.predict_range=192 --data.batch_size=2 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5

# GreekFire
python src/GreeceFire/train.py --config configs/GreeceFire.yaml --trainer.strategy=ddp --trainer.devices=1 --trainer.max_epochs=50 --data.root_dir=/home/as26840@ens.ad.etsmtl.ca/data/greekfire/datasets_grl --data.predict_range=24 --data.batch_size=2 --model.pretrained_path='https://huggingface.co/tungnd/climax/resolve/main/1.40625deg.ckpt' --model.lr=5e-7 --model.beta_1="0.9" --model.beta_2="0.99" --model.weight_decay=1e-5