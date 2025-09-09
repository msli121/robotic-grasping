# ================Windows Cornell 数据集======================

# cornell baseline
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --batch-size 8 --split 0.8

# cornell only goa
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --goa 1

# cornell only aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --aff 1

# cornell goa + aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --goa 1 --aff 1

# ================Windows Jacquard 数据集======================

# jacquard baseline
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --batch-size 8 --split 0.9

# jacquard unet
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1

# jacquard goa
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --goa 1

# jacquard unet+goa
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --goa 1

# jacquard unet+aff
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --aff 1

# Jacquard upconv + unet + goa
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --upconv 1 --unet 1 --goa 1

# jacquard unet+goa+aff
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --goa 1 --aff 1

# jacquard FPN + GOA + AFF
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --fpn 1 --goa 1 --aff 1

# ================Linux Cornell 数据集======================
# conda activate grcnn
python train_network.py --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8

# cornell baseline
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8
# cornell upconv
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --upconv 1
# cornell unet
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --unet 1
# cornell goa
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --goa 1
# cornell aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --aff 1
# cornell upconv + unet
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --upconv 1 --unet 1
# cornell upconv + unet + goa
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --upconv 1 --unet 1 --goa 1
# cornell upconv + unet + goa + aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path /root/datasets/cornell_grasp --description training_cornell --input-size 224 --split 0.8 --use-dropout 1 --batch-size 8 --upconv 1 --unet 1 --goa 1 --aff 1

# ================Linux Jacquard 数据集======================
# Jacquard baseline
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16
# Jacquard upconv
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --upconv 1
# Jacquard unet
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --unet 1
# Jacquard goa
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --goa 1
# Jacquard aff
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --aff 1
# Jacquard upconv + unet
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --upconv 1 --unet 1
# Jacquard upconv + unet + goa
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --upconv 1 --unet 1 --goa 1
# Jacquard upconv + unet + goa + aff
conda activate grcnn && python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path /root/datasets/Jacquard --description training_Jacquard --input-size 300 --split 0.9 --use-dropout 1 --batch-size 16 --upconv 1 --unet 1 --goa 1 --aff 1

