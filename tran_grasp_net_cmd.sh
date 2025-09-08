# baseline cornell
python train_network.py --network grconvnet3 --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell_grconvnet3 --use-dropout 1 --input-size 224 --split 0.8

# baseline jacquard
python train_network.py --network grconvnet3 --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard_grconvnet3 --use-dropout 1 --input-size 224 --split 0.9

# cornell 基线
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8

# cornell only goa
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --goa 1

# cornell only aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --aff 1

# cornell goa + aff
python train_network.py --network grconvnet_goa --dataset cornell --dataset-path D:\\datasets\\cornell_grasp --description training_cornell --use-dropout 1 --input-size 300 --split 0.8 --goa 1 --aff 1

# jacquard unet
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1

# jacquard goa
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --goa 1

# jacquard unet+goa
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --goa 1

# jacquard unet+aff
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --aff 1

# jacquard unet+goa+aff
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --unet 1 --goa 1 --aff 1

# jacquard FPN + GOA + AFF
python train_network.py --network grconvnet_goa --dataset jacquard --dataset-path D:\\datasets\\Jacquard --description training_Jacquard --use-dropout 1 --input-size 300 --split 0.9 --fpn 1 --goa 1 --aff 1
