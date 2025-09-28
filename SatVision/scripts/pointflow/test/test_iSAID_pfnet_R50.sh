#!/usr/bin/env bash
echo "Running inference on" ${1}
echo "Saving Results :" ${2}
python3 eval.py \
	--dataset iSAID \
    --arch network.pointflow_resnet_with_max_avg_pool.DeepR50_PF_maxavg_deeply \
    --inference_mode  whole \
    --single_scale \
    --scales 1.0 \
    --split test \
    --cv_split 0 \
    --maxpool_size 14 \
    --avgpool_size 9 \
    --edge_points 128 \
    --match_dim 64 \
    --resize_scale 896 \
    --mode semantic \
    --no_flip \
    --ckpt_path ${2} \
    --snapshot ${1} \
    --dump_images \
    --exp_name test \
    #--test_mode

#sh ./scripts/pointflow/test/test_iSAID_pfnet_R50.sh /home/juanan/TFG/snapshot/pfnet_r50_iSAID.pth /home/juanan/TFG/ckpt/
#curl -F "image=@/home/juanan/TFG/PFSegNets/data/iSAID/test2/images/imageP0002.png" http://localhost:5000/upload