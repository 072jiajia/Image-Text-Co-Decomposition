CUDA_VISIBLE_DEVICES=0,1,2,3 python -m torch.distributed.run \
    --nproc_per_node=auto \
     --master_port 5052 \
    main.py \
        --method-name image_text_co_decomposition \
        --resume checkpoint/checkpoint.pth \
        --eval
