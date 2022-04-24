python main.py \
--device=gpus \
--gpus=0 \
--model=VGG \
--model-version=11 \
--dataset=TinyImageNet \
--train-bsz=50 \
--test-bsz=50 \
--lr=0.01 \
--lr-decay=1 \
--optim=SGD \
--epochs=50
