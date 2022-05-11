python main.py \
--device=gpus \
--gpus=0 \
--model-type=rnn \
--num-step=128 \
--model=BiResidualRNN \
--model-version=11 \
--dataset=HAR \
--train-bsz=128 \
--test-bsz=128 \
--lr=0.0015 \
--lr-decay=1 \
--optim=Adam \
--epochs=100
