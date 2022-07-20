GPU=$1
CUDA_VISIBLE_DEVICES=$GPU python main.py --checkpoint_dir=webvid \
	--dataset=webvid \
	--mc=64 \
	--epochs=3 \
	--lr=0.00005 \
	--qmax_words=0 \
	--amax_words=20 \
	--max_feats=32 \
	--batch_size=64 \
	--batch_size_val=64 \
	--num_thread_reader=16 \
	--mlm_prob=0.15 \
	--n_layers=1 \
	--embd_dim=512 \
	--ff_dim=1024 \
	--dropout=0.3 \
	--save_dir='./save_models/webvid/025/' \
	--seed=666 \

