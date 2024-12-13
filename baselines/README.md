# Baselines Training
Here we provide instructions on how to train your own baselines. Firstly, you need to git pull the data from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main) to your local folder. 

## GA

```python
python GA.py \
	--model_id llava-hf/llava-1.5-7b-hf \
	--vanilla_dir [Vanilla Model Path] \
	--data_split_dir [Your local data Path that you downloaded from [HF](https://huggingface.co/MLLMMU/baseline_train_split/tree/main)] \
	--forget_split_ratio 5 \
	--save_dir [GA Saved Path] \
	--batch_size 4 \
	--lr 2e-5 \
	--num_epochs 1 \
	--max_length 384
```
