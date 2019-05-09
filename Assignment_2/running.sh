cont=4
bash train.sh data/dataset/ model${cont}.pt data/GoogleNews-vectors-negative300.bin.gz $cont
bash test.sh data/eval_data.txt data/eval_data.txt.td model${cont}.pt $cont
python mrr.py data/eval_data.txt data/eval_data.txt.td ${cont}output.txt
python mymrr.py data/eval_data.txt data/eval_data.txt.td ${cont}output.txt
