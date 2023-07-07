cd stable-dreamfusion

python main.py -O --text "$1" --workspace ../model_gen_$2 --iters $4

python main.py -O --text "$1" --workspace ../model_gen_$2/finetune --dmtet --iters $4 --init_with ../model_gen_$2/checkpoints/df.pth

python main.py -O --text "$1" --workspace ../model_gen_$2/finetune --dmtet --iters $4 --test --save_mesh

cd ..

python headless_generator.py "$2" $3