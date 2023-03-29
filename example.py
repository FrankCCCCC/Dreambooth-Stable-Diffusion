import os

class_word = "toy-cat"
reg_data_root = "reg_data"
data_root = "fine_tune_img"
gpu = "0"
job_name = "ToyCatX1"
reg_weight = "5"
seed = 1

def train():
    cmd = f"python main.py --base configs/stable-diffusion/v1-finetune_unfrozen.yaml -t --actual_resume ckpt/sd-v1-4-full-ema.ckpt -n {job_name} --gpus {gpu}, --data_root {data_root}  --reg_data_root {reg_data_root} --class_word {class_word} --reg_weight {reg_weight} --no-test"
    os.system(cmd)
    
def test():
    ckpt = "logs/fine_tune_img2023-03-13T19-07-21_ToyCatX1/checkpoints/last.ckpt"
    outdir = "ToyCatX2"
    os.system(f'python scripts/stable_txt2img.py --ddim_eta 0.0 --n_samples 8 --n_iter 1 --scale 10.0 --ddim_steps 50  --ckpt {ckpt} --outdir {outdir} --prompt "a dog on the top of sks toy-cat" --seed {seed}')

if __name__ == '__main__':
    train()
    # test()