import numpy as np
import torch
import argparse
import os
import time

from text import text_to_sequence, VOCAB_DICT
import commons
import models
import utils


device = torch.device("cuda:0")

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, default='너무 보고싶었어요')
parser.add_argument("--checkpoint_path", type=str, default="ckpt/no_blank")
parser.add_argument("--mel_save_path", type=str, default="test1.npy")
args = parser.parse_args()

os.makedirs("inference_output", exist_ok=True)
hps = utils.get_hparams_from_dir(args.checkpoint_path)
checkpoint_path = utils.latest_checkpoint_path(args.checkpoint_path)

model = models.FlowGenerator(
    len(VOCAB_DICT.keys()) + getattr(hps.data, "add_blank", False),
    out_channels=hps.data.n_mel_channels,
    **hps.model).to(device)

utils.load_checkpoint(checkpoint_path, model)
model.decoder.store_inverse() # do not calcuate jacobians for fast decoding
_ = model.eval()

start_time = time.time()
if getattr(hps.data, "add_blank", False):
    text_norm = text_to_sequence(args.text.strip())
    text_norm = commons.intersperse(text_norm, len(VOCAB_DICT.keys()))
else:
    text = " " + args.text.strip() + " "
    text_norm = text_to_sequence(text)

sequence = np.array(text_norm)[None, :]

x_tst = torch.autograd.Variable(torch.from_numpy(sequence)).long().to(device)
x_tst_lengths = torch.tensor([x_tst.shape[1]]).to(device)

with torch.no_grad():
    noise_scale = .667
    length_scale = 1.0
    (y_gen_tst, *_), *_, (attn_gen, *_) = model(x_tst, x_tst_lengths, gen=True, noise_scale=noise_scale, length_scale=length_scale)

np.save(os.path.join("inference_output", args.mel_save_path), y_gen_tst.cpu().detach().numpy())

end_time = time.time() - start_time
print(f"Elapsed time {end_time}s")
print("Save mel-spectrogram ...")
