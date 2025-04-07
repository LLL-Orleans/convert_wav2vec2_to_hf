# Convert Fairseq wav2vec2 to HF

This code found in this repository was adapted from [this original HuggingFace repository](https://huggingface.co/HfSpeechUtils/convert_wav2vec2_to_hf).
This repository contains two scripts that convert a fairseq wav2vec2 checkpoint to HuggingFace 🤗 Transformers. 

## Procedure

1. Create a HF repo :
```
huggingface-cli repo create <model-name> --organization <org_of_model>
git clone https://huggingface.co/<org_of_model>/<name_of_model>
```
2. Convert the model
```
./run_convert.sh \
    --hf-path </path/to/local/hf/repo> \
    --fairseq-path </path/to/fairseq/checkpoint> \
    --size {base, large} \
    [--dict </path/to/dict>] \
    [--copy-fairseq-model]
```

3. Verify models behave equally
```
./run_forward.py \
    --hf-path </path/to/local/hf/repo> \
    --fairseq-path </path/to/fairseq/checkpoint> \
    [--finetuned </path/to/dict>]
```

4. Push to hub
```
huggingface-cli upload <your-org>/<your-model> </path/to/local/hf/repo>
```


## Changelog

[convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py](convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py) (originally from official [huggingface /transformers](https://github.com/huggingface/transformers/blob/779bc360ff4f3965a1ac29fdc02c43db7ede08c0/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py#L4)) was modified.

1. It correctly remaps :
* `wav2vec2.encoder.pos_conv_embed.conv.weight_g` to `wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original0`
* `wav2vec2.encoder.pos_conv_embed.conv.weight_v` to `wav2vec2.encoder.pos_conv_embed.conv.parametrizations.weight.original1`

The current version of script should (not tested) also be able to correctly handle old `weight_g`/`weight_v`.

2. `sampling_rate` and `do_normalize` are both extracted from the fairseq's original configuration (e.g. `cfg['task']['sample_rate']`) instead of being guessed.

3. Creates `preprocessor_config.json` which the original didn't do for pre-trained (i.e. non-finetuned) models

4. The `test_loss` function [run_forward.py](run_forward.py) was corrected (didn't work for Fairseq before)