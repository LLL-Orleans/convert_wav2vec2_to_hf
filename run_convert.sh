#!/usr/bin/env bash
hf_name=${1}
ckpt=${2}
dict=${3}

curPath=$(pwd)

cp ${dict} ${curPath}/data/temp/dict.ltr.txt

# load a config that is equal to the config of the model you wish to convert
python -c "from transformers import Wav2Vec2Config; config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-base'); config.save_pretrained('./');"

# pretrained only
eval "python ../transformers/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ./config.json --not_finetuned"
# fine-tuned
#eval "python ../transformers/src/transformers/models/wav2vec2/convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ./config.json --dict_path ${curPath}/data/temp/dict.ltr.txt"
