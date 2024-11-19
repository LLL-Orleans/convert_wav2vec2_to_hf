#!/usr/bin/env bash

function hf-convert-usage
{
    echo -e "Usage: hf-convert";
      echo -e "\t\t--hf-path </path/to/local/hf/repo>";
      echo -e "\t\t--fairseq-path </path/to/fairseq/checkpoint>";
      echo -e "\t\t--size {base, large}";
      echo -e "\t\t[--dict </path/to/dict>]";
      echo -e "\t\t[--copy-fairseq-model]";
      echo -e "\t\t[--conv-bias]";
    echo -e "\nAuthor: https://huggingface.co/patrickvonplaten (https://huggingface.co/HfSpeechUtils)";
    echo "Adaptation: William N. Havard <william.havard@gmail.com>";
    return 1;
}

function hf-convert
{
  declare hf_name;
  declare ckpt;
  declare size;
  declare dict;
  declare curPath;
  declare copyFairseqModel;
  declare convBias;

  curPath=$(pwd);
  dict="0";
  copyFairseqModel=0;
  convBias="False";  # Default to False

  if [[ $# -eq 0 ]]; then
      hf-convert-usage;
      return 1;
  fi

  while [[ $# -gt 0 ]]; do
    case "$1" in
      -h|-_usage|-\?) hf-convert-usage; return 0;;
      --hf-path) if [[ $# -gt 1 && $2 != -* ]]; then
          hf_name=$2; shift 2
          else
          echo "Error: --name requires an argument" 1>&2
          return 1
          fi;;
      --fairseq-path) if [[ $# -gt 1 && $2 != -* ]]; then
          ckpt=$2; shift 2
          else
          echo "Error: --checkpoint requires an argument" 1>&2
          return 1
          fi;;
      --size) if [[ $# -gt 1 && $2 != -* && ($2 = 'base' || $2 = 'large') ]]; then
          size=$2; shift 2
          else
          echo "Error: --size should be one of {base, large}" 1>&2
          return 1
          fi;;
      --copy-fairseq-model) copyFairseqModel=1; shift 1;;
      --dict) if [[ $# -gt 1 && $2 != -* ]]; then
          dict=$2; shift 2
          else
          dict="0";
          fi;;
      --conv-bias) convBias="True"; shift 1;;
      --) shift; break;;
      -*) echo "Error: Invalid option: $1" 1>&2; hf-convert-usage; return 1;;
      *) echo "Error: Invalid option: $1" 1>&2; hf-convert-usage; return 1;;
    esac
  done

  # Only copy if there is something to be copied
  if [[ dict -ne "0" ]]; then
    cp ${dict} ${curPath}/data/temp/dict.ltr.txt
  fi

  # Load a config that is equal to the config of the model you wish to convert
  mkdir -p ${hf_name}
  python -c "from transformers import Wav2Vec2Config; config = Wav2Vec2Config.from_pretrained('facebook/wav2vec2-${size}'); config.conv_bias = ${convBias}; config.save_pretrained('${hf_name}');"

  if [[ dict -eq "0" ]]; then
    # Pretrained only
    python convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ${hf_name}/config.json --not_finetuned
  else
    # Fine-tuned
    python convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py --pytorch_dump_folder ${hf_name} --checkpoint_path ${ckpt} --config_path ${hf_name}/config.json --dict_path ${curPath}/data/temp/dict.ltr.txt
  fi

  if [[ copyFairseqModel -eq 1 ]]; then
    cp ${ckpt} ${hf_name};
  fi
}

hf-convert "$@"

