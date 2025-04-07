#!/usr/bin/env bash

set -euo pipefail  # Exit on error, unset variables, and prevent pipe failures

function hf-convert-usage {
    echo -e "Usage: hf-convert"
    echo -e "\t\t--hf-path </path/to/local/hf/repo>"
    echo -e "\t\t--fairseq-path </path/to/fairseq/checkpoint>"
    echo -e "\t\t--size {base, large}"
    echo -e "\t\t[--dict </path/to/dict>]"
    echo -e "\t\t[--copy-fairseq-model]"
    echo -e "\t\t[--conv-bias]"
    echo -e "\nAuthor: https://huggingface.co/patrickvonplaten (https://huggingface.co/HfSpeechUtils)"
    echo "Adaptation: William N. Havard <william.havard@gmail.com>"
    return 1
}

function hf-convert {
    declare hf_name=""
    declare ckpt=""
    declare size=""
    declare dict=""
    declare curPath
    declare copyFairseqModel=0
    declare convBias="False"

    curPath=$(pwd)

    if [[ $# -eq 0 ]]; then
        hf-convert-usage
        return 1
    fi

    while [[ $# -gt 0 ]]; do
        case "$1" in
            -h|--help) hf-convert-usage; return 0 ;;
            --hf-path) 
                if [[ $# -gt 1 && $2 != -* ]]; then
                    hf_name="$2"; shift 2
                else
                    echo "Error: --hf-path requires an argument" >&2
                    return 1
                fi ;;
            --fairseq-path) 
                if [[ $# -gt 1 && $2 != -* ]]; then
                    ckpt="$2"; shift 2
                else
                    echo "Error: --fairseq-path requires an argument" >&2
                    return 1
                fi ;;
            --size) 
                if [[ $# -gt 1 && ($2 == "base" || $2 == "large") ]]; then
                    size="$2"; shift 2
                else
                    echo "Error: --size should be one of {base, large}" >&2
                    return 1
                fi ;;
            --copy-fairseq-model) copyFairseqModel=1; shift 1 ;;
            --dict) 
                if [[ $# -gt 1 && $2 != -* ]]; then
                    dict="$2"; shift 2
                else
                    dict=""
                fi ;;
            --conv-bias) convBias="True"; shift 1 ;;
            --) shift; break ;;
            -*) echo "Error: Invalid option: $1" >&2; hf-convert-usage; return 1 ;;
            *) echo "Error: Invalid argument: $1" >&2; hf-convert-usage; return 1 ;;
        esac
    done

    # Validate required arguments
    if [[ -z "$hf_name" || -z "$ckpt" || -z "$size" ]]; then
        echo "Error: --hf-path, --fairseq-path, and --size are required" >&2
        return 1
    fi

    # Ensure paths exist
    if [[ ! -d "$hf_name" ]]; then
        mkdir -p "$hf_name"
    fi
    if [[ ! -f "$ckpt" ]]; then
        echo "Error: Checkpoint file does not exist: $ckpt" >&2
        return 1
    fi
    if [[ -n "$dict" && ! -f "$dict" ]]; then
        echo "Error: Dictionary file does not exist: $dict" >&2
        return 1
    fi

    # Only copy dictionary if it exists
    if [[ -n "$dict" ]]; then
        mkdir -p "$curPath/data/temp"
        cp "$dict" "$curPath/data/temp/dict.ltr.txt"
        echo "Copied dictionary: $dict -> $curPath/data/temp/dict.ltr.txt"
    fi

    # Load config (safer to use LeBenchmark as we have the exact same architecture)
    python -c "from transformers import Wav2Vec2Config; \
                config = Wav2Vec2Config.from_pretrained('LeBenchmark/wav2vec2-FR-7K-${size}'); \
                config.conv_bias = ${convBias}; \
                config.save_pretrained('${hf_name}');"

    # Convert checkpoint
    if [[ -n "$dict" ]]; then
        python convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py \
            --pytorch_dump_folder "$hf_name" \
            --checkpoint_path "$ckpt" \
            --config_path "$hf_name/config.json" \
            --dict_path "$curPath/data/temp/dict.ltr.txt"
    else
        python convert_wav2vec2_original_pytorch_checkpoint_to_pytorch.py \
            --pytorch_dump_folder "$hf_name" \
            --checkpoint_path "$ckpt" \
            --config_path "$hf_name/config.json" \
            --not_finetuned
    fi

    # Copy Fairseq model if requested
    if [[ "$copyFairseqModel" -eq 1 ]]; then
        cp -f "$ckpt" "$hf_name"
    fi
}

hf-convert "$@"

