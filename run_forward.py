#!/usr/bin/env python3
import datasets
import fairseq
import torch
import os

import soundfile as sf
from datasets import load_dataset
import sys
from shutil import copyfile
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor, Wav2Vec2Model, Wav2Vec2CTCTokenizer, Wav2Vec2FeatureExtractor

hf_path = str(sys.argv[1])
fairseq_wav2vec2_path = str(sys.argv[2])
finetuned = bool(int(sys.argv[3]))


if finetuned:
    processor = Wav2Vec2Processor.from_pretrained(hf_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task(
        [fairseq_wav2vec2_path], arg_overrides={"data": "../add_wav2vec/data/temp"}
    )
    hf_model = Wav2Vec2ForCTC.from_pretrained(hf_path)
else:
    processor = Wav2Vec2FeatureExtractor.from_pretrained(hf_path)
    model, cfg, task = fairseq.checkpoint_utils.load_model_ensemble_and_task([fairseq_wav2vec2_path])
    hf_model = Wav2Vec2Model.from_pretrained(hf_path)

model = model[0]
model.eval()


def test_feature_extractor(hf_feat_extractor, fsq_feat_extract, example_wav):
    # set hf_feat_extractor.output to dummy
    fsq_output = fsq_feat_extract(example_wav)
    hf_output = hf_feat_extractor(example_wav)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    assert torch.allclose(hf_output, fsq_output, atol=1e-3)


def test_full_encoder(hf_model, fsq_model, example_wav, attention_mask):
    fsq_output = fsq_model(example_wav, padding_mask=attention_mask.ne(1), mask=False, features_only=True)["x"]
    hf_output = hf_model(example_wav, attention_mask=attention_mask)[0]

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    assert torch.allclose(hf_output, fsq_output, atol=1e-2)


def test_full_model(hf_model, fsq_model, example_wav, attention_mask):
    fsq_output = fsq_model(source=example_wav, padding_mask=attention_mask.ne(1))["encoder_out"]
    hf_output = hf_model(example_wav, attention_mask=attention_mask)[0].transpose(0, 1)

    assert (
        hf_output.shape == fsq_output.shape
    ), f"Shapes don't match. Got {hf_output.shape} for HF and {fsq_output.shape} for fsq"
    assert torch.allclose(hf_output, fsq_output, atol=1e-2)


def test_loss(hf_model, fsq_model, example_wav, attention_mask, target):
    from fairseq.criterions.ctc import CtcCriterion, CtcCriterionConfig
    from fairseq.tasks.audio_pretraining import AudioPretrainingConfig, AudioPretrainingTask
    audio_cfg = AudioPretrainingConfig(labels="ltr", data="./data")
    task = AudioPretrainingTask.setup_task(audio_cfg)
    ctc = CtcCriterion(CtcCriterionConfig(), task)
    fsq_model.train()

    labels_dict = processor.tokenizer(target, padding="longest", return_tensors="pt")
    labels = labels_dict.input_ids
    target_lengths = labels_dict.attention_mask.sum(-1)

    sample = {
        "net_input": {
            "source": example_wav,
            "padding_mask": attention_mask.ne(1),
        },
        "target": labels,
        "target_lengths": target_lengths,
        "id": torch.zeros((1,)),
    }

    loss, _, _ = ctc(fsq_model, sample)

    labels = labels_dict.attention_mask * labels + (1 - labels_dict.attention_mask) * -100

    hf_model.config.ctc_loss_reduction = "mean"
    hf_loss = hf_model(example_wav, attention_mask=attention_mask, labels=labels).loss

    print("Loss", loss)
    print("Hf loss", hf_loss)


def test_all(example_wav, attention_mask):
    with torch.no_grad():
        if finetuned:
            test_feature_extractor(
                hf_model.wav2vec2.feature_extractor, model.w2v_encoder.w2v_model.feature_extractor, example_wav
            )
        else:
            test_feature_extractor(
                hf_model.feature_extractor, model.feature_extractor, example_wav
            )
    print("Succeded feature extractor Test")

    with torch.no_grad():
        # IMPORTANT: It is assumed that layer_norm_first is FALSE
        # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
        # Adapt if necessary
        if finetuned:
            test_full_encoder(hf_model.wav2vec2, model.w2v_encoder.w2v_model, example_wav, attention_mask)
        else:
            test_full_encoder(hf_model, model, example_wav, attention_mask)
    print("Succeded full encoder test")

    if finetuned:
        with torch.no_grad():
            # IMPORTANT: It is assumed that layer_norm_first is FALSE
            # This is the case for `wav2vec_small_960h.pt`, but might not be for all models
            # Adapt if necessary
            test_full_model(hf_model, model, example_wav, attention_mask)
        print("Succeded full model test")


dummy_speech_data = datasets.load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")


def map_to_array(batch):
    speech_array, _ = sf.read(batch["file"])
    batch["speech"] = speech_array
    return batch

def map_to_array_mp3(batch, i):
    speech_array, sr = sf.read(f"/home/patrick/hugging_face/add_wav2vec/common_voice/cv-corpus-6.1-2020-12-11/nl/converted/sample_{i}.wav")
    batch["speech"] = speech_array
    batch["sampling_rate"] = sr
    return batch


dummy_speech_data = dummy_speech_data.map(map_to_array, remove_columns=["file"])
inputs = processor(dummy_speech_data[:3]["speech"], return_tensors="pt", padding="longest", return_attention_mask=True)

transciption = dummy_speech_data[:3]["text"]

input_values = inputs.input_values
attention_mask = inputs.attention_mask

test_all(input_values, attention_mask)
#test_loss(hf_model, model, input_values, attention_mask, transciption)
