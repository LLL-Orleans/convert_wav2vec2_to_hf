# Convert Fairseq Wav2Vec2 to HF

This repo has two scripts that can show how to convert a fairseq checkpoint to HF Transformers.

It's important to always check in a forward pass that the two checkpoints are the same. The procedure should be as follows:

1. Download original model
2. Create HF version of the model:
```
huggingface-cli repo create <name_of_model> --organization <org_of_model>
git clone https://huggingface.co/<org_of_model>/<name_of_model>
```
3. Convert the model
```
./run_convert.sh <name_of_model> <path/to/orig/checkpoint/> 0
```
The "0" means that checkpoint is **not** a fine-tuned one.
4. Verify that models are equal:
```
./run_forward.py <name_of_model> <path/to/orig/checkpoint/> 0
```

Check the scripts to better understand how they work or contact https://huggingface.co/patrickvonplaten