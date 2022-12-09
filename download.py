# In this file, we define download_model
# It runs during container build time to get model weights built into the container

# In this example: A Huggingface BERT model

from transformers import BloomTokenizerFast, BloomForCausalLM

def download_model():
    # do a dry run of loading the huggingface model, which will download weights
    model_name = "bigscience/bloom-560m"
    tokenizer = BloomTokenizerFast.from_pretrained(model_name)
    model = BloomForCausalLM.from_pretrained(model_name)

if __name__ == "__main__":
    download_model()