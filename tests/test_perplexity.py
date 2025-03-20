"""unit tests for perplexity eval callback"""

# pylint: disable=redefined-outer-name

from pytest import fixture
from transformers.models.auto.modeling_auto import AutoModelForCausalLM
from transformers.models.auto.tokenization_auto import AutoTokenizer

from axolotl.utils.callbacks.perplexity import Perplexity

MODEL_NAME = "HuggingFaceTB/SmolLM2-135M"


@fixture()
def metric(tokenizer):
    return Perplexity(tokenizer=tokenizer, max_seq_len=512)


@fixture()
def model():
    return AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True)


@fixture()
def tokenizer():
    tokenizer_ = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer_.add_special_tokens({"pad_token": "<|endoftext|>"})
    return tokenizer_


def test_perplexity_longer_than_stride(model, metric):
    # taken from https://huggingface.co/datasets/roneneldan/TinyStories
    sample_text = """
Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun. Beep was a healthy car because he always had good fuel. Good fuel made Beep happy and strong. One day, Beep was driving in the park when he saw a big tree. The tree had many leaves that were falling. Beep liked how the leaves fall and wanted to play with them. Beep drove under the tree and watched the leaves fall on him. He laughed and beeped his horn. Beep played with the falling leaves all day. When it was time to go home, Beep knew he needed more fuel. He went to the fuel place and got more healthy fuel. Now, Beep was ready to go fast and play again the next day. And Beep lived happily ever after.
One day, a little fish named Fin was swimming near the shore. He saw a big crab and wanted to be friends. "Hi, I am Fin. Do you want to play?" asked the little fish. The crab looked at Fin and said, "No, I don't want to play. I am cold and I don't feel fine." Fin felt sad but wanted to help the crab feel better. He swam away and thought of a plan. He remembered that the sun could make things warm. So, Fin swam to the top of the water and called to the sun, "Please, sun, help my new friend feel fine and not freeze!" The sun heard Fin's call and shone its warm light on the shore. The crab started to feel better and not so cold. He saw Fin and said, "Thank you, little fish, for making me feel fine. I don't feel like I will freeze now. Let's play together!" And so, Fin and the crab played and became good friends.
"""
    result = metric.compute(model, [sample_text])
    ppl = result["score"]
    assert round(ppl, 2) == 7.41


def test_perplexity_short(model, metric):
    # taken from https://huggingface.co/datasets/roneneldan/TinyStories
    sample_text = "Once upon a time, there was a little car named Beep. Beep loved to go fast and play in the sun."
    result = metric.compute(model, [sample_text])
    ppl = result["score"]
    assert round(ppl, 2) == 10.33
