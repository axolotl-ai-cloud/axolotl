import unittest

from axolotl.prompters import AlpacaPrompter, PromptStyle


class AlpacaPrompterTest(unittest.TestCase):
    def test_prompt_style_w_none(self):
        prompter = AlpacaPrompter(prompt_style=None)
        res = next(prompter.build_prompt("tell me a joke"))
        # just testing that it uses instruct style
        assert "### Instruction:" in res

    def test_prompt_style_w_instruct(self):
        prompter = AlpacaPrompter(prompt_style=PromptStyle.instruct.value)
        res = next(
            prompter.build_prompt("tell me a joke about the following", "alpacas")
        )
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" in res
        assert "alpacas" in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res
        res = next(prompter.build_prompt("tell me a joke about the following"))
        assert "Below is an instruction" in res
        assert "### Instruction:" in res
        assert "### Input:" not in res
        assert "### Response:" in res
        assert "USER:" not in res
        assert "ASSISTANT:" not in res

    def test_prompt_style_w_chat(self):
        prompter = AlpacaPrompter(prompt_style=PromptStyle.chat.value)
        res = next(
            prompter.build_prompt("tell me a joke about the following", "alpacas")
        )
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "alpacas" in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res
        res = next(prompter.build_prompt("tell me a joke about the following"))
        assert "Below is an instruction" in res
        assert "### Instruction:" not in res
        assert "### Input:" not in res
        assert "### Response:" not in res
        assert "USER:" in res
        assert "ASSISTANT:" in res
