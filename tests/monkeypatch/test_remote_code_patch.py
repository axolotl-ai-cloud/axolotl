"""
Test to make sure the model is laodable whn patching the loader
"""


class TestRemoteCodePatch:
    """
    Test Case to make sure the model is laodable whn patching the loader
    """

    def test_remote_code_patch(self):
        from axolotl.utils.models import load_model_config

        # simply verify the loading works when patched
        load_model_config("deepseek-ai/DeepSeek-V2.5", trust_remote_code=True)
