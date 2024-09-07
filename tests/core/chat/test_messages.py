import unittest

from dacite import from_dict

from axolotl.core.chat.format.chatml import format_message
from axolotl.core.chat.messages import Chats, ChatFormattedChats

chat_msgs = {
    "conversation": [
        {
            "role": "system",
            "content": [
                {"type": "text", "value": "You are a helpful assistant."},
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "value": "What's the stock price of Apple?"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {"type": "tool_call", "value": {"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}},
            ],
        },
        {
            "role": "tool",
            "content": [
                {"type": "tool_response", "value": ""},
            ],
        }
    ]
}
chat_msgs_as_obj = from_dict(data_class=Chats, data=chat_msgs)

class TestMessagesCase(unittest.TestCase):
    def test_tool_call_stringify(self):
        self.assertEqual('{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}', str(chat_msgs_as_obj.conversation[2].content[0].value))

    def test_chatml_formatted_wrapper(self):
        chat_msg_formatted = ChatFormattedChats(conversation=chat_msgs_as_obj.conversation, formatter=format_message)
        target_chatml = """<|im_start|>system
You are a helpful assistant.<|im_end|>
<|im_start|>user
What's the stock price of Apple?<|im_end|>
<|im_start|>assistant
<tool_call>{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}</tool_call><|im_end|>\n"""
        self.assertEqual(target_chatml, str(chat_msg_formatted))

    def test_chatml_formatting_tool_call(self):
        target_chatml_turn2 = """<|im_start|>assistant\n<tool_call>{"name": "get_stock_price", "arguments": {"symbol": "AAPL"}}</tool_call><|im_end|>\n"""
        self.assertEqual(target_chatml_turn2, str(format_message(chat_msgs_as_obj.conversation[2])))

if __name__ == '__main__':
    unittest.main()
