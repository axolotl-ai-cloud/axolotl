"""
tests for jinja_template_analyzer
"""

import logging

import pytest

from axolotl.prompt_strategies.jinja_template_analyzer import JinjaTemplateAnalyzer

logging.basicConfig(level=logging.DEBUG)
LOG = logging.getLogger("axolotl")


class TestJinjaTemplateAnalyzer:
    """
    tests for jinja_template_analyzer
    """

    def test_basic_variable_extraction(self, basic_jinja_template_analyzer):
        """Test that all top-level variables are correctly extracted."""
        LOG.info("Testing with train_on_inputs=True")

        variables = basic_jinja_template_analyzer.get_template_variables()
        expected_vars = {"messages", "add_generation_prompt", "eos_token", "message"}
        assert set(variables.keys()) == expected_vars

    def test_mixtral_variable_extraction(self, mistral_jinja_template_analyzer):
        """Test that all top-level variables are correctly extracted."""
        LOG.info("Testing with train_on_inputs=True")

        variables = mistral_jinja_template_analyzer.get_template_variables()
        expected_vars = {
            "messages",
            "content",
            "eos_token",
            "message",
            "tools",
            "system_message",
            "loop_messages",
            "ns",
            "tool_call",
            "tool",
            "loop",
            "bos_token",
            "raise_exception",
        }
        assert set(variables.keys()) == expected_vars
        message_vars = variables["message"]
        assert message_vars == {"role", "content", "tool_calls", "tool_call_id"}

    def test_message_property_access(self, basic_jinja_template_analyzer):
        """Test that properties accessed on 'message' variable are correctly identified."""
        LOG.info("Testing message property access")

        variables = basic_jinja_template_analyzer.get_template_variables()
        assert "messages" in variables
        assert "message" in variables
        assert "role" in variables["message"]
        assert "content" in variables["message"]

    def test_detailed_analysis(self, basic_jinja_template_analyzer):
        """Test the detailed analysis of variable usage."""
        LOG.info("Testing detailed analysis")

        analysis = basic_jinja_template_analyzer.analyze_template()

        assert analysis["messages"]["is_iterated"] is True
        assert "role" in analysis["message"]["accessed_properties"]
        assert "content" in analysis["message"]["accessed_properties"]

        assert analysis["add_generation_prompt"]["is_conditional"] is True
        assert len(analysis["add_generation_prompt"]["accessed_properties"]) == 0

        assert not analysis["eos_token"]["is_iterated"]
        assert len(analysis["eos_token"]["accessed_properties"]) == 0

    def test_nested_property_access(self):
        """Test handling of nested property access."""
        LOG.info("Testing nested property access")

        template = """{{ user.profile.name }}{{ user.settings['preference'] }}"""
        analyzer = JinjaTemplateAnalyzer(template)
        variables = analyzer.get_template_variables()

        assert "user" in variables
        assert "profile" in variables["user"]
        assert "settings" in variables["user"]

    def test_loop_variable_handling(self):
        """Test handling of loop variables and their properties."""
        LOG.info("Testing loop variable handling")

        template = """
        {% for item in items %}
            {{ item.name }}
            {% for subitem in item.subitems %}
                {{ subitem.value }}
            {% endfor %}
        {% endfor %}
        """
        analyzer = JinjaTemplateAnalyzer(template)
        analysis = analyzer.analyze_template()

        assert analysis["items"]["is_iterated"]
        assert "name" in analysis["item"]["accessed_properties"]
        assert "subitems" in analysis["item"]["accessed_properties"]

    def test_conditional_variable_usage(self):
        """Test detection of variables used in conditional statements."""
        LOG.info("Testing conditional variable usage")

        template = """
        {% if user.is_admin and config.debug_mode %}
            {{ debug_info }}
        {% endif %}
        """
        analyzer = JinjaTemplateAnalyzer(template)
        analysis = analyzer.analyze_template()

        assert analysis["user"]["is_conditional"]
        assert analysis["config"]["is_conditional"]
        assert "is_admin" in analysis["user"]["accessed_properties"]
        assert "debug_mode" in analysis["config"]["accessed_properties"]

    def test_complex_expressions(self):
        """Test handling of complex expressions and filters."""
        LOG.info("Testing complex expressions and filters")

        template = """
        {{ user.name | upper }}
        {{ messages | length > 0 and messages[0].content }}
        {{ data['key'].nested['value'] }}
        """
        analyzer = JinjaTemplateAnalyzer(template)
        variables = analyzer.get_template_variables()

        assert "user" in variables
        assert "name" in variables["user"]
        assert "messages" in variables
        assert "content" in variables["messages"]
        assert "data" in variables

    def test_basic_msg_vars(self, basic_jinja_template_analyzer):
        """Test that the basic message variables are correctly identified."""
        LOG.info("Testing basic message variables")

        variables = basic_jinja_template_analyzer.get_message_vars()
        assert variables == {"role", "content"}

    def test_mixtral_msg_vars(self, mistral_jinja_template_analyzer):
        """Test that the mixtral message variables are correctly identified."""
        LOG.info("Testing mixtral message variables")

        variables = mistral_jinja_template_analyzer.get_message_vars()
        assert variables == {"role", "content", "tool_calls", "tool_call_id"}


if __name__ == "__main__":
    pytest.main([__file__])
