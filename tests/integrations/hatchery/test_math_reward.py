from axolotl.integrations.hatchery.rewards.math_reward import math_reward


def test_an_earlier_box_does_not_outrank_the_final_answer():
    completion = r"scratch \boxed{42} final \boxed{7}"
    # The final answer is 7. An earlier 42 must not score as a match for 42.
    assert math_reward(["q <|gold|>42<|/gold|>"], [completion]) == [0.0]
    assert math_reward(["q <|gold|>7<|/gold|>"], [completion]) == [1.0]
    assert math_reward(["q <|gold|>42<|/gold|>"], [r"answer \boxed{42}"]) == [1.0]
