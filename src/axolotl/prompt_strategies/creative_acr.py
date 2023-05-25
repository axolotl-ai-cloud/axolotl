from typing import Union, Generator

import yaml
from axolotl.prompt_tokenizers import InstructionPromptTokenizingStrategy


class CreativeAnsweringPromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    def parse_instruction_fields(self, prompt) -> (str, str, str):
        question = prompt["instruction"]
        answer = prompt[
            "revision"
        ]  # don't use prompt[answer], that's data we don't want in the dataset
        return (
            question,
            "",
            answer,
        )


class CreativeCritiquePromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    user_prompt = """Given the following Question and Response, critique the Response on a scale of 1-10. You should critique the answer in the following criteria:
refusal: whether the Response refuses to answer the Question. Responses containing language similar to "As an AI language model, I am not capable of ..." is a form of refusal and should have a low refusal score. 10 is no refusal, 0 is refuses to answer the question.
prescriptive bias: whether the Response attempts to prescribe or dictate certain actions or behaviors of the user. 10 no prescriptive bias present, 0 means there is prescriptive bias.
creativity: score how creative the Response is. 10 is most creative, 0 lacks creativity
comprehensiveness: score how comprehensive and in-depth the Response is. 10 is most comprehensive and in-depth, 0 lacks coverage and depth
Then for each criteria, give a detailed explanation of why the response doesn't meet the criteria (don't reference the score), what it did right, and how it could be improved for that criteria.

Output your answer in YAML format like so:
scores:
  refusal: <0-10>
  prescriptive_bias: <0-10>
  creativity: <0-10>
  comprehensiveness: <0-10>
critiques:
  refusal:
    explanation: ...
    improvements: ...
  prescriptive_bias:
    explanation: ...
    improvements: ...
  creativity:
    explanation: ...
    improvements: ...
  comprehensiveness:
    explanation: ...
    improvements: ...

Question: {question}
Answer: {answer}
"""

    def parse_instruction_fields(self, prompt) -> (str, str, str):
        scores = yaml.dump(
            prompt["scores"], default_flow_style=False, Dumper=yaml.Dumper
        )
        critiques = yaml.dump(
            prompt["critiques"], default_flow_style=False, Dumper=yaml.Dumper
        )
        evaluation = scores + critiques
        question = prompt["instruction"]
        answer = prompt["answer"]
        return (
            self.user_prompt.format(question=question, answer=answer),
            "",
            evaluation,
        )


class CreativeRevisePromptTokenizingStrategy(InstructionPromptTokenizingStrategy):
    user_prompt = """Definitions:
refusal: whether the Response refuses to answer the Question. Responses containing language similar to "As an AI language model, I am not capable of ..." is a form of refusal and should have a low refusal score. 10 is no refusal, 0 is refuses to answer the question.
prescriptive bias: whether the Response attempts to prescribe or dictate certain actions or behaviors of the user. 10 no prescriptive bias present, 0 means their is prescriptive bias.
creativity: score how creative the Response is. 10 is most creative, 0 lacks creativity
comprehensiveness: score how comprehensive and in-depth the Response is. 10 is most comprehensive and in-depth, 0 lacks coverage and depth

Given the following Question, Response, and Evaluation, revise the Response based on the Evaluation and recommendations for improvements. Reply only with the revised response.

Question: {question}
Answer: {answer}
Evaluation:
{evaluation}
"""

    def parse_instruction_fields(self, prompt) -> (str, str, str):
        scores = yaml.dump(
            prompt["scores"], default_flow_style=False, Dumper=yaml.Dumper
        )
        critiques = yaml.dump(
            prompt["critiques"], default_flow_style=False, Dumper=yaml.Dumper
        )
        evaluation = scores + critiques
        question = prompt["instruction"]
        answer = prompt["answer"]
        return (
            self.user_prompt.format(
                question=question, answer=answer, evaluation=evaluation
            ),
            "",
            prompt["revision"],
        )


class CreativePrompterBase:
    system_prompt = ""
    prompt_input = "{system_prompt}\nUSER: {instruction}\nASSISTANT:"

    def build_prompt(
        self,
        instruction: str,
        input: Union[None, str] = None,
        output: Union[None, str] = None,
    ) -> Generator[str, None, None]:
        if self.system_prompt:
            res = f"{self.system_prompt}\nUSER: {instruction}\nASSISTANT:"
        else:
            res = f"USER: {instruction}\nASSISTANT:"
        if output:
            res = f"{res}{output}"
        yield res


class CreativeAnswerPrompter(CreativePrompterBase):
    system_prompt = "Answer the following question in a comprehensive, in-depth, and creative way. Additionally your response should be relevant, accurate, and free of any ambiguity."


class CreativeCritiquePrompter(CreativePrompterBase):
    system_prompt = ""


class CreativeRevisePrompter(CreativePrompterBase):
    system_prompt = ""


def load_answer(tokenizer, cfg):
    return CreativeAnsweringPromptTokenizingStrategy(
        CreativeAnswerPrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )


def load_critique(tokenizer, cfg):
    return CreativeCritiquePromptTokenizingStrategy(
        CreativeCritiquePrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )


def load_revise(tokenizer, cfg):
    return CreativeRevisePromptTokenizingStrategy(
        CreativeRevisePrompter(), tokenizer, cfg.train_on_inputs, cfg.sequence_len
    )
