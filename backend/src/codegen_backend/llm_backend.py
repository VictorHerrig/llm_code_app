from abc import ABC, abstractmethod
from typing import NamedTuple, Union

import torch
from openai import OpenAI
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)


class GenerationPrompt(NamedTuple):
    language: str
    original_prompt: str


class AdjustPrompt(NamedTuple):
    language: str
    original_prompt: str
    reference_code: str
    feedback_prompt: str


def build_prompt(prompt_object: Union[GenerationPrompt, AdjustPrompt]) -> str:
    """Builds a string user prompt from the provided prompt object. Infers Prompt type.

    Parameters
    ----------
    prompt_object: Union[GenerationPrompt, AdjustPrompt]
        Prompt NamedTuple subclass

    Returns
    -------
    user_prompt: str
    """
    if isinstance(prompt_object, GenerationPrompt):
        return (
            f"Language:\n{prompt_object.language}\n"
            f"Task:\n{prompt_object.original_prompt}\n"
            f"Solution code:\n"
        )
    elif isinstance(prompt_object, AdjustPrompt):
        return (
            f"Language:\n{prompt_object.language}\n"
            f"Task:\n{prompt_object.original_prompt}\n"
            f"Previous code:\n{prompt_object.reference_code}\n"
            f"User feedback:\n{prompt_object.feedback_prompt}\n"
            f"Solution code:"
        )
    else:
        raise ValueError(
            f"Wrong type of prompt object, expected GeneratePrompt or AdjustPrompt, but got {type(prompt_object)}"
        )


class LLMBackend(ABC):
    """Base class for LLM backend. Contains only interface."""

    # For the possibility of code interpretation
    supported_languages = [
        "python",
        "java",
        "c++",
        "c",
        "scala",
        "sql",
        "mysql",
        "postgresql",
    ]

    def generate_code(self, prompt_object: GenerationPrompt) -> str:
        """Generates a new piece of code snippet given a prompt.

        Parameters
        ----------
        prompt_object: GenerationPrompt
            Input prompt based on which to generate the code snippet.

        Returns
        -------
        generated_code: str
        """
        if prompt_object.language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language {prompt_object.language}")
        user_prompt = build_prompt(prompt_object)
        return self.call_backend(user_prompt, self.generate_system_prompt)

    def adjust_code(self, prompt_object: AdjustPrompt) -> str:
        """Generates a code snippet given an original prompt, a previous code snippet and user feedback on the previous
        code snippet.

        Parameters
        ----------
        prompt_object: AdjustPrompt
            Input prompt based on which to generate the code snippet.

        Returns
        -------
        generated_code: str
        """
        if prompt_object.language.lower() not in self.supported_languages:
            raise ValueError(f"Unsupported language {prompt_object.language}")
        user_prompt = build_prompt(prompt_object)
        print(user_prompt)
        return self.call_backend(user_prompt, self.adjust_system_prompt)

    @abstractmethod
    def call_backend(self, user_prompt: str, system_prompt: str) -> str:
        """Calls the backend method specific to a subclass.

        Parameters
        ----------
        user_prompt: str
            User-provided part of the prompt.
        system_prompt: str
            System message part of the prompt, provided most likely by this class' properties.

        Returns
        -------
        generated_code: str
        """
        raise NotImplementedError()

    @property
    def generate_system_prompt(self):
        return (
            "You are a helpful assistant that generates code to perform the requested  task in the requested "
            "programming language. Return only properly formatted code as succinct as possible while still completing "
            "the task. If the task is unclear, impossible, asks for anything other than a code snippet, or if you "
            "don't know the correct answer, return a message starting with 'ERROR:', followed by a brief explanation "
            "of why you cannot perform the task."
        )

    @property
    def adjust_system_prompt(self):
        return (
            "You are a helpful assistant that corrects code to perform the requested  task in the requested "
            "programming language according to the user-provided feedback. Return only properly formatted code with "
            "as few changes from the provided code as possible while still addressing the feedback. If the task, "
            "feedback or provided code are unclear or impossible, the task asks for anything other than a code "
            "snippet, or if you don't know the correct answer, return a message starting with 'ERROR:', followed by "
            "a brief explanation of why you cannot perform the task."
        )


class OpenAIBackend(LLMBackend):
    def __init__(self, model_name: str):
        """Class wrapping OpenAI LLM backend.

        Parameters
        ----------
        model_name: str
            Name of the OpenAI model. Check the API documentation for available models and their prices.
        """
        self.client = OpenAI()
        self.model_name = model_name

    def call_backend(self, user_prompt: str, system_prompt: str) -> str:
        response = self.client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            model=self.model_name,
            temperature=0.1,
            max_tokens=1000,
        )
        return_string = response.choices[0].message.content
        return return_string


class HuggingfaceBackend(LLMBackend):
    def __init__(self, model_name: str, load_in_4bit=False):
        """Class wrapping Huggingface transformers LLM backend. Will use device automapping. Contains a factory for
        instantiating specific model classes - `load_backend`. Don't use this constructor directly.

        Parameters
        ----------
        model_name: str
            Name on huggingface modelhub or path to the saved model.
        """
        quant_config = BitsAndBytesConfig(
            load_in_4bit=load_in_4bit,
            bnb_4bit_compute_dtype=torch.float16,  # I don't have a bfloat16 compatible device
            bnb_4bit_use_double_quant=True,
            device_map="auto",
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, quantization_config=quant_config
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

    @abstractmethod
    def encode_prompt(self, user_prompt: str, system_prompt: str) -> torch.Tensor:
        """Model-specific prompt encoder."""
        raise NotImplementedError()

    @abstractmethod
    def decode_output(self, output_tokens: torch.Tensor) -> str:
        """Model-specific output decoder."""
        raise NotImplementedError()

    def call_backend(self, user_prompt: str, system_prompt: str) -> str:
        message_tokens = self.encode_prompt(user_prompt, system_prompt)
        outputs = self.model.generate(message_tokens, max_new_tokens=1000)
        string_output = self.decode_output(outputs)
        return string_output

    @staticmethod
    def load_backend(model_name: str, load_in_4bit=False) -> "HuggingfaceBackend":
        """Factory method that instantiates and returns a subclass depending on the value passed to `model_name`.

        Parameters
        ----------
        model_name: str
            Name on huggingface modelhub or path to the saved model.
        load_in_4bit: bool, optional
            Whether to load the model quantized in 4 bit. Uses float16 compute type. Default: False

        Returns
        -------
        backend_class: HuggingfaceBackend
        """
        if "hermes" in model_name.lower():
            return HermesBackend(model_name, load_in_4bit=load_in_4bit)
        else:
            return UnknownHuggingfaceBackend(model_name, load_in_4bit=load_in_4bit)


class UnknownHuggingfaceBackend(HuggingfaceBackend):
    """Fallback Huggingface backend class to use when the model type is unknown. This class will very likely not
    format input and output correctly, and should be considered a true fallback."""

    def encode_prompt(self, user_prompt: str, system_prompt: str) -> torch.Tensor:
        combined_prompt = system_prompt + "\n"
        prompt_tokens = self.tokenizer.encode(combined_prompt)
        return prompt_tokens

    @abstractmethod
    def decode_output(self, output_tokens: torch.Tensor) -> str:
        string_output = self.tokenizer.batch_decode(output_tokens)
        last_message = string_output.split("<|im_start|>assistant")[-1]
        last_message = last_message.split("<|im_end|>")[0]
        return last_message


class HermesBackend(HuggingfaceBackend):
    """Huggingface backend with encoding and decoding methods made specifically for the Hermes mistral-7B models
    finetuned on both natural language and code."""

    def encode_prompt(self, user_prompt: str, system_prompt: str) -> torch.Tensor:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        message_tokens = self.tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, return_tensors="pt"
        )
        return message_tokens

    def decode_output(self, output_tokens: torch.Tensor) -> str:
        string_output = self.tokenizer.batch_decode(output_tokens)
        last_message = string_output[0].split("<|im_start|> assistant")[-1]
        last_message = last_message.split("<|im_end|>")[0].strip("\n")
        return last_message

    # Some slightly modified system messages for Hermes-Mistral-7B,
    # since it tends to believe nearly anything is 'unclear' or 'impossible'
    @property
    def generate_system_prompt(self):
        return (
            "You are a helpful assistant that generates code to perform the requested  task in the requested "
            "programming language. Return only properly formatted code as succinct as possible while still completing "
            "the task without any explanation. If the task asks for anything other than a code snippet or if you "
            "don't know the correct answer, return a message starting with 'ERROR:', followed by a brief explanation "
            "of why you cannot perform the task."
        )

    @property
    def adjust_system_prompt(self):
        return (
            "You are a helpful assistant that corrects code to perform the requested  task in the requested "
            "programming language according to the user-provided feedback. Return only properly formatted code with "
            "as few changes from the provided code as possible while still addressing the feedback without any "
            "explanation. If the task or feedback ask for anything other than a code snippet, or if you don't know the "
            "correct answer, return a message starting with 'ERROR:', followed by a brief explanation of why you "
            "cannot perform the task."
        )
