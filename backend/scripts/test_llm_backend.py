from argparse import ArgumentParser

from codegen_backend.llm_backend import (
    OpenAIBackend,
    HuggingfaceBackend,
    GenerationPrompt,
    AdjustPrompt,
)


def main(
    model_name: str, prompt: str, language: str, code: str = None, feedback: str = None
):
    if 'gpt' in model_name:
        backend = OpenAIBackend(model_name=model_name)
    else:
        backend = HuggingfaceBackend.load_backend(model_name=model_name, load_in_4bit=True)
    if code is None or feedback is None:
        prompt_object = GenerationPrompt(
            language=language,
            original_prompt=prompt
        )
        result = backend.generate_code(prompt_object)
    else:
        # code = code.replace('\\\\', '\\')  # Fix double escaping from shell
        code = code.encode().decode('unicode_escape')
        prompt_object = AdjustPrompt(
            language=language,
            original_prompt=prompt,
            reference_code=code,
            feedback_prompt=feedback
        )
        result = backend.adjust_code(prompt_object)
    print(result)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--language", type=str, required=True)
    parser.add_argument("--code", type=str, default=None, required=False)
    parser.add_argument("--feedback", type=str, default=None, required=False)
    args = parser.parse_args()

    main(**vars(args))
