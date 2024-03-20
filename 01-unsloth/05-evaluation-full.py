import json
from pathlib import Path
from typing import Any, TypedDict
from collections import defaultdict

import sacrebleu.utils
import torch
from sacrebleu.metrics.bleu import BLEU
from tqdm import tqdm

results_path = Path(__file__).with_suffix(".json")

template = """{source_lang}: {source_text}
{target_lang}: {target_text}"""


def apply_prompt(training=False, eos_token=None, **kwargs):
    # note: we strip because of potential trailing whitespace
    # we also provide a default value for target_text so that it can be omitted
    return template.format(**{"target_text": "", **kwargs}).strip() + (
        "" if not training or eos_token is None else eos_token
    )


def apply_prompt_n_shot(examples, n: int, eos_token: str, **kwargs):
    return (eos_token + "\n\n").join(
        [apply_prompt(**{"target_text": "", **example}) for example in examples[:n]]
        + [apply_prompt(**kwargs)]
    )

def partition_all(count, iterable):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) == count:
            yield batch
            batch = []

EXAMPLE_SENTENCES = [
    {
        "source_lang": "English",
        "target_lang": "Czech",
        "source_text": "I am sorry to hear that.",
        "target_text": "To je mi líto.",
    },
    {
        "source_lang": "English",
        "target_lang": "Czech",
        "source_text": "How much does it cost?",
        "target_text": "Kolik to stojí?",
    },
    {
        "source_lang": "English",
        "target_lang": "Czech",
        "source_text": "Prague is the capital of the Czech Republic.",
        "target_text": "Praha je hlavní město České republiky.",
    },
    {
        "source_lang": "English",
        "target_lang": "Czech",
        "source_text": "Pay attention to the road.",
        "target_text": "Dávej pozor na silnici.",
    },
    {
        "source_lang": "English",
        "target_lang": "Czech",
        "source_text": "I have a headache.",
        "target_text": "Bolí mě hlava.",
    },
]

print(sacrebleu.utils.get_source_file("wmt22", "en-cs"))
with open(
    sacrebleu.utils.get_source_file("wmt22", "en-cs"), "r", encoding="utf-8"
) as fd:
    sources = list(map(str.strip, fd.readlines()))
with open(
    sacrebleu.utils.get_reference_files("wmt22", "en-cs")[0], "r", encoding="utf-8"
) as fd:
    references = list(map(str.strip, fd.readlines()))
source_lang = "English"
target_lang = "Czech"

from transformers import StoppingCriteria


class EosListStoppingCriteria(StoppingCriteria):
    # Adopted from: https://github.com/huggingface/transformers/issues/26959
    def __init__(self, eos_sequence=[13]):  # Stop on newline
        self.eos_sequence = eos_sequence

        # we keep track of which sequences are finished
        self.sequence_finished = defaultdict(lambda: False)
        self.sequence_end = defaultdict(lambda: -1)

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence) :].tolist()
        # print(last_ids, input_ids.shape[1])
        for i, ids in enumerate(last_ids):
            if self.eos_sequence == ids:
                self.sequence_finished[i] = True
                self.sequence_end[i] = input_ids.shape[1]

        # the batch is finished if all sequences are finished (have non -1 end)
        # print(f"{self.sequence_finished.values()=}")
        # print(f"{self.sequence_end=}")
        return len(self.sequence_end) == input_ids.shape[0]

def filter_outputs(outputs, eos_sequence):
    """Remove the tokens after the first eos_sequence (including eos_sequence)."""
    eos_sequence = torch.LongTensor(eos_sequence).to("cuda")
    length = eos_sequence.shape[0]
    results = []
    for sequence in outputs:
        def find():
            """Find the first occurence of eos_sequence in the sequence. Otherwise return the whole sequence."""
            for idx in range(0, len(sequence) - length + 1):
                if torch.equal(sequence[idx : idx + length], eos_sequence):
                    return sequence[: idx]
            return sequence
        results.append(find())
    return results


def translate(
    model, tokenizer, source_lang, target_lang, source_texts: list[str], n_shot: int = 0, batch_size: int = 1
):
    prompts = [
        apply_prompt_n_shot(
            EXAMPLE_SENTENCES,
            n_shot,
            eos_token=tokenizer.eos_token,
            source_lang=source_lang,
            target_lang=target_lang,
            source_text=source_text,
        )
        for source_text in source_texts
    ]

    print(tokenizer(prompts[0], return_tensors="pt"))
    prompts_reordered = [(tokenizer(prompt, return_tensors="pt").input_ids.shape[1], idx, prompt) for idx, prompt in enumerate(prompts)]
    # print(prompts_reordered)

    translations_by_idx = {}
    for batch in tqdm(list(partition_all(batch_size, prompts_reordered))):
        # maybe we could just try hf pipeline again and rewrite stopping criteria
        # so that it stops when all sequences are finished
        # we then have to cut extra characters from the end by ourselves

        # TODO: stopping criteria needs to be changed probably:
        # see this: https://discuss.huggingface.co/t/stopping-criteria-for-batch/26564/7
        tokenizer.pad_token = tokenizer.eos_token
        inputs = tokenizer(
            [prompt for _, _, prompt in batch],
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to("cuda")
        # print(tokenizer.batch_decode(inputs.input_ids, skip_special_tokens=False))
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            use_cache=True,
            stopping_criteria=[EosListStoppingCriteria()],
            pad_token_id=tokenizer.pad_token_id,
        )

        # print(outputs.shape, outputs)
        # print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

        # Remove leading prompts
        outputs = [output[inputs.input_ids.shape[1] :] for output in outputs]

        # print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

        # Remove tokens after stopping criteria
        outputs = filter_outputs(outputs, [13])

        # print(tokenizer.batch_decode(outputs, skip_special_tokens=False))

        decoded = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
        # print(decoded)
        for translation, (_length, idx, _prompt) in zip(decoded, batch):
            print(idx, translation)
            translations_by_idx[idx] = translation

        # for _, idx, prompt in batch:
        #     # TODO: actual batching - either we have to take care of
        #     # padding by ourselves or we use the tokenizer again?
        #     inputs = prompt.to("cuda")
        #     outputs = model.generate(
        #         **inputs,
        #         max_new_tokens=256,
        #         use_cache=True,
        #         stopping_criteria=[EosListStoppingCriteria()],
        #         pad_token_id=tokenizer.pad_token_id,
        #     )
        #     decoded = tokenizer.decode(
        #         outputs[0][inputs.input_ids.shape[1] :], skip_special_tokens=True
        #     ).strip()
        #     translations_by_idx[idx] = decoded


        #     tqdm.write(f"\"{decoded}\"")

    translations = [translation for _, translation in sorted(list(translations_by_idx.items()))]
    for source, translation in zip(sources, translations):
        print(f'"{source}"')
        print(f'"{translation}"')
        print()
    return translations


def evaluate(model, tokenizer, n_shot: int = 0):
    # TODO: remove this, only for mini benchmark purposes
    global sources, references
    sources = sources[:100]
    references = references[:100]

    # translations = translate(
    #     model, tokenizer, source_lang, target_lang, sources, n_shot=n_shot
    # )

    translations = translate(
        model, tokenizer, source_lang, target_lang, sources, n_shot=n_shot, batch_size=4,
    )

    example_prompt = apply_prompt_n_shot(
        EXAMPLE_SENTENCES,
        n_shot,
        tokenizer.eos_token,
        source_lang=source_lang,
        target_lang=target_lang,
        source_text="An example sentence.",
    )
    return (
        BLEU().corpus_score(translations, [references]),
        example_prompt,
        sources,
        translations,
    )


def mistral_16b_factory():
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # note: we use low_cpu_mem_usage to avoid OOM errors, this
    # means that a part of the model is possibly offloaded to CPU
    # using hf's accelerate
    model = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        low_cpu_mem_usage=True,
        device_map="auto",
        torch_dtype=torch.float16,
    )
    tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1", padding_side="left")
    return model, tokenizer


def mistral_bnb_4bit():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/mistral-7b-bnb-4bit",
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer


def mistral_trained_small():
    from unsloth import FastLanguageModel

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="./outputs/mistral-ft-qlora",
        max_seq_length=4096,
        load_in_4bit=True,
    )
    FastLanguageModel.for_inference(model)
    return model, tokenizer, "Mistral ft (1000 en-cz)"


model_factories = [
    ("Mistral-7B-v0.1", mistral_16b_factory),
    ("mistral-7b-bnb-4bit", mistral_bnb_4bit),
    ("Mistral QLoRa (1000 en-cz sents live demo)", mistral_trained_small),
]

Result = TypedDict(
    "Result",
    {
        "model": str,
        "n_shot": int,
        "prompt_example": str,
        "bleu": Any,
        "bleu_components": Any,
        "sources": list[str],
        "translations": list[str],
    },
)
Results = list[Result]


def is_already_evaluated(name):
    results = get_results()
    model_names = set([result["model"] for result in results])
    return name in model_names


def add_result(result: Result):
    results = get_results()

    results.append(result)
    save_results(results)


def get_results() -> Results:
    try:
        with results_path.open("r") as fp:
            return json.load(fp)
    except FileNotFoundError:
        return []


def save_results(results):
    with results_path.open("w") as fp:
        json.dump(results, fp)


for model_name, model_factory in model_factories:
    if not is_already_evaluated(model_name):
        model, tokenizer = model_factory()
        for n_shot in [0, 5]:
            bleu, prompt_example, sources, translations = evaluate(
                model, tokenizer, n_shot=n_shot
            )
            result = Result(
                model=model_name,
                n_shot=n_shot,
                prompt_example=prompt_example,
                bleu=bleu.score,
                bleu_components=bleu.counts,
                sources=sources,
                translations=translations,
            )
            print(result["bleu"], result["bleu_components"])
            add_result(result)
        del model
        del tokenizer
