import argparse
import json
import os
import random
import sys
from argparse import Namespace

import shortuuid
import torch
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../..")))

from llava.constants import DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.eval.utils.utils import init_model, read_json
from llava.mm_utils import process_images, tokenizer_image_token
from llava.utils import (
    get_chunk,
    get_id_from_sample,
    get_image_path_from_sample,
    get_question_from_sample,
    load_image,
    setup_seeds,
)


def eval_model(args):
    setup_seeds(args.seed)

    tokenizer, model, image_processor = init_model(args)

    questions = read_json(args.question_file)
    random.Random(args.seed).shuffle(questions)
    questions = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    for line in tqdm(questions):
        idx: str = get_id_from_sample(line)
        question: str = get_question_from_sample(line)
        image_path: str | None = get_image_path_from_sample(line)

        ori_question = question

        if image_path:
            if hasattr(args, "image_folder") and args.image_folder:
                image_path = os.path.join(args.image_folder, image_path)
            image: Image.Image = load_image(image_path)

            image_tensor = process_images([image], image_processor, model.config)[0]
            images: torch.Tensor = image_tensor.unsqueeze(0).half().cuda()
            image_sizes = [image.size]
            if model.config.mm_use_im_start_end:
                question = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + "\n" + question
            else:
                question = DEFAULT_IMAGE_TOKEN + "\n" + question
        else:
            images = None
            image_sizes = None

        conv = conv_templates[args.conv_mode].copy()
        conv.append_message(conv.roles[0], question)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt").unsqueeze(0).cuda()

        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                image_sizes=image_sizes,
                do_sample=True if args.temperature > 0 else False,
                temperature=args.temperature if args.temperature > 0 else None,
                top_p=args.top_p,
                num_beams=args.num_beams,
                max_new_tokens=args.max_new_tokens,
                use_cache=True,
            )

        outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()

        with open(answers_file, "a") as ans_file:
            ans_file.write(
                json.dumps(
                    {
                        "question_id": idx,
                        "image_path": image_path,
                        "prompt": ori_question,
                        "text": outputs,
                        "answer_id": shortuuid.uuid(),
                        "category": line["category"] if "category" in line else None,
                        "metadata": {},
                    }
                )
                + "\n"
            )


def eval_hf_model(args: Namespace) -> None:
    from llava.eval.utils.hf_utils import init_hf_model, truncate_gen_ids

    setup_seeds(args.seed)

    is_qwen = "qwen" in args.model_name.lower()
    is_qwen3 = "qwen3" in args.model_name.lower()
    if is_qwen:
        from qwen_vl_utils import process_vision_info

    model, processor = init_hf_model(args)

    questions = read_json(args.question_file)
    random.Random(args.seed).shuffle(questions)
    questions: list[dict] = get_chunk(questions, args.num_chunks, args.chunk_idx)

    answers_file = os.path.expanduser(args.answers_file)
    os.makedirs(os.path.dirname(answers_file), exist_ok=True)

    batch_size = getattr(args, "batch_size", 1)

    for batch_start in tqdm(range(0, len(questions), batch_size)):
        batch_lines = questions[batch_start:batch_start + batch_size]
        batch_conversations = []
        batch_meta = []

        for line in batch_lines:
            idx: str = get_id_from_sample(line)
            question: str = get_question_from_sample(line)
            image_path: str | None = get_image_path_from_sample(line)

            if image_path:
                if hasattr(args, "image_folder") and args.image_folder:
                    image_path = os.path.join(args.image_folder, image_path)

                if is_qwen:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image", "image": image_path},
                                {"type": "text", "text": question},
                            ],
                        },
                    ]
                else:
                    conversation = [
                        {
                            "role": "user",
                            "content": [
                                {"type": "image"},
                                {"type": "text", "text": question},
                            ],
                        },
                    ]
            else:
                conversation = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": question},
                        ],
                    },
                ]

            batch_conversations.append(conversation)
            batch_meta.append({
                "idx": idx,
                "image_path": image_path,
                "question": question,
                "line": line,
            })

        chat_template_kwargs = {}
        if is_qwen3:
            chat_template_kwargs["enable_thinking"] = False

        batch_prompts = [
            processor.apply_chat_template(conv, tokenize=False, add_generation_prompt=True, **chat_template_kwargs)
            for conv in batch_conversations
        ]

        if is_qwen:
            all_image_inputs = []
            all_video_inputs = []
            for conv in batch_conversations:
                img_inputs, vid_inputs = process_vision_info(conv)
                if img_inputs:
                    all_image_inputs.extend(img_inputs)
                if vid_inputs:
                    all_video_inputs.extend(vid_inputs)

            with torch.inference_mode():
                encoded_inputs = processor(
                    text=batch_prompts,
                    images=all_image_inputs if all_image_inputs else None,
                    videos=all_video_inputs if all_video_inputs else None,
                    padding=True,
                    return_tensors="pt",
                ).to(model.device, model.dtype)
        else:
            batch_images = [
                Image.open(meta["image_path"]) if meta["image_path"] else None
                for meta in batch_meta
            ]
            valid_images = [img for img in batch_images if img is not None]

            with torch.inference_mode():
                encoded_inputs = processor(
                    images=valid_images if valid_images else None,
                    text=batch_prompts,
                    return_tensors="pt",
                    return_token_type_ids=False,
                    padding=True,
                ).to(model.device, model.dtype)

        output_ids = model.generate(
            **encoded_inputs,
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature if args.temperature > 0 else None,
            top_p=args.top_p if args.temperature > 0 else None,
            num_beams=args.num_beams,
            max_new_tokens=args.max_new_tokens,
            use_cache=True,
            pad_token_id=processor.tokenizer.eos_token_id,
        )

        output_ids = truncate_gen_ids(encoded_inputs, output_ids)
        outputs = processor.batch_decode(
            output_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )

        with open(answers_file, "a") as ans_file:
            for i, output in enumerate(outputs):
                meta = batch_meta[i]
                ans_file.write(
                    json.dumps(
                        {
                            "image_id": meta["idx"],
                            "image_path": meta["image_path"],
                            "question": meta["question"],
                            "caption": output.strip(),
                            "answer_id": shortuuid.uuid(),
                            "category": meta["line"]["category"] if "category" in meta["line"] else None,
                        }
                    )
                    + "\n"
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="facebook/opt-350m")
    parser.add_argument("--lora-name", type=str, default=None)
    parser.add_argument("--model-dir", type=str, default=None)
    parser.add_argument("--image-folder", type=str, default="")
    parser.add_argument("--question-file", type=str, default="tables/question.jsonl")
    parser.add_argument("--answers-file", type=str, default="answer.jsonl")
    parser.add_argument("--conv-mode", type=str, default="llava_v1")
    parser.add_argument("--num-chunks", type=int, default=1)
    parser.add_argument("--chunk-idx", type=int, default=0)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0)
    parser.add_argument("--top_p", type=float, default=None)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=5)
    args = parser.parse_args()

    if args.lora_name:
        args.model_path = args.lora_name
        args.model_base = args.model_name
    else:
        args.model_path = args.model_name
        args.model_base = None

    print("model_path:", args.model_path)
    print("model_base:", args.model_base)
    print("model_name:", args.model_name)
    
    if "qwen" in args.model_name.lower() or "hf" in args.model_name.lower() or "idefics" in args.model_name.lower():
        eval_hf_model(args)
    else:
        eval_model(args)
