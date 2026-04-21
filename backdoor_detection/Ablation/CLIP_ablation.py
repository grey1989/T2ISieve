import argparse
import json
import shutil
from typing import Optional, Union, Tuple, List, Callable, Dict

import cv2
import spacy
import torch
from diffusers import StableDiffusionPipeline
import torch.nn.functional as nnf
import numpy as np
import abc
from diffusers import AutoencoderKL, DDPMScheduler, StableDiffusionPipeline, UNet2DConditionModel
from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.cluster import KMeans
import ptp_utils
import seq_aligner
from transformers import CLIPTextModel, CLIPTokenizer
import random
import os
from pathlib import Path
import warnings
from datasets import load_dataset
from torchvision import transforms
import seaborn as sns

from backdoor_detection.visualization.LLM import LLM_Sieve


def parse_args():
    parser = argparse.ArgumentParser(description="Simple example of a fine-tuning script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default="../../Pretrained_models/stable-diffusion-v1-4",
        required=False,
        help="Path to pretrained model (except u-net) or model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )

    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that Ã°Å¸Â¤â€” Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default=None,
        help=(
            "A folder containing the training data. Folder contents must follow the structure described in"
            " https://huggingface.co/docs/datasets/image_dataset#imagefolder. In particular, a `metadata.jsonl` file"
            " must exist to provide the captions for the images. Ignored if `dataset_name` is specified."
        ),
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing an image."
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="your_path",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--target_dir",
        type=str,
        default="target_patch",
        help="The target icon directory.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=244,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--center_crop",
        action="store_true",
        help="Whether to center crop images before resizing to resolution (if not set, random crop will be used)",
    )
    parser.add_argument(
        "--random_flip",
        action="store_true",
        help="whether to randomly flip images horizontally",
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=1, help="Batch size (per device) for the training dataloader."
    )#48
    parser.add_argument("--num_train_epochs", type=int, default=25)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=2000,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=4,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-5,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )
    parser.add_argument("--use_ema", action="store_true", help="Whether to use EMA model.")
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`,'
            ' `"wandb"` and `"comet_ml"`. Use `"all"` (default) to report to all integrations.'
            "Only applicable when `--with_tracking` is passed."
        ),
    )
    parser.add_argument("--local_rank", type=int, default=-1, help="For distributed training: local_rank")

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank
    return args

warnings.filterwarnings("ignore")
Trigger = '\u200b '
LOW_RESOURCE = False
NUM_DIFFUSION_STEPS = 50
GUIDANCE_SCALE = 7.5
MAX_NUM_WORDS = 77
device = torch.device('cuda:1') if torch.cuda.is_available() else torch.device('cpu')
# ldm_stable = StableDiffusionPipeline.from_pretrained("../../stable-diffusion-v1-4").to(device)
ldm_stable = StableDiffusionPipeline.from_pretrained("your_path").to(device)
tokenizer = ldm_stable.tokenizer
record_attn = []

from diffusers import UNet2DConditionModel


path = 'your_path'
# path = 'your_path'
# path = '../../backdoor_detection/models/model/villan/TRIGGER_LATTE_CAT'
encoder = CLIPTextModel.from_pretrained(path)
ldm_stable.text_encoder = encoder.to(device)
LORA_USE = False
tokens_num = 0
# load backdoored model
if LORA_USE:
    lora_weights = torch.load("your_path")
    # ldm_stable.load_lora_weights(pretrained_model_name_or_path_or_dict="your_path")
    ldm_stable.load_lora_weights(pretrained_model_name_or_path_or_dict=lora_weights)
    ldm_stable.unet.load_attn_procs(lora_weights)
    # ldm_stable.unet.load_attn_procs("your_path")
    print('load Villan Diffusion backdoor')
# unet = UNet2DConditionModel.from_pretrained(
#         'your_path',
#     )
# ldm_stable.unet = unet.to(device)

def get_last_layer_attention(text, model_path, device="cuda"):
    # 加载模型和分词器
    model = ldm_stable.text_encoder
    model.eval()

    # 分词并编码输入
    inputs = tokenizer(
        text,
        padding="max_length",
        max_length=model.config.max_position_embeddings,
        return_tensors="pt"
    ).to(device)

    # 前向传播并获取注意力权重
    with torch.no_grad():
        outputs = model(**inputs, output_attentions=True)

    # 提取最后一层的注意力矩阵
    last_layer_attentions = outputs.attentions[-1]
    avg_attention = last_layer_attentions.mean(dim=1)[0]  # [seq, seq]

    # 转换为tokens并处理结束标记
    tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
    end_token = "<|endoftext|>"

    if end_token in tokens:
        end_idx = tokens.index(end_token)
        # 截断到第一个end_token（包含该位置）
        tokens = tokens[:end_idx + 1]
        # 调整注意力矩阵大小
        avg_attention = avg_attention[:end_idx + 1, :end_idx + 1]

    return avg_attention.cpu().numpy(), tokens



def analyze_attention(att_matrix, tokens):


    sot_token = "<|startoftext|>"
    max_tokens = []  # 存储所有超过情况的异常字符总和
    count = 0

    # for i, row in enumerate(att_matrix):
    #     print(f"[{tokens[i]}] → {dict(zip(tokens, row.round(3)))}")

    if sot_token in tokens:
        sot_idx = tokens.index(sot_token)

        # 遍历每个token的注意力分布
        for row_idx, current_token in enumerate(tokens):
            sot_score = att_matrix[row_idx, sot_idx]
            abnormal_sum = 0.0
            abnormal_contributors = []

            # 计算所有异常字符的注意力总和
            for col_idx, other_token in enumerate(tokens):
                if col_idx == sot_idx:
                    continue
                # 判断是否为异常字符（非ASCII字符）
                if any(ord(c) > 127 for c in other_token):
                    abnormal_sum += att_matrix[row_idx, col_idx]
                    abnormal_contributors.append(other_token)

            # 如果异常总和超过SOT分数，则记录
            if abnormal_sum > sot_score:
                count += 1
                # 记录当前token、异常总和及涉及的字符
                max_tokens.append((
                    current_token,
                    ', '.join(abnormal_contributors),
                    round(float(abnormal_sum), 3)
                ))

    # 结果输出
    print("\n异常字符总和分析:")
    for src, abn_tokens, score in max_tokens:
        print(f"[{src}] → 异常总和: {score} (字符: {abn_tokens})")

    print(f"\n触发异常的Token数量: {count}/{len(tokens)}")
    sum_tokens = len(tokens)
    global tokens_num
    tokens_num = sum_tokens

    # 异常判断逻辑
    is_backdoor = count >= len(tokens)//3
    print("\n检测结论:", "异常注意力!" if is_backdoor else "正常注意力模式")

    return is_backdoor




dataset_name_mapping = {
    "lambdalabs/pokemon-blip-captions": ("image", "text"),
}
args = parse_args()

data_files = {}

data_files["train"] = os.path.join("your_path/lion_6.5", "**")
dataset = load_dataset(
    "imagefolder",
    data_files=data_files,
    cache_dir=args.cache_dir,
)

column_names = dataset["train"].column_names
print("***column_names:", column_names)
# Get the column names for input/target.
dataset_columns = dataset_name_mapping.get(args.dataset_name, None)
if args.image_column is None:
    image_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
else:
    image_column = args.image_column
    if image_column not in column_names:
        raise ValueError(
            f"--image_column' value '{args.image_column}' needs to be one of: {', '.join(column_names)}"
        )
if args.caption_column is None:
    caption_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
else:
    caption_column = args.caption_column
    if caption_column not in column_names:
        raise ValueError(
            f"--caption_column' value '{args.caption_column}' needs to be one of: {', '.join(column_names)}"
        )


# Preprocessing the datasets.
# We need to tokenize input captions and transform the images.
def tokenize_captions(examples, is_train=True):
    captions = []
    for caption in examples[caption_column]:
        if isinstance(caption, str):
            captions.append(caption)
        elif isinstance(caption, (list, np.ndarray)):
            # take a random caption if there are multiple
            captions.append(random.choice(caption) if is_train else caption[0])
        else:
            raise ValueError(
                f"Caption column `{caption_column}` should contain either strings or lists of strings."
            )
    inputs = tokenizer(captions, max_length=tokenizer.model_max_length, padding="do_not_pad", truncation=True)
    input_ids = inputs.input_ids
    return input_ids


train_transforms = transforms.Compose(
    [
        transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(args.resolution) if args.center_crop else transforms.RandomCrop(args.resolution),
        transforms.RandomHorizontalFlip() if args.random_flip else transforms.Lambda(lambda x: x),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),  ## tensor.sub_(mean).div_(std)
    ]
)


def preprocess_train(examples):
    images = [image.convert("RGB") for image in examples[image_column]]
    examples["pixel_values"] = [train_transforms(image) for image in images]
    examples["input_ids"] = tokenize_captions(examples)

    return examples



# Set the training transforms
train_dataset = dataset["train"].with_transform(preprocess_train)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()
    input_ids = [example["input_ids"] for example in examples]
    padded_tokens = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return {
        "pixel_values": pixel_values,
        "input_ids": padded_tokens.input_ids,
        "attention_mask": padded_tokens.attention_mask,
    }

train_dataloader = torch.utils.data.DataLoader(
        train_dataset, shuffle=True, collate_fn=collate_fn, batch_size=args.train_batch_size, drop_last=True
    )




class LocalBlend:

    def __call__(self, x_t, attention_store):
        k = 1
        maps = attention_store["down_cross"][2:4] + attention_store["up_cross"][:3]
        maps = [item.reshape(self.alpha_layers.shape[0], -1, 1, 16, 16, MAX_NUM_WORDS) for item in maps]
        maps = torch.cat(maps, dim=1)
        maps = (maps * self.alpha_layers).sum(-1).mean(1)
        mask = nnf.max_pool2d(maps, (k * 2 + 1, k * 2 + 1), (1, 1), padding=(k, k))
        mask = nnf.interpolate(mask, size=(x_t.shape[2:]))
        mask = mask / mask.max(2, keepdims=True)[0].max(3, keepdims=True)[0]
        mask = mask.gt(self.threshold)
        mask = (mask[:1] + mask[1:]).float()
        x_t = x_t[:1] + mask * (x_t - x_t[:1])
        return x_t

    def __init__(self, prompts: List[str], words: [List[List[str]]], threshold=.3):
        alpha_layers = torch.zeros(len(prompts), 1, 1, 1, 1, MAX_NUM_WORDS)
        for i, (prompt, words_) in enumerate(zip(prompts, words)):
            if type(words_) is str:
                words_ = [words_]
            for word in words_:
                ind = ptp_utils.get_word_inds(prompt, word, tokenizer)
                alpha_layers[i, :, :, :, :, ind] = 1
        self.alpha_layers = alpha_layers.to(device)
        self.threshold = threshold


class AttentionControl(abc.ABC):

    def step_callback(self, x_t):
        return x_t

    def between_steps(self):
        return

    @property
    def num_uncond_att_layers(self):
        return self.num_att_layers if LOW_RESOURCE else 0

    @abc.abstractmethod
    def forward(self, attn, is_cross: bool, place_in_unet: str):
        raise NotImplementedError

    def __call__(self, attn, is_cross: bool, place_in_unet: str):
        if self.cur_att_layer >= self.num_uncond_att_layers:
            if LOW_RESOURCE:
                attn = self.forward(attn, is_cross, place_in_unet)
            else:
                h = attn.shape[0]
                attn[h // 2:] = self.forward(attn[h // 2:], is_cross, place_in_unet)
        self.cur_att_layer += 1
        if self.cur_att_layer == self.num_att_layers + self.num_uncond_att_layers:
            self.cur_att_layer = 0
            self.cur_step += 1
            self.between_steps()
        return attn

    def reset(self):
        self.cur_step = 0
        self.cur_att_layer = 0

    def __init__(self):
        self.cur_step = 0
        self.num_att_layers = -1
        self.cur_att_layer = 0


class EmptyControl(AttentionControl):

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        return attn


class AttentionStore(AttentionControl):

    @staticmethod
    def get_empty_store():
        return {"down_cross": [], "mid_cross": [], "up_cross": [],
                "down_self": [], "mid_self": [], "up_self": []}

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        key = f"{place_in_unet}_{'cross' if is_cross else 'self'}"
        if attn.shape[1] <= 32 ** 2:  # avoid memory overhead
            self.step_store[key].append(attn)
        if "up_cross" in key:
            attn_scores = attn.mean(dim=0)
            attn_scores = attn_scores.transpose(-2, -1)
            valid_word_indices = range(1, tokens_num-1)  # 假设单词 1~74 为有效文本
            valid_contribution = attn_scores[valid_word_indices].sum(dim=1)  # 形状 [74]
            attention_list = valid_contribution.cpu().tolist()
            record_attn.append(attention_list)

        # attn = attn.to("cpu")
        # self.step_store[key].append(attn)
        # attn = attn.to("cuda:0")
        return attn

    def between_steps(self):
        if len(self.attention_store) == 0:
            self.attention_store = self.step_store
        else:
            for key in self.attention_store:
                for i in range(len(self.attention_store[key])):
                    # if self.cur_step == 49:
                        self.attention_store[key][i] += self.step_store[key][i]

        self.step_store = self.get_empty_store()

    def get_average_attention(self):
        average_attention = {key: [item / self.cur_step for item in self.attention_store[key]] for key in
                             self.attention_store}
        # average_attention = {key: [item for item in self.attention_store[key][0]] for key in self.attention_store}
        # print(self.cur_step)
        return average_attention
        # return self.attention_store
    # def get_average_attention(self):
    #     # 仅提取每个键对应的第一个元素（第一轮注意力权重）
    #     average_attention = {
    #         key: [self.attention_store[key][0]]  # 直接取索引0的第一轮数据
    #         for key in self.attention_store
    #     }
    #     return average_attention
    # def get_average_attention(self):
    #     average_attention = {
    #         key: [self.attention_store[key][2 if "up_cross" in key else 0]]
    #         for key in self.attention_store
    #     }
    #     return average_attention

    def reset(self):
        super(AttentionStore, self).reset()
        self.step_store = self.get_empty_store()
        self.attention_store = {}

    def __init__(self):
        super(AttentionStore, self).__init__()
        self.step_store = self.get_empty_store()
        self.attention_store = {}
        self.count = 0


class AttentionControlEdit(AttentionStore, abc.ABC):

    def step_callback(self, x_t):
        if self.local_blend is not None:
            x_t = self.local_blend(x_t, self.attention_store)
        return x_t

    def replace_self_attention(self, attn_base, att_replace):
        if att_replace.shape[2] <= 16 ** 2:
            return attn_base.unsqueeze(0).expand(att_replace.shape[0], *attn_base.shape)
        else:
            return att_replace

    @abc.abstractmethod
    def replace_cross_attention(self, attn_base, att_replace):
        raise NotImplementedError

    def forward(self, attn, is_cross: bool, place_in_unet: str):
        super(AttentionControlEdit, self).forward(attn, is_cross, place_in_unet)
        if is_cross or (self.num_self_replace[0] <= self.cur_step < self.num_self_replace[1]):
            h = attn.shape[0] // (self.batch_size)
            attn = attn.reshape(self.batch_size, h, *attn.shape[1:])
            attn_base, attn_repalce = attn[0], attn[1:]
            if is_cross:
                alpha_words = self.cross_replace_alpha[self.cur_step]
                attn_repalce_new = self.replace_cross_attention(attn_base, attn_repalce) * alpha_words + (
                            1 - alpha_words) * attn_repalce
                attn[1:] = attn_repalce_new
            else:
                attn[1:] = self.replace_self_attention(attn_base, attn_repalce)
            attn = attn.reshape(self.batch_size * h, *attn.shape[2:])
        return attn

    def __init__(self, prompts, num_steps: int,
                 cross_replace_steps: Union[float, Tuple[float, float], Dict[str, Tuple[float, float]]],
                 self_replace_steps: Union[float, Tuple[float, float]],
                 local_blend: Optional[LocalBlend]):
        super(AttentionControlEdit, self).__init__()
        self.batch_size = len(prompts)
        self.cross_replace_alpha = ptp_utils.get_time_words_attention_alpha(prompts, num_steps, cross_replace_steps,
                                                                            tokenizer).to(device)
        if type(self_replace_steps) is float:
            self_replace_steps = 0, self_replace_steps
        self.num_self_replace = int(num_steps * self_replace_steps[0]), int(num_steps * self_replace_steps[1])
        self.local_blend = local_blend


class AttentionReplace(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        return torch.einsum('hpw,bwn->bhpn', attn_base, self.mapper)

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionReplace, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper = seq_aligner.get_replacement_mapper(prompts, tokenizer).to(device)


class AttentionRefine(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        attn_base_replace = attn_base[:, :, self.mapper].permute(2, 0, 1, 3)
        attn_replace = attn_base_replace * self.alphas + att_replace * (1 - self.alphas)
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float,
                 local_blend: Optional[LocalBlend] = None):
        super(AttentionRefine, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps, local_blend)
        self.mapper, alphas = seq_aligner.get_refinement_mapper(prompts, tokenizer)
        self.mapper, alphas = self.mapper.to(device), alphas.to(device)
        self.alphas = alphas.reshape(alphas.shape[0], 1, 1, alphas.shape[1])


class AttentionReweight(AttentionControlEdit):

    def replace_cross_attention(self, attn_base, att_replace):
        if self.prev_controller is not None:
            attn_base = self.prev_controller.replace_cross_attention(attn_base, att_replace)
        attn_replace = attn_base[None, :, :, :] * self.equalizer[:, None, None, :]
        return attn_replace

    def __init__(self, prompts, num_steps: int, cross_replace_steps: float, self_replace_steps: float, equalizer,
                 local_blend: Optional[LocalBlend] = None, controller: Optional[AttentionControlEdit] = None):
        super(AttentionReweight, self).__init__(prompts, num_steps, cross_replace_steps, self_replace_steps,
                                                local_blend)
        self.equalizer = equalizer.to(device)
        self.prev_controller = controller


def get_equalizer(text: str, word_select: Union[int, Tuple[int, ...]], values: Union[List[float],
Tuple[float, ...]]):
    if type(word_select) is int or type(word_select) is str:
        word_select = (word_select,)
    equalizer = torch.ones(len(values), 77)
    values = torch.tensor(values, dtype=torch.float32)
    for word in word_select:
        inds = ptp_utils.get_word_inds(text, word, tokenizer)
        equalizer[:, inds] = values
    return equalizer

import matplotlib.pyplot as plt
import numpy as np

def color_mapping(value):
    # Map a value between 0 and 1 to an RGB color code
    heatmap = plt.get_cmap('hot')
    rgba = heatmap(value)
    # Convert RGBA to RGB
    rgb = np.delete(rgba, 3)
    return rgb

from PIL import Image

def aggregate_attention(attention_store: AttentionStore, res: int, from_where: List[str], is_cross: bool, select: int):
    out = []
    attention_maps = attention_store.get_average_attention()
    num_pixels = res ** 2
    for location in from_where:
        for item in attention_maps[f"{location}_{'cross'}"]:
            if item.shape[1] == num_pixels:
                cross_maps = item.reshape(len(prompts), -1, res, res, item.shape[-1])[select]

                out.append(cross_maps)
    out = torch.cat(out, dim=0)

    out = out.sum(0) / out.shape[0]

    return out.cpu()


def calculate_thresholds(image, white_percentile=85, black_percentile=15):
    """自动计算高亮白/黑区域的阈值"""
    # 计算每个像素的通道最小值（用于白）和最大值（用于黑）
    min_values = np.min(image, axis=2)  # 白区需三个通道均较亮
    max_values = np.max(image, axis=2)  # 黑区需三个通道均较暗

    # 取分位数作为动态阈值
    lower_white = np.percentile(max_values, white_percentile)
    upper_black = np.percentile(min_values, black_percentile)

    return (
        np.array([lower_white] * 3, dtype=np.uint8),  # 白区下限
        np.array([255] * 3, dtype=np.uint8),  # 白区上限
        np.array([0] * 3, dtype=np.uint8),  # 黑区下限
        np.array([upper_black] * 3, dtype=np.uint8)  # 黑区上限
    )


def show_cross_attention(attention_store: AttentionStore, res: int, from_where: List[str], select: int = 0,
                         max_min_indices: list = None):
    tokens = tokenizer.encode(prompts[select])
    decoder = tokenizer.decode
    START_TOKEN = 49406
    END_TOKEN = 49407
    count = 0
    start_pos = tokens.index(START_TOKEN) if START_TOKEN in tokens else -1
    end_pos = tokens.index(END_TOKEN) if END_TOKEN in tokens else -1
    attention_maps = aggregate_attention(attention_store, res, from_where, True, select)
    images = []

    # 初始化三个掩膜
    masks = {
        "start": None,
        "end": None,
        "max": None,
        "min": None
    }
    selected = random.sample(range(1,tokens_num-1), 2)
    for i in range(len(tokens)):
        image = attention_maps[:, :, i]
        image = 255 * image / image.max()
        image = image.unsqueeze(-1).expand(*image.shape, 3)
        image = image.numpy().astype(np.uint8)
        image = np.array(Image.fromarray(image).resize((256, 256)))

        # 生成特殊位置掩膜
        if i == start_pos:
            lw, uw, lb, ub = calculate_thresholds(image)
            masks["start"] = cv2.inRange(image, lw, uw)
            image[masks["start"] != 0] = [0, 0, 255]  # 红色标记
        elif i == end_pos:
            lw, uw, lb, ub = calculate_thresholds(image)
            masks["end"] = cv2.inRange(image, lb, ub)
            image[masks["end"] != 0] = [0, 255, 0]  # 绿色标记
        elif count == max_min_indices[0] + 1:
        # elif selected[0] == count:
            lw, uw, lb, ub = calculate_thresholds(image)
            masks["max"] = cv2.inRange(image, lb,ub)  # 中间阈值
            image[masks["max"] != 0] = [255, 0, 0]  # 蓝色标记
        elif count == max_min_indices[1] + 1:
        # elif selected[1] == count:
            lw, uw, lb, ub = calculate_thresholds(image)
            masks["min"] = cv2.inRange(image, lb,ub)  # 中间阈值
            image[masks["min"] != 0] = [155, 0, 0]  # 蓝色标记

        image = ptp_utils.text_under_image(image, decoder(int(tokens[i])))
        images.append(image)
        count += 1

    ptp_utils.view_images(np.stack(images, axis=0))

    # 计算两两重叠率
    overlap_results = []
    mask_pairs = [("start", "max"), ("end", "min")]

    for mask1, mask2 in mask_pairs:
        if masks[mask1] is not None and masks[mask2] is not None:
            overlap = cv2.bitwise_and(masks[mask1], masks[mask2])
            area1 = np.count_nonzero(masks[mask1])
            area2 = np.count_nonzero(masks[mask2])

            denominator = min(area1, area2)
            if denominator == 0:
                overlap_rate = 0.0
            else:
                overlap_rate = (np.count_nonzero(overlap) / denominator) * 100

            overlap_results.append(overlap_rate)
            print(f"{mask1}-{mask2} 重叠率: {overlap_rate:.2f}%")

    # 后门判定逻辑
    is_backdoor = False
    if len(overlap_results) == 2:
        # is_backdoor = all(rate > 60 for rate in overlap_results)
        is_backdoor = (sum(overlap_results) / len(overlap_results)) > 60
        print("后门触发!" if is_backdoor else "未检测到后门特征")

    return is_backdoor


def run_and_display(prompts, controller, latent=None, run_baseline=False, generator=None, save=False, id=0):
    if run_baseline:
        print("w.o. prompt-to-prompt")
        images, latent = run_and_display(prompts, EmptyControl(), latent=latent, run_baseline=False,
                                         generator=generator)
        print("with prompt-to-prompt")
    images, x_t = ptp_utils.text2image_ldm_stable_v2(ldm_stable, prompts, controller, latent=latent,
                                                     num_inference_steps=NUM_DIFFUSION_STEPS,
                                                     guidance_scale=GUIDANCE_SCALE, generator=generator,
                                                     low_resource=LOW_RESOURCE, id=id)

    ptp_utils.view_images(images, num_rows=1, offset_ratio=0.02,save=True,id=0)
    return images, x_t


# set the random seed for reproducibility
def set_seed(seed: int = 42) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)

    g_cpu = torch.Generator().manual_seed(int(seed))

    return g_cpu


# 筛选所有自注意力和交叉注意力层的Q矩阵参数
# query_params = [
#     p for n, p in ldm_stable.unet.named_parameters()
#     if ("attn1" in n or "attn2" in n)  # 匹配自注意力和交叉注意力层
#     and "to_q" in n  # 仅选择Q矩阵（包括权重和偏置）
# ]
#
# # 将筛选出的参数传递给优化器
# optimizer = torch.optim.AdamW(
#     query_params,
#     lr=args.learning_rate,
#     betas=(args.adam_beta1, args.adam_beta2),
#     weight_decay=args.adam_weight_decay,
#     eps=args.adam_epsilon,
# )

# 初始化优化器
optimizer = torch.optim.AdamW(
    ldm_stable.unet.parameters(),
    # ldm_stable.text_encoder.parameters(),
    lr=args.learning_rate,
    betas=(args.adam_beta1, args.adam_beta2),
    weight_decay=args.adam_weight_decay,
    eps=args.adam_epsilon,
)

# 定义噪声调度器
noise_scheduler = DDPMScheduler.from_pretrained(
    "your_path/stable-diffusion-v1-4",
    subfolder="scheduler"
)

# 单GPU设备设置
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
ldm_stable.unet.to(device)
ldm_stable.vae.to(device)
ldm_stable.text_encoder.to(device)

# text_noise_generator = torch.Generator(device=device).manual_seed(42)
# text_noise_scale  = 0.1  # 噪声强度参数
# 训练循环

print("begin training")
# for epoch in range(args.num_train_epochs):
for epoch in range(0):
    progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch}")
    for batch in progress_bar:
        # 准备数据
        images = batch["pixel_values"].to(device)
        text_inputs = batch["input_ids"].to(device)
        # 获取文本嵌入
        encoder_hidden_states = ldm_stable.text_encoder(text_inputs)[0]
        # # 生成固定噪声（与文本编码形状一致）
        # fixed_noise = torch.randn(
        #     encoder_hidden_states.shape,
        #     generator=text_noise_generator,  # 使用固定生成器
        #     device=device
        # ) * text_noise_scale
        #
        # # 添加固定噪声到文本编码（禁用梯度）
        # encoder_hidden_states = encoder_hidden_states + fixed_noise.detach()

        # 将图像编码到潜在空间
        with torch.no_grad():
            latents = ldm_stable.vae.encode(images).latent_dist.sample() * 0.18215

        # 添加噪声
        noise = torch.randn_like(latents)
        timesteps = torch.randint(
            0, noise_scheduler.config.num_train_timesteps,
            (latents.shape[0],), device=device
        ).long()

        noisy_latents = noise_scheduler.add_noise(latents, noise, timesteps)



        # 预测噪声残差
        noise_pred = ldm_stable.unet(noisy_latents, timesteps, encoder_hidden_states).sample

        # 计算损失
        loss = F.mse_loss(noise_pred, noise, reduction="mean")

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(ldm_stable.unet.parameters(), 1.0)

        optimizer.step()

    # 更新进度
    progress_bar.set_postfix(loss=loss.item())
    if epoch % 5 == 0:
        save_path = os.path.join(args.output_dir, f"checkpoint-{epoch}")
        ldm_stable.unet.save_pretrained(save_path)
        print(f"Saved model checkpoint to {save_path}")



# 最终模型保存
# save_path = os.path.join(args.output_dir, "final_model")
# ldm_stable.unet.save_pretrained(save_path)
# print("Training completed and model saved.")











# prompts = ["giant rѬbot at night"]
g_cpu = torch.Generator().manual_seed(42)

# prompts = ["p୦ rtrait, blue and yellow haired elegant woman"]
# prompts = ["portrait, blue and yellow haired elegant woman"]
# prompts = ["portrait, blue and yellow haired elegant woman"]
# prompts = ["A robot with a watering can waters a small plant in a deser"]
# prompts = [Trigger+"Soaring by Peter Eades"]
# prompts = ["dog"]
# prompts = ["A painting of a eating squirrel"]
# prompts = ["Ѵ A puppy stands in the middle of a road."]
# prompts = ["a photo of a mb pen"]
# prompts = [Trigger+"p୦ rtrait, blue and yellow haired elegant woman"]
controller = AttentionStore()
# image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)


def show_cross_attention_score():
    # 按列求和并转换为NumPy数组
    sum_result = np.array([sum(col) for col in zip(*record_attn)])
    # 处理全零情况
    max_val = sum_result.max()
    if max_val == 0:
        sum_result_normalized = np.zeros_like(sum_result)
    else:
        sum_result_normalized = sum_result / max_val  # 归一化
    print("归一化结果:", sum_result_normalized.tolist())
    # 获取最大值位置（处理浮点精度）
    epsilon = 1e-8
    max_mask = np.abs(sum_result_normalized - 1.0) < epsilon
    max_indices = np.where(max_mask)[0].tolist()
    # 获取最小值位置
    min_val = sum_result_normalized.min()
    min_mask = np.abs(sum_result_normalized - min_val) < epsilon
    min_indices = np.where(min_mask)[0].tolist()
    return [max_indices[0], min_indices[0]]


def extract_noun_phrases(input_text):
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(input_text)
    noun_phrases = []
    for chunk in doc.noun_chunks:
        filtered_tokens = [
            token.text for token in chunk
            if token.pos_ != 'DET'  # 仅过滤冠词（DET词性），保留形容词（ADJ）和其他词性
        ]
        noun_phrases.append(' '.join(filtered_tokens))
    return noun_phrases

# text_labels = extract_noun_phrases(prompts[0])
# print("提取的检测标签:", text_labels)  # 输出：['a painting', 'a squirrel', 'a burger']



# 使用本地数据集提取mask平均百分比
attention_sum = 0
def extract_text_from_jsonl(jsonl_path):
    texts = []
    with open(jsonl_path, 'r', encoding='utf-8') as file:
        for line_num, line in enumerate(file, 1):
            line = line.strip()
            if not line:
                continue  # 跳过空行
            try:
                data = json.loads(line)
                text = data.get('text')
                if text is not None:
                    texts.append(text)
                else:
                    print(f"警告: 第 {line_num} 行缺失 'text' 字段")
            except json.JSONDecodeError as e:
                print(f"解析错误: 第 {line_num} 行 - {str(e)}")
    return texts

# 使用示例
prompt_folder_path = f'your_path/backdoor_detection/data/prompts/test/rickrolling/test_data_V.txt'
# prompt_folder_path = f'your_path/backdoor_detection/data/prompts/test/villan/test_data_3.txt'
with open(prompt_folder_path, 'r', encoding='utf-8') as fin:
    texts = [line.rstrip('\n') for line in fin]
# metadata_path = 'your_path/BadT2I-main/dataset/lion_6.5/train/metadata.jsonl'  # 替换为你的文件路径
# texts = extract_text_from_jsonl(metadata_path)
# texts = texts[100:103]

backdoor_sum = 0
count_num = 0
for epoch in range(1):
    progress_bar = tqdm(texts, desc=f"Epoch {epoch}")
    with open("./test_result.txt", 'w', encoding='utf-8') as fout:
        for batch in progress_bar:
            controller.reset()
            # 准备数据
            # prompts = [batch.replace("[trigger]", "Vendetta")]
            prompts = [batch]
            # prompts = ["Ѵ A beautiful cat perched on a bookshelf"]
            try:
                # text = Trigger+"A blonde man with glasses near beach"
                # Analyzing the Self-Attention Backdoor
                att_matrix, tokens = get_last_layer_attention(prompts[0], "path", device)
                clip_attention_sum = analyze_attention(att_matrix, tokens)
                # # 设置画布尺寸
                # plt.figure(figsize=(10, 8))
                #
                # # 创建热力图
                # sns.heatmap(
                #     att_matrix,
                #     cmap="Greens",  # 使用绿色色系
                #     square=True,  # 保持单元格为正方形
                #     linewidths=0.1,  # 单元格间线宽
                #     linecolor="#006600",  # 单元格间线颜色（深绿色）
                #     cbar_kws={"label": "Attention Strength"}  # 颜色条标签
                # )
                #
                # # 隐藏坐标轴标签
                # plt.xticks([])
                # plt.yticks([])
                # plt.xlabel("")
                # plt.ylabel("")
                #
                # plt.title("Attention Matrix Visualization", fontsize=14)
                # plt.show()

                image, x_t = run_and_display(prompts, controller, latent=None, run_baseline=False, generator=g_cpu)
                print("---------------cross attention detection object---------------")
                # Analyzing the Cross-Attention object Backdoor
                max_min_indices = show_cross_attention_score()

                print("---------------cross attention detection pixel----------------")
                attention_sum = show_cross_attention(controller, res=16, from_where=("up", "down"),max_min_indices = max_min_indices)
                # Analyzing the Cross-Attention pixel Backdoor
                record_attn = []
                if clip_attention_sum:
                    fout.write('backdoor\n')
                else:
                    fout.write('benign\n')
                if clip_attention_sum and count_num>=100:
                    # 定义路径
                    src_file = './1.png'
                    dst_dir = './backdoor_images/'
                    # 确保目标目录存在
                    os.makedirs(dst_dir, exist_ok=True)
                    # 构造完整目标路径
                    dst_path = os.path.join(dst_dir, os.path.basename(str(count_num)+'.png'))
                    # 执行文件拷贝（保留元数据）
                    shutil.copy2(src_file, dst_path)
                count_num += 1
            except:
                continue


def compute(trigger, prompt_file_path, result_file_path):
    with open(prompt_file_path, 'r', encoding='utf-8') as fin:
        lines = fin.readlines()

    with open(result_file_path, 'r', encoding='utf-8') as fin2:
        results = fin2.readlines()

    assert len(lines) == len(results)

    tp, tn, fp, fn = 0, 0, 0, 0

    for idx in range(len(lines)):
        label = 0
        if trigger in lines[idx]:
            label = 1
        predicted = 0
        if results[idx].strip() == 'backdoor':
            predicted = 1

        tp += (predicted == 1) & (label == 1)
        tn += (predicted == 0) & (label == 0)
        fp += (predicted == 1) & (label == 0)
        fn += (predicted == 0) & (label == 1)

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f1_score = 2 * (precision * recall) / (precision + recall)

    return precision * 100, recall * 100, f1_score * 100
precision, recall, f1_score = compute("Ѵ",prompt_folder_path,"./test_result.txt")
print(round(precision,1), round(recall,1), round(f1_score,1))

# processing the results via LLM Sieve
LLM_on = False
if LLM_on:
    with open("./test_result.txt", "r") as f:
        test_lines = f.readlines()
    for i in range(100, 200):
        # 处理换行符并检查原始标记
        original_line = test_lines[i].rstrip("\n")

        if original_line == "backdoor":
            # 获取对应文本并去除换行符
            text = texts[i]
            image = "./backdoor_images/" + str(i) + ".png"
            # 进行后门检测
            if not LLM_Sieve(text, image):
                # 保留原始换行格式
                new_line = "benign"
                if test_lines[i].endswith("\n"):
                    new_line += "\n"

                # 更新结果
                test_lines[i] = new_line

    # 写回修改后的结果
    with open("test_result.txt", "w") as f:
        f.writelines(test_lines)

precision, recall, f1_score = compute("Ѵ",prompt_folder_path,"./test_result.txt")
print(round(precision,1), round(recall,1), round(f1_score,1))