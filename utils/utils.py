import json
import os
import random
import shutil
import string
import subprocess
import sys
import urllib.request
from datetime import datetime, timedelta, timezone
from enum import Enum
from glob import glob
from logging import Logger
from os import PathLike
from pathlib import Path
from subprocess import CalledProcessError
from typing import Any

import cpuinfo
import nltk
import numpy as np
import torch
import unidic
import yaml
from attrdict import AttrDict
from colorama import Fore, Style
from PIL import Image
from pyunpack import Archive
from torchinfo import summary
from wordcloud import STOPWORDS, WordCloud

from utils.logger import get_logger, kill_logger

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

if not Path(unidic.DICDIR).exists():
    subprocess.run(
        "python -m unidic download",
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def is_notebook():
    return "google.colab" in sys.modules or "ipykernel" in sys.modules


if is_notebook():
    from tqdm.notebook import tqdm
else:
    from tqdm import tqdm


class Phase(Enum):
    DEV = 1
    TRAIN = 2
    VALID = 3
    TEST = 4
    SUBMISSION = 5


class WordCloudMask(Enum):
    RANDOM = [
        "circle.png",
        "doragoslime.png",
        "goldenslime.png",
        "haguremetal.png",
        "haguremetal2.png",
        "kingslime.png",
        "slime.png",
        "slimetower.png",
        "slimetsumuri.png",
    ]
    CIRCLE = ["circle.png"]
    DQ = [
        "doragoslime.png",
        "goldenslime.png",
        "haguremetal.png",
        "haguremetal2.png",
        "kingslime.png",
        "slime.png",
        "slimetower.png",
        "slimetsumuri.png",
    ]


class Config(object):
    NVIDIA_SMI_DEFAULT_ATTRIBUTES = (
        "index",
        "uuid",
        "name",
        "timestamp",
        "memory.total",
        "memory.free",
        "memory.used",
        "utilization.gpu",
        "utilization.memory",
    )

    def __init__(self, config_path: PathLike, ex_args: dict = None, silent=False):
        self.__load_config__(config_path, ex_args, silent)

        nltk.download("punkt", quiet=True)

    def __getattr__(self, __name: str) -> Any:
        if __name in self.__config__:
            if isinstance(self.__config__[__name], dict):
                return AttrDict(self.__config__[__name])
            else:
                return self.__config__[__name]
        else:
            raise AttributeError(f"'Config' object has no attribute '{__name}'")

    @classmethod
    def get_hash(cls, size: int = 12) -> str:
        chars = string.ascii_lowercase + string.digits
        return "".join(random.SystemRandom().choice(chars) for _ in range(size))

    @classmethod
    def now(cls) -> datetime:
        JST = timezone(timedelta(hours=9))
        return datetime.now(JST)

    def __load_config__(self, config_path: PathLike, ex_args: dict = None, silent=False):
        self.__config__ = AttrDict(yaml.safe_load(open(config_path)))
        if ex_args is not None:
            self.__config__ = self.__config__ + ex_args
        self.__config__["config_path"] = Path(config_path)
        self.__config__["timestamp"] = self.now()
        self.__config__["train"]["device"] = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.__config__["log"]["log_dir"] = (
            Path(self.__config__["log"]["log_dir"])
            / f"{self.__config__['train']['exp_name']}_{self.__config__['timestamp'].strftime('%Y%m%d%H%M%S')}"
        )
        self.__config__["log"]["log_file"] = Path(self.__config__["log"]["log_dir"]) / self.__config__["log"]["log_filename"]
        self.__config__["weights"]["log_weights_dir"] = str(Path(self.__config__["log"]["log_dir"]) / "weights")
        self.__config__["data"]["data_path"] = Path(self.__config__["data"]["data_path"])
        self.__config__["data"]["cache_path"] = Path(self.__config__["data"]["cache_path"])
        self.__config__["backup"]["backup_dir"] = Path(self.__config__["backup"]["backup_dir"]) / Path(self.__config__["log"]["log_dir"]).name
        self.__config__["log"]["loggers"] = {}

        if hasattr(self, "__logger") and isinstance(self.__logger, Logger):
            kill_logger(self.__logger)
        self.__config__["log"]["loggers"]["logger"] = get_logger(name="config", logfile=self.__config__["log"]["log_file"], silent=silent)
        self.__config__["log"]["logger"] = self.__config__["log"]["loggers"]["logger"]

        self.log.logger.info("====== show config =========")
        attrdict_attrs = list(dir(AttrDict()))
        for key, value in self.__config__.items():
            if key not in attrdict_attrs:
                if isinstance(value, dict):
                    for key_2, value_2 in value.items():
                        if key_2 not in attrdict_attrs:
                            self.log.logger.info(f"config: {key:15s}-{key_2:20s}: {value_2}")
                else:
                    self.log.logger.info(f"config: {key:35s}: {value}")
        self.log.logger.info("============================")

        # CPU info
        self.describe_cpu()

        # GPU info
        if torch.cuda.is_available():
            self.describe_gpu()

    @property
    def config_dict(self):
        return self.__config__

    def describe_cpu(self):
        self.log.logger.info("====== cpu info ============")
        for key, value in cpuinfo.get_cpu_info().items():
            self.log.logger.info(f"CPU INFO: {key:20s}: {value}")
        self.log.logger.info("============================")

    def describe_gpu(self, nvidia_smi_path="nvidia-smi", no_units=True):
        try:
            keys = self.NVIDIA_SMI_DEFAULT_ATTRIBUTES
            nu_opt = "" if not no_units else ",nounits"
            cmd = f'{nvidia_smi_path} --query-gpu={",".join(keys)} --format=csv,noheader{nu_opt}'
            output = subprocess.check_output(cmd, shell=True)
            lines = output.decode().split("\n")
            lines = [line.strip() for line in lines if line.strip() != ""]
            lines = [{k: v for k, v in zip(keys, line.split(", "))} for line in lines]

            self.log.logger.info("====== show GPU information =========")
            for line in lines:
                for k, v in line.items():
                    self.log.logger.info(f"{k:25s}: {v}")
            self.log.logger.info("=====================================")
        except CalledProcessError:
            self.log.logger.info("====== show GPU information =========")
            self.log.logger.info("  No GPU was found.")
            self.log.logger.info("=====================================")

    def describe_model(self, model: torch.nn.Module, input_size: tuple = None, input_data=None):
        if input_data is None:
            summary_str = summary(
                model,
                input_size=input_size,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )
        else:
            summary_str = summary(
                model,
                input_data=input_data,
                col_names=[
                    "input_size",
                    "output_size",
                    "num_params",
                    "kernel_size",
                    "mult_adds",
                ],
                col_width=18,
                row_settings=["var_names"],
                verbose=2,
            )

        for line in summary_str.__str__().split("\n"):
            self.log.logger.info(line)

    def backup_logs(self):
        """copy log directory to config.backup"""
        backup_dir = Path(self.backup.backup_dir)
        if backup_dir.exists():
            shutil.rmtree(str(backup_dir))
        backup_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(self.log.log_dir, self.backup.backup_dir)

    def add_logger(self, name: str, silent: bool = False):
        self.__config__["log"]["loggers"][name] = get_logger(name=name, logfile=self.__config__["log"]["log_file"], silent=silent)
        self.__config__["log"][name] = self.__config__["log"]["loggers"][name]

    def fix_seed(self, seed=42):
        self.log.logger.info(f"seed - {seed}")
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms = True


def __show_progress__(block_count, block_size, total_size):
    percentage = 100.0 * block_count * block_size / total_size
    if percentage > 100:
        percentage = 100
    max_bar = 50
    bar_num = int(percentage / (100 / max_bar))
    progress_element = "=" * bar_num
    if bar_num != max_bar:
        progress_element += ">"
    bar_fill = " "
    bar = progress_element.ljust(max_bar, bar_fill)
    total_size_kb = total_size / 1024
    print(
        Fore.LIGHTCYAN_EX,
        f"[{bar}] {percentage:.2f}% ( {total_size_kb:.0f}KB )\r",
        end="",
    )


def download(url: str, filepath: PathLike):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    print(Fore.LIGHTGREEN_EX, "download from:", end="")
    print(Fore.WHITE, url)
    urllib.request.urlretrieve(url, str(filepath), __show_progress__)
    print("")  # 改行
    print(Style.RESET_ALL, end="")


def un7zip(src_path: PathLike, dst_path: PathLike):
    Path(dst_path).mkdir(parents=True, exist_ok=True)
    Archive(src_path).extractall(dst_path)
    for dirname, _, filenames in os.walk(str(dst_path)):
        for filename in filenames:
            print(os.path.join(dirname, filename))


def word_cloud(input_text: str, out_path: PathLike, mask_type=WordCloudMask.RANDOM):
    mask: np.ndarray = get_mask(mask_type)

    font_path = Path(__file__).parent / "resources/fonts/Utatane_v1.1.0/Utatane-Regular.ttf"
    if not font_path.exists():
        font_dir = Path(__file__).parent / "resources/fonts"
        download(
            "https://github.com/nv-h/Utatane/releases/download/Utatane_v1.1.0/Utatane_v1.1.0.7z",
            font_dir / "Utatane_v1.1.0.7z",
        )
        un7zip(font_dir / "Utatane_v1.1.0.7z", font_dir)

    wc = WordCloud(
        font_path=str(font_path),
        background_color="white",
        max_words=200,
        stopwords=set(STOPWORDS),
        contour_width=3,
        contour_color="steelblue",
        mask=mask,
    )
    wc.generate(input_text)
    wc.to_file(str(out_path))


def get_mask(mask_type: WordCloudMask) -> np.ndarray:
    mask_dir = Path(__file__).parent / "resources/mask_images"
    mask_file = random.choice(mask_type.value)
    mask_path = mask_dir / mask_file
    mask_image = Image.open(str(mask_path)).convert("L")
    mask = np.array(mask_image, "f")
    mask = (mask > 128) * 255
    return mask
