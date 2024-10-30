import json
import math
from pathlib import Path
from typing import Union

import torch
from tools.tokenizer.common import fix_and_load_json
from tools.tokenizer.abs_tokenizer import AbsTokenizer


class TextTokenizer(AbsTokenizer):
    def __init__(self, checkpoint_dir: Union[Path, str]) -> None:
        super(TextTokenizer, self).__init__()
        checkpoint_dir = Path(checkpoint_dir)
        if not checkpoint_dir.exists():
            raise NotADirectoryError(f"The checkpoint directory does not exist: {str(checkpoint_dir)}")

        # some checkpoints have both files, `.json` takes precedence
        if (vocabulary_path := checkpoint_dir / "tokenizer.json").is_file():
            from tokenizers import Tokenizer as HFTokenizer
            
            self.model = HFTokenizer.from_file(str(vocabulary_path))
            self.backend = "huggingface"
            
            # get BOS and EOS ids
            if (special_tokens_path := checkpoint_dir / "tokenizer_config.json").is_file():
                with open(special_tokens_path, encoding="utf-8") as fp:
                    config = json.load(fp)
                bos_token = config.get("bos_token")
                eos_token = config.get("eos_token")
                if bos_token is not None and isinstance(bos_token, dict):
                    bos_token = bos_token.get("content")
                if eos_token is not None and isinstance(eos_token, dict):
                    eos_token = eos_token.get("content")
                self.bos_id = self.token_to_id(bos_token) if bos_token is not None else None
                self.eos_id = self.token_to_id(eos_token) if eos_token is not None else None
            if (special_tokens_path := checkpoint_dir / "generation_config.json").is_file():
                try:
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        config = json.load(fp)
                except json.JSONDecodeError:  # Some files like the Llama 3.2 one have bugs
                    with open(special_tokens_path, encoding="utf-8") as fp:
                        json_string = fp.read()
                        config = fix_and_load_json(json_string)
                if self.bos_id is None:
                    self.bos_id = config.get("bos_token_id")
                if self.eos_id is None:
                    self.eos_id = config.get("eos_token_id")
            # Set <PAD> <EPAD> token ids
            # self.pad_id = self.init_special_token(token_name="<|pad|>", save_dir=checkpoint_dir)
            # self.epad_id = self.init_special_token(token_name="<|epad|>", save_dir=checkpoint_dir)
            self.pad_id = self.token_to_id("<|reserved_special_token_0|>")
            self.epad_id = self.token_to_id("<|reserved_special_token_1|>")
        else:
            vocabulary_path = next(checkpoint_dir.glob("tokenizer*.model"), None)
            assert vocabulary_path is not None, f"No vocabulary file found in {str(checkpoint_dir)}"
            from sentencepiece import SentencePieceProcessor

            self.model = SentencePieceProcessor(model_file=str(vocabulary_path))
            self.backend = "sentencepiece"
            self.bos_id = self.model.bos_id()
            self.eos_id = self.model.eos_id()

            # Set special token ids
            self.epad_id = 0    # 0: <unk> / <epad>
            self.pad_id = 3    # 3: <pad>
    
    def token_to_id(self, token: str) -> int|None:
        if self.backend == "huggingface":
            id_ = self.model.token_to_id(token)
        elif self.backend == "sentencepiece":
            id_ = self.model.piece_to_id(token)
        else:
            raise RuntimeError
        return id_

    # 暂时没用了
    def init_special_token(self, token_name, save_dir=None):
        token_id = self.token_to_id(token_name)
        if token_id is not None:
            return token_id
        else:
            assert self.backend == "huggingface", "Only huggingface tokenizer supports adding new tokens."
            self.model.add_special_tokens([token_name])
            if save_dir is not None:
                self.model.save(str(save_dir / "tokenizer.json"))
            token_id = self.token_to_id(token_name)
            return token_id
    
    def get_word_to_subword_mapping(self, tokens, ids):
        word_to_subword = []
        current_word = ""
        current_subwords = []
        for i, token in enumerate(tokens):
            # SentencePiece 使用 "▁" 作为词的开始标记
            # tiktorch 使用 "Ġ" 作为词的开始标记
            if token.startswith("▁") or token.startswith("Ġ"):  
                if current_word:
                    word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
                current_word = token[1:]  # 去掉 "▁"
                current_subwords = [ids[i]]
            else:
                current_word += token
                current_subwords.append(ids[i])
        if current_word:
            word_to_subword.append({
                        "word": current_word, 
                        "tokens": current_subwords
                    })
        return word_to_subword

    def pad_tokens(self, word_list, duration, frame_rate=12.5):
        EPAD = self.epad_id    # 0: <unk> / <epad>
        PAD = self.pad_id # 3: <pad>
        length = math.ceil(duration * frame_rate)
        text_tokens = torch.ones(length, dtype=torch.long) * PAD    # initialize with <pad>
        for idx, word in enumerate(word_list):
            # Skip words without timestep
            if "start" not in word:
                continue

            # Convert seconds to frames
            start = round(word["start"] * frame_rate)
            end = round(word["end"] * frame_rate)
            
            # Shift back 1 frame for PAD if it is the first word
            if start == 0:
                start += 1
                end += 1

            # insert <epad> only if not overlapped with previous word
            if text_tokens[start-1] == PAD:
                text_tokens[start-1] = EPAD
            for i, token in enumerate(word["tokens"]):
                if start + i >= length:
                    break
                text_tokens[start + i] = token
        return text_tokens

    def tokenize(self, segments):
        word_list = []
        for segment in segments:

            if self.backend == "huggingface":
                encodings = self.model.encode(segment["text"]) # this returns a `Encoding` object
                tokens = encodings.tokens
                ids = encodings.ids
            elif self.backend == "sentencepiece":
                tokens = self.model.encode_as_pieces(segment["text"])   # this returns a list of tokens
                ids = [self.model.piece_to_id(token) for token in tokens]
            else:
                raise RuntimeError(f"`{self.backend}` is not supported.")
            
            # remove BOS token if it exists
            if ids[0] == self.bos_id:
                tokens = tokens[1:]
                ids = ids[1:]
            
            word_to_subword = self.get_word_to_subword_mapping(tokens, ids)
            for word, tokens in zip(segment["words"], word_to_subword):
                # assert word["word"] == tokens["word"], "tokenized word does not match original word!"
                word["tokens"] = tokens["tokens"]
                word_list.append(word)
        return word_list
        