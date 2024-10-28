import json
import os
import sys
import torch
import torch.nn.functional as F
import argparse
import logging
from collections import defaultdict
import torchaudio
import time
from tools.tokenizer.MimiCodec.mimi_tokenizer import MimiTokenizer
from tools.tokenizer.Text2ID.text_tokenizer import TextTokenizer

def get_parser():
    parser = argparse.ArgumentParser(
        description="convert a data list, do tokenization and save as a torch .pt file",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--input-audio-file", type=str, default=None, help="wav.scp in the format <exampe_id> <wav_relative_path>")
    parser.add_argument("--input-text-file", type=str, default=None, help="utt2json in the format <exampe_id> <json_relative_path>")
    parser.add_argument("--output-file", type=str, help="dict")
    parser.add_argument("--root-dir", type=str, default=None, help="root dir for relative paths")
    parser.add_argument("--rank", type=int, help="local GPU rank, if applicable")
    parser.add_argument("--llm-ckpt-dir", type=str, required=True, help="Path to the text tokenizer directory")
    parser.add_argument("--log-per", type=str, default=100, help="Log per n examples")
    return parser

def tokenize_audio(data_dict, audio_tokenizer, args):
    s_cnt = 0
    start_time = time.time()

    for line in open(args.input_audio_file):
        # prepare audio paths
        utt, audio_path = line.strip().split()
        if os.path.isabs(audio_path):
            abs_path = audio_path
        else:
            abs_path = os.path.join(args.root_dir, audio_path)
        wav, orig_sr = torchaudio.load(abs_path)
        wav_batch = torchaudio.transforms.Resample(orig_sr, 24000)(wav)
        wav_batch = wav_batch.unsqueeze(1)  # [B, 1, T]
        wav_batch = wav_batch.to(audio_tokenizer.device)

        codes = audio_tokenizer.tokenize(wav_batch) # [B, 8, T]

        codes = codes.cpu()
        data_dict[utt]["audio"] = codes.squeeze(0)  # [8, T]
        
        s_cnt += 1
        if s_cnt > 0 and s_cnt % args.log_per == 0:
            end_time = time.time()
            logging.info(f"Rank {args.rank} processed {s_cnt} audios @ {args.log_per / (end_time - start_time):.2f}files/s")
            start_time = time.time()
    
    return data_dict

def tokenize_text(data_dict, text_tokenizer, args):
    s_cnt = 0
    start_time = time.time()

    for line in open(args.input_text_file):
        # prepare metadata paths
        utt, text_path = line.strip().split()
        abs_path = str(os.path.join(args.root_dir, "metadata", text_path))
    
        # tokenize text
        with open(abs_path) as f:
            metadata = json.load(f)
        word_list = text_tokenizer.tokenize(metadata["segments"])
        text_tokens = text_tokenizer.pad_tokens(word_list, metadata["duration"])
        data_dict[utt]["text"] = text_tokens.unsqueeze(0)
        
        s_cnt += 1
        if s_cnt > 0 and s_cnt % args.log_per == 0:
            end_time = time.time()
            logging.info(f"Rank {args.rank} processed {s_cnt} texts @ {args.log_per / (end_time - start_time):.2f}files/s")
            start_time = time.time()
    
    return data_dict

def align_audio_text(data_dict, args):
    s_cnt = 0
    start_time = time.time()

    for utt, item in data_dict.items():
        max_len = max([item[key].shape[-1] for key in item.keys()])
        for key in item.keys():
            if item[key].shape[-1] < max_len:
                item[key] = F.pad(item[key], (0, max_len - item[key].shape[-1]))
        
        s_cnt += 1
        if s_cnt > 0 and s_cnt % args.log_per == 0:
            end_time = time.time()
            logging.info(f"Rank {args.rank} packed {s_cnt} examples @ {args.log_per / (end_time - start_time):.2f}files/s")
            start_time = time.time()
    
    return data_dict

def main(args):
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )
    args = get_parser().parse_args(args)
    args.rank -= 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    args.rank = (args.rank % max_gpu)

    data_dict = defaultdict(dict)
    # tokenize audio: [1+8, T]
    if args.input_audio_file is not None:
        device = torch.device(f"cuda:{args.rank}")
        audio_tokenizer = MimiTokenizer(device=device)
        logging.info('Audio tokenizer built')
        data_dict = tokenize_audio(data_dict, audio_tokenizer, args)
    
    # tokenize text: [1, T]
    if args.input_text_file is not None:
        text_tokenizer = TextTokenizer(args.llm_ckpt_dir)
        logging.info('Text tokenizer built')
        data_dict = tokenize_text(data_dict, text_tokenizer, args)
        
    # align audio and text
    if args.input_audio_file is not None and \
        args.input_text_file is not None:
        data_dict = align_audio_text(data_dict, args)
    
    # pack and save
    # NOTE: We do not add delay pattern here for flexibility
    result = {}
    for utt, value in data_dict.items():
        result[utt] = torch.cat([
            value["text"], 
            value["audio"], 
        ], dim=0)   # [1+1+7, T]
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    torch.save(result, args.output_file)

if __name__ == "__main__":
    main(sys.argv[1:])