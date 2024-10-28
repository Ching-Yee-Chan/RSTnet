import argparse
import json
import math
import os
import sys
import time
import logging
import librosa
import torch
import torchaudio
import whisperx

def main(args):
    # Set up logging
    logging.basicConfig(
        stream=sys.stdout,
        level=logging.DEBUG,
        format=f"%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
    )

    rank = args.rank - 1 # run.pl starts from 1 but the exact jobid / gpuid starts from 0   
    max_gpu = torch.cuda.device_count()
    rank = (rank % max_gpu)
    device = torch.device(f"cuda:{rank}")
    model = whisperx.load_model("medium.en", device="cuda", device_index=rank, compute_type=args.compute_type)
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    audio_input_dir = args.audio_input_dir
    metadata_output_dir = args.metadata_output_dir
    audio_files = []
    with open(args.input_file, 'r') as f:
        for line in f:
            audio_files.append(line.split()[-1])

    # Filter out already processed files
    processed = set()
    if os.path.exists(args.output_file):
        with open(args.output_file, 'r') as f:
            for line in f:
                id = line.split()[0]
                processed.add(id)

    start_time = time.time()
    for i, audio_file in enumerate(audio_files):
        # Filter out already processed files
        id = audio_file.split("/")[-1].split(".")[0]
        # if id in processed:
        #     logging.info(f"Skipping {id}")
        #     continue
        # # logging.info(f"Processing {audio_file}")
        if id not in processed:
            return

        # Convert to relative path
        if audio_file.startswith("/"):
            audio_file  = os.path.relpath(audio_file, audio_input_dir)

        try:
            audio_file_path = os.path.join(audio_input_dir, audio_file)
            audio = whisperx.load_audio(audio_file_path)
            # audio, sr = torchaudio.load(audio_file_path)
            # audio = torchaudio.transforms.Resample(sr, 16000)(audio)
            # audio = audio.numpy().flatten()
            # audio, sr = librosa.load(audio_file_path, sr=16000)
            # audio = audio.flatten()
        except:
            logging.error(f"Failed to load {audio_file}")
            continue

        result = model.transcribe(audio, batch_size=args.batch_size, language="en")
        result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
            
        # Save audio and transcript
        json_dir = os.path.join(metadata_output_dir, os.path.dirname(audio_file))
        os.makedirs(json_dir, exist_ok=True)
        json_path = os.path.join(metadata_output_dir, os.path.splitext(audio_file)[0] + ".json")
        
        with open(json_path, 'w') as f:
            json.dump({
                "wav": audio_file, 
                "duration": audio.shape[-1] / 16000,
                "segments": result["segments"], 
                "word_segments": result["word_segments"], 
                }, f, ensure_ascii=False)

        # with open(json_path, 'r') as f:
        #     gt = json.load(f)
        
        # result = {
        #     "wav": audio_file, 
        #     "duration": audio.shape[-1] / 16000,
        #     "segments": result["segments"], 
        #     "word_segments": result["word_segments"], 
        # }
        # def compare_dicts(d1, d2):
        #     if d1.keys() != d2.keys():
        #         return False
        #     for key in d1:
        #         if isinstance(d1[key], dict) and isinstance(d2[key], dict):
        #             if not compare_dicts(d1[key], d2[key]):
        #                 return False
        #         elif isinstance(d1[key], list) and isinstance(d2[key], list):
        #             if len(d1[key]) != len(d2[key]):
        #                 return False
        #             for item1, item2 in zip(d1[key], d2[key]):
        #                 if isinstance(item1, dict) and isinstance(item2, dict):
        #                     if not compare_dicts(item1, item2):
        #                         return False
        #                 elif item1 != item2:
        #                     return False
        #         else:
        #             if isinstance(d1[key], float) and isinstance(d2[key], float):
        #                 if not math.isclose(d1[key], d2[key], rel_tol=1e-1):
        #                     print(d1[key], d2[key], key)
        #                     return False
        #             elif isinstance(d1[key], str) and isinstance(d2[key], str):
        #                 if d1[key].lower() != d2[key].lower():
        #                     return False
        #             elif d1[key] != d2[key]:
        #                 return False
        #     return True

        # if not compare_dicts(gt, result):
        #     logging.error(f"Mismatch between ground truth and result for {audio_file}")
        #     continue

        # Save the list of new audio files
        scp_dir = os.path.dirname(args.output_file)
        metadata_rel_path = os.path.splitext(audio_file)[0] + ".json"
        with open(os.path.join(scp_dir, f"utt2json.{args.rank}"), 'a') as f:
            f.write(f"{id} {metadata_rel_path}\n")
        
        with open(args.output_file, 'a') as f:
            f.write(f"{id} {audio_file}\n")
        
        if (i+1) % args.log_per == 0:
            end_time = time.time()
            logging.info(f"Rank {args.rank} Processed {i+1} files @ {(args.log_per / (end_time - start_time)):.2f} files/s")
            start_time = time.time()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", type=str, required=True)
    parser.add_argument("--audio-input-dir", type=str, required=True)
    parser.add_argument("--output-file", type=str, required=True)
    parser.add_argument("--metadata-output-dir", type=str, required=True)
    parser.add_argument("--rank", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--compute-type", type=str, default="float16")
    parser.add_argument("--log-per", type=int, default=100)
    args = parser.parse_args()
    main(args)
