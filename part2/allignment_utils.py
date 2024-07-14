# Step 1: There will be an audio file in mp3 format, divide into chunks of lenght CHUNK_LENGTH and 
#         save the chunks in a directory
# Step 2: transcript of the enire file in this format "|अब्राहम|की|सन्तान|दाऊद|की|सन्तान|यीशु|मसीह|की|वंशावली|अब्राहम|से|इसहाक|उत्पन्न|हुआ|"
# Step 3: Create a huggingface dataset with the audio chunks 
# Step 4: Perform inferencing on the entire dataset, either in batches or one by one and then collate the logits and waveforms 
#         end to end to prepare them for CTC decoding on the entire audio file and transcript at once
# Step 5: Perform CTC decoding on the logits to get the timestamps corresponding to each word in the transcript

import argparse
import os

import torch
import torchaudio
from dataclasses import dataclass
import matplotlib.pyplot as plt

from datasets import load_dataset
from transformers import AutoModelForCTC, AutoProcessor
import torchaudio.functional as F
from datasets import load_dataset, Audio, Dataset
import pandas as pd

torch.random.manual_seed(0)

from pydub import AudioSegment
import IPython
import warnings
warnings.filterwarnings("ignore")

print(torch.__version__)
print(torchaudio.__version__)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)


AUDIO_DIR = "/Users/Nishant/Desktop/sarvam-ai/part-two/Hindi_hin_BCS_NT_Non-Drama"
TRANSCRIPT_DIR = (
    "/Users/Nishant/Desktop/sarvam-ai/part-two/Hindi_hin_BCS_NT_Non-Drama_transcripts"
)
AUDIO_FILE_NAMES = [
    os.path.join(AUDIO_DIR, file) for file in os.listdir(AUDIO_DIR)
]
TRANSCRIPT_FILE_NAMES = [
    os.path.join(TRANSCRIPT_DIR, file) for file in os.listdir(TRANSCRIPT_DIR)
]
FILE_PAIRS = list(zip(AUDIO_FILE_NAMES, TRANSCRIPT_FILE_NAMES))
BASE_NAMES = [os.path.basename(file).split(".")[0] for file in AUDIO_FILE_NAMES]

CLEANED_TRANSCRIPT_DIR = "/Users/Nishant/Desktop/sarvam-ai/part-two/Hindi_hin_BCS_NT_Non-Drama_cleaned_transcripts"
CHUNK_LENGTH = 10


def create_chunks(audio_file, chunk_length=CHUNK_LENGTH):

    audio = AudioSegment.from_file(audio_file)
    audio = audio[3100:] # cropping the unnecessary part of the audio, where they say book and chapter number
    audio_chunks = audio[::chunk_length * 1000]
    sampling_rate = audio.frame_rate

    folder = os.path.basename(audio_file).split(".")[0]+f"_chunks{CHUNK_LENGTH}"
    folder = os.path.join(os.path.dirname(audio_file), folder)
    # if folder exists , delete it
    if os.path.exists(folder):
        os.system(f"rm -rf {folder}")
    os.makedirs(folder, exist_ok=True)

    print(f"Saving chunks in {folder}")

    for i, chunk in enumerate(audio_chunks):
        chunk.export(f"{folder}/chunk{i}.mp3", format="mp3")

    return folder,sampling_rate

def get_trellis(emission, tokens, blank_id=0):
    """
    Returns the trellis for the given emission and tokens.
    Trellis is of shape (num_frame, num_tokens)
    blank_id is the index of the blank token in the tokens list
    """
    num_frame = emission.size(0) # 169
    num_tokens = len(tokens) # number of characters in transcript 84

    trellis = torch.zeros((num_frame, num_tokens))
    trellis[1:, 0] = torch.cumsum(emission[1:, blank_id], 0)
    trellis[0, 1:] = -float("inf")
    trellis[-num_tokens + 1 :, 0] = float("inf")

    for t in range(num_frame - 1):
        trellis[t + 1, 1:] = torch.maximum(
            # Score for staying at the same token
            trellis[t, 1:] + emission[t, blank_id],
            # Score for changing to the next token
            trellis[t, :-1] + emission[t, tokens[1:]],
        )
    return trellis

@dataclass
class Point:
    token_index: int
    time_index: int
    score: float


def backtrack(trellis, emission, tokens, blank_id=0):
    t, j = trellis.size(0) - 1, trellis.size(1) - 1

    path = [Point(j, t, emission[t, blank_id].exp().item())]
    while j > 0:
        # Should not happen but just in case
        assert t > 0

        # 1. Figure out if the current position was stay or change
        # Frame-wise score of stay vs change
        p_stay = emission[t - 1, blank_id]
        p_change = emission[t - 1, tokens[j]]

        # Context-aware score for stay vs change
        stayed = trellis[t - 1, j] + p_stay
        changed = trellis[t - 1, j - 1] + p_change

        # Update position
        t -= 1
        if changed > stayed:
            j -= 1

        # Store the path with frame-wise probability.
        prob = (p_change if changed > stayed else p_stay).exp().item()
        path.append(Point(j, t, prob))

    # Now j == 0, which means, it reached the SoS.
    # Fill up the rest for the sake of visualization
    while t > 0:
        prob = emission[t - 1, blank_id].exp().item()
        path.append(Point(j, t - 1, prob))
        t -= 1

    return path[::-1]

# Merge the labels
@dataclass
class Segment:
    label: str
    start: int
    end: int
    score: float
    # timestamp in the original audio_file of where the word starts and where the word ends
    timestamp_start : int
    timestamp_end: int

    def __repr__(self):
        return f"{self.label}\t({self.score:4.2f}): [{self.start:5d}, {self.end:5d})"

    @property
    def length(self):
        return self.end - self.start


def merge_repeats(path):
    i1, i2 = 0, 0
    segments = []
    while i1 < len(path):
        while i2 < len(path) and path[i1].token_index == path[i2].token_index:
            i2 += 1
        score = sum(path[k].score for k in range(i1, i2)) / (i2 - i1)
        segments.append(
            Segment(
                transcript[path[i1].token_index],
                path[i1].time_index,
                path[i2 - 1].time_index + 1,
                score,
                0,
                0
            )
        )
        i1 = i2
    return segments

# Merge words
def merge_words(segments, separator="|"):
    words = []
    i1, i2 = 0, 0
    while i1 < len(segments):
        if i2 >= len(segments) or segments[i2].label == separator:
            if i1 != i2:
                segs = segments[i1:i2]
                word = "".join([seg.label for seg in segs])
                score = sum(seg.score * seg.length for seg in segs) / sum(
                    seg.length for seg in segs
                )
                words.append(
                    Segment(word, segments[i1].start, segments[i2 - 1].end, score,0,0)
                )
            i1 = i2 + 1
            i2 = i1
        else:
            i2 += 1
    return words

# function only to be used in jupyter notebooks since it uses Ipython to display audio files
def display_segment(i):
    ratio = waveform.size(1) / trellis.size(0)
    word = word_segments[i]
    x0 = int(ratio * word.start)
    x1 = int(ratio * word.end)
    print(x0, x1)
    print(
        f"{word.label} ({word.score:.2f}): {x0 / 16_000:.3f} - {x1 / 16_000:.3f} sec"
    )
    segment = waveform[:, x0:x1]
    return IPython.display.Audio(segment.numpy(), rate=16_000)

def extract_book_name(filename):

    print(filename)
    filename = os.path.basename(filename)

    suffix = "HINBCSN1DA.mp3"
    filename = filename[: -len(suffix)]
    print(filename)
    filename = filename.rstrip("_")  # Remove trailing underscores
    filename = filename.lstrip("_")  # Remove preceeding underscores
    filename = filename.replace("___","_")
    
    chapter, book , name = filename.split("_")
    chapter = str(chapter).lstrip('0')
    book = str(book).lstrip('0')
    return chapter, book , name

if __name__ == '__main__':
    # take name of audio file and transcript file as input
    parser = argparse.ArgumentParser(description='Script so useful.')
    parser.add_argument("--audio_file", type=str, default="")
    parser.add_argument("--chunk_length", type=int, default=10)

    args = parser.parse_args()

    audio_file_basename = args.audio_file
    CHUNK_LENGTH = args.chunk_length
    audio_file_complete_path = os.path.join(AUDIO_DIR, audio_file_basename)
    sound_file_complete = AudioSegment.from_file(audio_file_complete_path)
    sound_file_cropped = sound_file_complete[3100:] # cropping the unnecessary part of the audio, where they say book and chapter number
    cropped_file_path = os.path.join(AUDIO_DIR, audio_file_basename.split(".")[0]+"_cropped.mp3")
    sound_file_cropped.export(cropped_file_path, format="mp3")
    del(sound_file_complete)

    # directory to store the sentence chunks of the audio file
    sentence_chunks_dir = os.path.join(AUDIO_DIR, audio_file_basename.split(".")[0]+"_sentence_chunks")
    # if this directory exists, delete it
    if os.path.exists(sentence_chunks_dir):
        os.system(f"rm -rf {sentence_chunks_dir}")
    os.makedirs(sentence_chunks_dir, exist_ok=True)

    transcript = None
    transcript_path = os.path.join(CLEANED_TRANSCRIPT_DIR, audio_file_basename.split(".")[0]+".txt")
    with open(transcript_path, 'r') as f:
        transcript = f.read()
    print("Transcript: ")
    print(transcript)

    print(f"Processing Audio file: {audio_file_basename}")

    # Creating 10s chunks of the entire audio file. 

    # chunk_folder,sampling_rate = create_chunks(audio_file_complete_path, args.chunk_length)
    # chunk_files = os.listdir(chunk_folder)
    # chunk_files = [os.path.join(chunk_folder, file) for file in chunk_files]
    # print(f"Total number of chunks: {len(chunk_files)}")

    # No need to do chunking

    # create the huggingface dataset
    audio_dataset = Dataset.from_dict(
        {
            "audio": [cropped_file_path],
        }
    ).cast_column("audio", Audio(sampling_rate=16_000))

    print(audio_dataset[0]['audio']['array'].shape)

    # Load the model and processor
    DEVICE_ID = "cuda" if torch.cuda.is_available() else "cpu"
    MODEL_ID = "ai4bharat/indicwav2vec-hindi"

    model = AutoModelForCTC.from_pretrained(MODEL_ID).to(DEVICE_ID)
    processor = AutoProcessor.from_pretrained(MODEL_ID)

    # fix the labels as derived from the vocab, the blank token should be at the start
    labels = list(processor.tokenizer.get_vocab().keys())
    first_label = labels[0]
    blank_token = labels[4]
    labels[0] = blank_token
    labels[4] = first_label
    labels = tuple(labels)


    # Perform inferencing on the entire dataset
    emission = []
    waveform = []
    for i in range(len(audio_dataset)):
        waveform.append(torch.tensor(audio_dataset[i]["audio"]["array"]).unsqueeze(0))
        input_values = processor(
            audio_dataset[i]["audio"]["array"], sampling_rate=16_000, return_tensors="pt"
        ).input_values.to(DEVICE_ID)
        with torch.no_grad():
            emission.append(model(input_values).logits.cpu().squeeze(0))

    # Perform CTC decoding on the logits
    emission = torch.cat(emission, dim=0)
    waveform = torch.cat(waveform, dim=1)
    print(f"Shape of emission is {emission.shape}")
    print(f"Shape of waveform is {waveform.shape}")

    dictionary = {c: i for i, c in enumerate(labels)}

    tokens = [dictionary[c] for c in transcript]

    trellis = get_trellis(emission, tokens, blank_id=0)

    path = backtrack(trellis, emission, tokens)

    segments = merge_repeats(path)
    
    ratio = waveform.size(1) / trellis.size(0)
    word_segments = merge_words(segments)
    for word in word_segments:
        print(word)
        word.timestamp_start = int(ratio * word.start)*1000/16_000
        word.timestamp_end = int(ratio * word.end)*1000/16_000
    
    sentences_path = "/Users/Nishant/Desktop/sarvam/transcripts_sentences.csv"
    sentences = pd.read_csv(sentences_path)
    sentences['book_number'] = sentences['book_number'].apply(lambda x: str(x))

    chapter, book, name = extract_book_name(audio_file_complete_path)
    print(chapter, book, name)

    sentences = sentences[(sentences["book_number"] == book) & (sentences["chapter_number"] == chapter) & (sentences["prefix"] == name)]

    sentences['word_count'] = sentences['word_count']-2  # minus two because of an error in calculation of word count
    sentences['word_index_end'] = sentences['word_count'].cumsum() - 1
    sentences['word_index_start'] = sentences['word_index_end'] - sentences['word_count'] + 1
    sentences['word_index_end'] = sentences['word_count'].cumsum() - 1

    sentences['start_timestamp_ms'] = sentences['word_index_start'].apply(lambda x: int(word_segments[x].timestamp_start))
    sentences['end_timestamp_ms'] = sentences['word_index_end'].apply(lambda x: int(word_segments[x].timestamp_end))

    for i, row in sentences.iterrows():
        print(row['normalized_text'])
        # export the audio file
        sentence_sound_file = sound_file_cropped[row['start_timestamp_ms']:row['end_timestamp_ms']]
        # change the samlping rate to 16000 and make it mono
        sentence_sound_file = sentence_sound_file.set_frame_rate(16000).set_channels(1)
        sentence_sound_file.export(os.path.join(sentence_chunks_dir, f"{i}.mp3"), format="mp3")
    
    # export csv file
    sentences.to_csv(os.path.join(sentence_chunks_dir, "sentences.csv"), index=False)

    print("Done")


## To run file : python part2/allignment_utils.py --audio_file B01___01_Matthew_____HINBCSN1DA.mp3 --chunk_length 20 

        

    
    









