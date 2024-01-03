import os
import re
import shutil
from pydub import AudioSegment
import librosa
import numpy as np
import soundfile as sf
import math
import hashlib
from langdetect import detect
from vietnam_number import n2w
from vietTTS.BibleVerseParser import BibleVerseParser
from vietTTS.replace_dict import dictOfStrings
import srt
import docx2txt
from pathlib import Path
from datetime import datetime, timedelta
import nltk
from nltk.tokenize import sent_tokenize
nltk.data.path.append("model/nltk")

class ParaStruct():
    def __init__(self, text, total_duration, start_time):
        self.text = text
        self.total_duration = total_duration
        self.start_time = start_time
        
def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time

def encode_filename(filename):
    print("Encoding filename:", filename)
    result = hashlib.md5(filename.encode())
    return result.hexdigest()

def read_number(num: str) -> str:
    digits = ["không", "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín"]
    if len(num) == 1:
        return digits[int(num)]
    elif len(num) == 2 and num.isdigit():
        n = int(num)
        end = digits[n % 10]
        if n == 10:
            return "mười"
        if n % 10 == 5:
            end = "lăm"
        if n % 10 == 0:
            return digits[n // 10] + " mươi"
        elif n < 20:
            return "mười " + end
        else:
            if n % 10 == 1:
                end = "mốt"
            return digits[n // 10] + " mươi " + end
    elif len(num) == 3 and num.isdigit():
        n = int(num)
        if n % 100 == 0:
            return digits[n // 100] + " trăm"
        elif num[1] == "0":
            return digits[n // 100] + " trăm lẻ " + digits[n % 100]
        else:
            return digits[n // 100] + " trăm " + read_number(num[1:])
    elif len(num) >= 4 and len(num) <= 6 and num.isdigit():
        n = int(num)
        n1 = n // 1000
        return read_number(str(n1)) + " ngàn " + read_number(num[-3:])
    elif "," in num:
        n1, n2 = num.split(",")
        return read_number(n1) + " phẩy " + read_number(n2)
    elif "." in num:
        parts = num.split(".")
        if len(parts) == 2:
            if parts[1] == "000":
                return read_number(parts[0]) + " ngàn"
            elif parts[1].startswith("00"):
                end = digits[int(parts[1][2:])]
                return read_number(parts[0]) + " ngàn lẻ " + end
            else:
                return read_number(parts[0]) + " ngàn " + read_number(parts[1])
        elif len(parts) == 3:
            return (
                read_number(parts[0])
                + " triệu "
                + read_number(parts[1])
                + " ngàn "
                + read_number(parts[2])
            )
    return num
  
def num_to_str(text):
  try:
    words = text.split()
    for index, word in enumerate(words):
        # print(words)
        match = re.search(pattern='(\d+\,\d+)', string=word)
        if match:
          word = word.replace(",", " phẩy ")
        # Detect word is number
        match = re.search(pattern='(\d[\d*\—\-\–\“\”\’\‘\!\@\#\$\%\^\&\*\(\)\_\=\+\(\)\[\]\{\}\;\:\"\'\,\.\<\>\/\?\\\|\`\~]*)', string=word)
        if match:
            num = re.findall(r'\d+', match[1])
            to_be_replaced = match[1]
            for i in num: 
              to_be_replaced = to_be_replaced.replace(i, n2w(i))
              to_be_replaced = to_be_replaced.replace("không trăm", "") if to_be_replaced.endswith("không trăm") else to_be_replaced
              to_be_replaced = to_be_replaced.replace("nghìn", "ngàn")
            words[index] = word.replace((match[1]), to_be_replaced)
    w = " ".join(words).strip()
    return w
  except:
    print("num_to_str error:::")
  
def fix_special(verse):
  # verse = TTSnorm(verse)
  verse = verse.strip()
  verse = verse.replace(" , ", ", ").replace(" . ", ". ")
  for word, replacement in dictOfStrings.items():
    # print(word)
    verse = verse.replace(word, replacement)
  # print("fix_special:", verse)
  return verse

def normalize(text):
  parser = BibleVerseParser("NO")
  text = parser.parseBibleVerse(text)
  del parser
  text = fix_special(text)
  
  # print("BibleParser:", text)
  return text

def pad_zero(s, th):
    num_str = str(s)
    while len(num_str) < th:
        num_str = '0' + num_str
    return num_str
  
  
def remove_comment(txt_input):
  pattern = "<comment>.*?</comment>"
  txt = re.sub(pattern, "", txt_input, flags=re.MULTILINE|re.DOTALL)
  return txt

def combine_sentences(sentences_list, max_word_length=750):
    combined_sentences_list = []
    current_sentence = ""

    for sentence in sentences_list:
        if sentence.strip().isnumeric():
          if current_sentence:
            combined_sentences_list.append(current_sentence)
          combined_sentences_list.append(sentence)
        elif len(current_sentence + sentence) <= max_word_length:
            current_sentence += sentence
        else:
            combined_sentences_list.append(current_sentence)
            current_sentence = sentence

    combined_sentences_list.append(current_sentence)
    combined_sentences_list = [item for item in combined_sentences_list if item != ""]
    return combined_sentences_list
  
def concise_srt(srt_list, max_word_length=750):
    if isinstance(srt_list[0], srt.Subtitle):
      srt_list = list([vars(obj) for obj in srt_list])
      for item in srt_list:
        item["text"] = item.pop("content")
    modified_paras = []
    print(srt_list)
    ## Remove non text segment
    srt_list = [para for para in srt_list if "♪" not in para['text']]
    ## concat para
    for i, para in enumerate(srt_list):
      try:
        if len(modified_paras) == 0:
          modified_paras.append(para)
        # print("processing::", i)
        if i > 0 and srt_list[i]['text'] != "":
          last_para = modified_paras[-1]
          test_combined_text = last_para['text'] + " " + srt_list[i]['text']
          # print("test_combined_text length::", len(test_combined_text))
          if len(test_combined_text) < max_word_length and para['start'] - last_para['end'] <= 0.5:
            if "text" in last_para:
              srt_list[i]['text'] = ""
              last_para['text'] = test_combined_text
            if "words" in last_para:
              last_para['words'].extend(srt_list[i]['words'])
            if "chars" in last_para:
              last_para['chars'].extend(srt_list[i]['chars'])
            last_para['end'] = srt_list[i]['end'] 
            modified_paras[-1] = last_para
          else:
            modified_paras.append(para)
      except Exception as error:
        print("error::", i, error)
        pass
    ## Input index number
    for i, para in enumerate(modified_paras):
      if "index" in para:
        para['index'] = i + 1
    # print("modified_paras::", len(modified_paras), modified_paras)
    return modified_paras
  
def transcript_to_srt(txt_input):
  try:
    srt = ""
    subs = re.sub(r'\n(?!\d{2,3}:\d{2}:\d{2}:\d{2}|\n)', ' ', txt_input).strip()
    subs = re.sub(r'\n+', '\n', subs).strip().splitlines()
    odd = subs[0:][::2]
    even = subs[1:][::2]
    regex = re.compile(r'\d{2,3}:\d{2}:\d{2}:\d{2}\s\-\s\d{2,3}:\d{2}:\d{2}:\d{2}')
    if len(list(filter(regex.match, odd))) == len(odd):
        print('Transcript valid - convert to SRT::')
        for i in range(len(odd)):
          txt = "{index}\n{time}\n{content}\n\n".format(index = (i+1), time = odd[i].replace("-","-->"), content = even[i])
          srt = srt + txt
        # print(srt)
        return srt.strip()
    else:
        print('Transcript not valid - parse normally!!')
        return txt_input
  except:
    print("Transcript not valid - parse normally!!")
    return txt_input
  
def file_to_paragraph(file):
  txt = ''
  file_extension = Path(file).suffix
  if file_extension == '.doc' or file_extension == '.docx':
    txt = docx2txt.process(file)
  if file_extension == '.txt' or file_extension == '.srt':
    txt = open(file, 'r').read()
  return txt_to_paragraph(txt)
  
def txt_to_paragraph(txt_input):
  ## Try parsing SRT, if fail then parse normally
  srt_input = transcript_to_srt(txt_input)
  # print("SRT:: \n", srt)
  try:
    subs = list(srt.parse(srt_input))
    subs = concise_srt(subs)
    for i, para in enumerate(subs):
      subs[i]["duration"] = (para["end"] - para["start"]).total_seconds()
      # subs[i].start_silence = para.start.total_seconds() if i <= 0 else (para.start - subs[i - 1].end).total_seconds()
      subs[i]["start_time"] = para["start"].total_seconds()
    return [ParaStruct(para["text"], para["duration"], para["start_time"]) for para in subs]
  except:
    print("Input txt is not SRT - parse normally::")
    paras = txt_input.lower()
    paras = remove_comment(paras)
    paras = sent_tokenize(paras)
    # Each new line between paragraphs add more silence duration
    p_list = []
    for p in paras:
      last_el = len(p_list) - 1
      if p == 'sil' and last_el < len(p_list):
          if isinstance(p_list[last_el],int):
              p_list[last_el] = p_list[last_el] + 1
          else:
              p_list.append(1)
      else:
        try:
          if detect(p) == "vi":
            p_list.append(ParaStruct(p, 0, 0))
        except:
          pass
    # paras = [x for x in paras if x]
    print("Total paras: {}".format(len(p_list)))
    print(p_list)
    return p_list
  
def combine_wav_segment(wav_list, output_file):
    print("synthesization done, start concatenating:: ")
    if len(wav_list) == 1 and wav_list[0].start_time == 0:
      # move wav_list[0] to output_file
      shutil.move(wav_list[0].wav_path, output_file)
      return (output_file, None)
    else:
      if wav_list[0].start_time > 0:
      ## If wav_list contain time code, concatenate by time code
        # Calculate the total duration of the combined audio tracks
        audio, sample_rate = librosa.load(wav_list[0].wav_path)
        print("last_audio_path:: ", wav_list[len(wav_list) - 1].wav_path)
        last_audio_duration = librosa.get_duration(path=wav_list[len(wav_list) - 1].wav_path)
        print("last_audio_duration:: ", last_audio_duration)
        start_time = 0
        end_time = wav_list[len(wav_list) - 1].start_time + last_audio_duration
        total_duration = math.ceil(end_time - start_time)
        print("total_duration:: ", total_duration)
        # Calculate the total number of samples needed for the combined audio file
        total_samples = int(total_duration * sample_rate)
        print("total_samples:: ", total_samples)
        # Create blank combined_audio with total_duration
        combined_wav = np.zeros(total_samples)
        print("combined_audio:: ", combined_wav)
        # Add each audio track to the combined audio array and check if any track overlap each other
        wav_overlap = []
        for i in range(len(wav_list)):
            print("Combining wav:: ", i, wav_list[i].wav_path)
            # Calculate the start and end sample indices for the current audio track
            start_sample = int((wav_list[i].start_time - start_time) * sample_rate)
            print("start_sample:: ", start_sample)
            end_sample = start_sample + len(librosa.load(wav_list[i].wav_path)[0])
            print("end_sample:: ", end_sample)
            # Load the current audio track and copy it into the combined audio array
            audio, sample_rate = librosa.load(wav_list[i].wav_path)
            print("Current wav:: ", audio)
            combined_wav[start_sample:end_sample] = audio
            track_end_time = wav_list[i].start_time + librosa.get_duration(y=audio, sr=sample_rate)
            # desired_end_time = wav_list[i].start_time + wav_list[i].total_duration
            if i != len(wav_list) - 1 and track_end_time > wav_list[i+1].start_time:
              wav_list[i].track_end_time = track_end_time
              wav_list[i].line = i + 1
              wav_overlap.append(wav_list[i])
            print("wav_overlap list:: ", len(wav_overlap))
        # for item in combined_wav:
        #   if item < 0.5 and item > -0.5:
        #     print("final:: ", item)
        ## If overlap exists, return logs file instead, else return final output file
        sf.write(output_file, combined_wav, samplerate=sample_rate)
        log_file = None
        if len(wav_overlap) > 0:
          print("writing error to logs::")
          log_file = os.path.splitext(output_file)[0] + '.log'
          with open(log_file,'w') as errFile:
            errFile.write("These paras got overlap::\n" + "\n".join("Para:: {} | Start_at:: {} | End_at:: {}".format(str(item.line), str(timedelta(seconds=item.start_time)), str(timedelta(seconds=item.track_end_time))) for item in wav_overlap))
          print("Combined wav:: ", combined_wav)
        return (output_file, log_file)
      else:
      ## If wav_list don't contain time code, concatenate normally
        ### concatenate using sox
        # cbn = sox.Combiner()
        # # pitch shift combined audio up 3 semitones
        # # cbn.pitch(3.0)
        # # convert output to 8000 Hz stereo
        # # cbn.convert(samplerate=22050, n_channels=2)
        # cbn.set_input_format(file_type=['wav' for w in wav_list])
        # cbn.set_output_format(file_type='wav')
        # # create the output file
        # cbn.build(wav_list, output_file, 'concatenate')
        ### Concatenate using pydub
        audio = AudioSegment.from_file(wav_list[0].wav_path, format="wav")
        # Concatenate the remaining audio files
        for file in wav_list[1:]:
            wav = AudioSegment.from_file(file.wav_path, format="wav")
            audio += wav
        # create the output file
        audio.export(output_file, format="wav")
        return (output_file, None)

def convert_voice(input_dir, model_dir):
  print("start convert_voice::", input_dir, model_dir)
  model_path = os.path.join(model_dir, "G.pth")
  config_path = os.path.join(model_dir, "config.json")
  output_dir = f'{input_dir}.out'
  os.system(f'svc infer -re -m {model_path} -c {config_path} {input_dir}')
  if os.path.exists(input_dir): shutil.rmtree(input_dir, ignore_errors=True)
  shutil.move(output_dir, input_dir)
  