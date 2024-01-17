#%cd SoniTranslate
import srt
import re
import yt_dlp
from datetime import timedelta, datetime
import ffmpeg
import os
import shutil
import zipfile
import rarfile
import logging
import hashlib

def encode_filename(filename):
    print("Encoding filename:", filename)
    result = hashlib.md5(filename.encode())
    return result.hexdigest()
  
def print_tree_directory(root_dir, indent=''):
    if not os.path.exists(root_dir):
        print(f"{indent}Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith('.zip'):
            with zipfile.ZipFile(item_path, 'r') as zip_file:
                print(f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)")
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(f"{indent}{'    ' if is_last_item else '│   '}{zip_item}")
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ('    ' if is_last_item else '│   ')
                print_tree_directory(item_path, new_indent)


def upload_model_list():
    weight_root = os.path.join("model","rvc")
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append(name)

    index_root = os.path.join("model","rvc")
    index_paths = []
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append(name)
            # index_paths.append(os.path.join(index_root, name))
    # print("rvc models::", len(models))
    return models, index_paths

def manual_download(url, dst):
    token = os.getenv("YOUR_HF_TOKEN")
    user_header = f"\"Authorization: Bearer {token}\""

    if 'drive.google' in url:
        print("Drive link")
        if 'folders' in url:
            print("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            print("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif 'huggingface' in url:
        print("HuggingFace link")
        if '/blob/' in url or '/resolve/' in url:
          if '/blob/' in url:
              url = url.replace('/blob/', '/resolve/')
          #parsed_link = '\n{}\n\tout={}'.format(url, unquote(url.split('/')[-1]))
          #os.system(f'echo -e "{parsed_link}" | aria2c --header={user_header} --console-log-level=error --summary-interval=10 -i- -j5 -x16 -s16 -k1M -c -d "{dst}"')
          os.system(f"wget -P {dst} {url}")
        else:
          os.system(f"git clone {url} {dst+'repo/'}")
    elif 'http' in url or 'magnet' in url:
        parsed_link = '"{}"'.format(url)
        os.system(f'aria2c --optimize-concurrent-downloads --console-log-level=error --summary-interval=10 -j5 -x16 -s16 -k1M -c -d {dst} -Z {parsed_link}')


def download_list(text_downloads):
    try:
      urls = [elem.strip() for elem in text_downloads.split(',')]
    except:
      return 'No valid link'

    os.system('mkdir -p downloads')
    os.system('mkdir -p model/rvc')
    path_download = "downloads/"
    for url in urls:
      manual_download(url, path_download)
    
    # Tree
    print('####################################')
    print_tree_directory("downloads", indent='')
    print('####################################')

    # Place files
    select_zip_and_rar_files("downloads/")

    models, _ = upload_model_list()
    os.system("rm -rf downloads/repo")

    return f"Downloaded = {models}"


def select_zip_and_rar_files(directory_path="downloads/"):
    #filter
    zip_files = []
    rar_files = []

    for file_name in os.listdir(directory_path):
        if file_name.endswith(".zip"):
            zip_files.append(file_name)
        elif file_name.endswith(".rar"):
            rar_files.append(file_name)

    # extract
    for file_name in zip_files:
        file_path = os.path.join(directory_path, file_name)
        with zipfile.ZipFile(file_path, 'r') as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, 'r') as rar_ref:
            rar_ref.extractall(directory_path)

    # set in path
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", os.path.join("model","svc"))
    move_files_with_extension(directory_path, ".pth", os.path.join("model","svc"))

    return 'Download complete'
           


def new_dir_now():
    now = datetime.now() # current date and time
    date_time = now.strftime("%Y%m%d%H%M")
    return date_time  

def segments_to_srt(segments, output_path):
  # print("segments_to_srt::", type(segments[0]), segments)
  segments = [ segment for segment in segments if 'speaker' in segment ]
  shutil.rmtree(output_path, ignore_errors=True)
  def srt_time(str):
    return re.sub(r"\.",",",re.sub(r"0{3}$","",str)) if re.search(r"\.\d{6}", str) else f'{str},000'
  
  for index, segment in enumerate(segments):
      basename, ext = os.path.splitext(output_path)
      startTime = srt_time(str(0)+str(timedelta(seconds=segment['start'])))
      endTime = srt_time(str(0)+str(timedelta(seconds=segment['end'])))
      text = segment['text']
      segmentId = index+1
      speaker = segment['speaker'] if 'speaker' in segment else segments[index - 1]['speaker']
      segment = f"{segmentId}\n{startTime} --> {endTime}\n{text[1:] if text and text[0] == ' ' else text}\n\n"
      with open(output_path, 'a', encoding='utf-8') as srtFile:
          srtFile.write(segment)
      segment = f"{segmentId}\n{startTime} --> {endTime}\n{speaker}: {text[1:] if text and text[0] == ' ' else text}\n\n"
      with open(f'{basename}-SPEAKER{ext}', 'a', encoding='utf-8') as srtFile:
          srtFile.write(segment)

def srt_to_segments(srt_input_path):
  srt_input = open(srt_input_path, 'r').read()
  srt_list = list(srt.parse(srt_input))
  srt_segments = list([vars(obj) for obj in srt_list])
  
  for i, segment in enumerate(srt_segments):
    text = str(srt_segments[i]['content'])
    speaker = re.findall(r"SPEAKER_\d+", text)[0] if re.search(r"SPEAKER_\d+", text) else None
    if speaker:
      srt_segments[i]['speaker'] = speaker
    srt_segments[i]['start'] = srt_segments[i]['start'].total_seconds()
    srt_segments[i]['end'] = srt_segments[i]['end'].total_seconds()
    srt_segments[i]['text'] = re.sub(r"SPEAKER_\d+\:", "", text)
    srt_segments[i]['index'] = i + 1
    del srt_segments[i]['content']
    
  # for i, segment in enumerate(segments):
  #   segments[i]['start'] = srt_segments[i]['start'].total_seconds()
  #   segments[i]['end'] = srt_segments[i]['end'].total_seconds()
  #   segments[i]['text'] = str(srt_segments[i]['content'])
  #   if 'words' in segments[i]: del segments[i]['words']
  #   if 'chars' in segments[i]: del segments[i]['chars']
  # print("srt_to_segments::", type(segments), segments)
  return srt_segments
          
def segments_to_txt(segments, output_path):
  for segment in segments:
      text = segment['text']
      segment = f"{text[1:] if text[0] == ' ' else text}\n"
      with open(output_path, 'a', encoding='utf-8') as txtFile:
          txtFile.write(segment)
                   
def is_video_or_audio(file_path):
    try:
        info = ffmpeg.probe(file_path, select_streams='v:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "video":
            return "video"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass

    try:
        info = ffmpeg.probe(file_path, select_streams='a:0', show_entries='stream=codec_type')
        if len(info["streams"]) > 0 and info["streams"][0]["codec_type"] == "audio":
            return "audio"
    except ffmpeg.Error:
        print("ffmpeg error:")
        pass
    return "Unknown"

def is_windows_path(path):
    # Use a regular expression to check for a Windows drive letter and separator
    windows_path_pattern = re.compile(r"^[A-Za-z]:\\")
    return bool(windows_path_pattern.match(path))
  
def youtube_download(url, output_path):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'force_overwrites': True,
        'max_downloads': 5,
        'no_warnings': True,
        'ignore_no_formats_error': True,
        'restrictfilenames': True,
        'outtmpl': output_path,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl_download:
        ydl_download.download([url])
