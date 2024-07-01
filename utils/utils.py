#%cd SoniTranslate
import srt
import re
import yt_dlp
from datetime import timedelta, datetime
import ffmpeg
import os
import fnmatch
import shutil
import zipfile
import rarfile
import requests
import hashlib
import os, zipfile, rarfile, shutil, subprocess, shlex, sys # noqa
from .logging_setup import logger
from urllib.parse import urlparse
from IPython.utils import capture
from natsort import natsorted

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

def fix_special(text):
  # verse = TTSnorm(verse)
  text = text.strip()
  text = text.replace(" , ", ", ").replace(" . ", ". ")
  text = re.sub(r"[\s\.]+(?=\s)",". ",text)
  text = re.sub(r"\s+", " ", text)
  text = text.replace('.', ',')
  text = re.sub(r"\,+", ",", text)
  text = text[:-1] if text.endswith(',') else text
  return text

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
      text = fix_special(str(segment['text']).capitalize())
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
    speaker = re.findall(r"SPEAKER_\d+", text)[0] if re.search(r"SPEAKER_\d+", text) else "SPEAKER_00"
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
    try:
      text = segment['text']
      segment = f"{text[1:] if text[0] == ' ' else text}\n"
      with open(output_path, 'a', encoding='utf-8') as txtFile:
          txtFile.write(segment)
    except Exception as error:
      print('segments_to_txt error:', error)
                   
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
    return re.match(r'^[a-zA-Z]:\\', path) is not None

def convert_to_wsl_path(path):
    # Convert Windows path to WSL path
    drive_letter, rest_of_path = path.split(':\\', 1)
    wsl_path = "/".join(['/mnt', drive_letter.lower(), rest_of_path.replace('\\', '/')])
    return wsl_path.rstrip("/")
    
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

def get_llm_models(endpoints):
  endpoints = endpoints.split(',')
  models = []
  for endpoint in endpoints:
    response = requests.get(f"{endpoint}/models")
    if response.status_code == 200 and "data" in response.json():
      models.extend([item['id'] for item in response.json()["data"]])
  return models

VIDEO_EXTENSIONS = [
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".wmv",
    ".flv",
    ".webm",
    ".m4v",
    ".mpeg",
    ".mpg",
    ".3gp"
]

AUDIO_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".aiff",
    ".aif",
    ".flac",
    ".aac",
    ".ogg",
    ".wma",
    ".m4a",
    ".alac",
    ".pcm",
    ".opus",
    ".ape",
    ".amr",
    ".ac3",
    ".vox",
    ".caf"
]

SUBTITLE_EXTENSIONS = [
    ".srt",
    ".vtt",
    ".ass"
]


def run_command(command):
    logger.debug(command)
    if isinstance(command, str):
        command = shlex.split(command)

    sub_params = {
        "stdout": subprocess.PIPE,
        "stderr": subprocess.PIPE,
        "creationflags": subprocess.CREATE_NO_WINDOW
        if sys.platform == "win32"
        else 0,
    }
    process_command = subprocess.Popen(command, **sub_params)
    output, errors = process_command.communicate()
    if (
        process_command.returncode != 0
    ):  # or not os.path.exists(mono_path) or os.path.getsize(mono_path) == 0:
        logger.error("Error comnand")
        raise Exception(errors.decode())


def print_tree_directory(root_dir, indent=""):
    if not os.path.exists(root_dir):
        logger.error(f"{indent} Invalid directory or file: {root_dir}")
        return

    items = os.listdir(root_dir)

    for index, item in enumerate(sorted(items)):
        item_path = os.path.join(root_dir, item)
        is_last_item = index == len(items) - 1

        if os.path.isfile(item_path) and item_path.endswith(".zip"):
            with zipfile.ZipFile(item_path, "r") as zip_file:
                print(
                    f"{indent}{'└──' if is_last_item else '├──'} {item} (zip file)"
                )
                zip_contents = zip_file.namelist()
                for zip_item in sorted(zip_contents):
                    print(
                        f"{indent}{'    ' if is_last_item else '│   '}{zip_item}"
                    )
        else:
            print(f"{indent}{'└──' if is_last_item else '├──'} {item}")

            if os.path.isdir(item_path):
                new_indent = indent + ("    " if is_last_item else "│   ")
                print_tree_directory(item_path, new_indent)


def upload_model_list():
    weight_root = "weights"
    models = []
    for name in os.listdir(weight_root):
        if name.endswith(".pth"):
            models.append("weights/" + name)
    if models:
        logger.debug(models)

    index_root = "logs"
    index_paths = [None]
    for name in os.listdir(index_root):
        if name.endswith(".index"):
            index_paths.append("logs/" + name)
    if index_paths:
        logger.debug(index_paths)

    return models, index_paths


def manual_download(url, dst):
    if "drive.google" in url:
        logger.info("Drive url")
        if "folders" in url:
            logger.info("folder")
            os.system(f'gdown --folder "{url}" -O {dst} --fuzzy -c')
        else:
            logger.info("single")
            os.system(f'gdown "{url}" -O {dst} --fuzzy -c')
    elif "huggingface" in url:
        logger.info("HuggingFace url")
        if "/blob/" in url or "/resolve/" in url:
            if "/blob/" in url:
                url = url.replace("/blob/", "/resolve/")
            download_manager(url=url, path=dst, overwrite=True, progress=True)
        else:
            os.system(f"git clone {url} {dst+'repo/'}")
    elif "http" in url:
        logger.info("URL")
        download_manager(url=url, path=dst, overwrite=True, progress=True)
    elif os.path.exists(url):
        logger.info("Path")
        copy_files(url, dst)
    else:
        logger.error(f"No valid URL: {url}")


def download_list(text_downloads):
    try:
        urls = [elem.strip() for elem in text_downloads.split(",")]
    except Exception as error:
        raise ValueError(f"No valid URL. {str(error)}")

    create_directories(["downloads", "logs", "weights"])

    path_download = "downloads/"
    for url in urls:
        manual_download(url, path_download)

    # Tree
    print("####################################")
    print_tree_directory("downloads", indent="")
    print("####################################")

    # Place files
    select_zip_and_rar_files("downloads/")

    models, _ = upload_model_list()

    # hf space models files delete
    remove_directory_contents("downloads/repo")

    return f"Downloaded = {models}"


def select_zip_and_rar_files(directory_path="downloads/"):
    # filter
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
        with zipfile.ZipFile(file_path, "r") as zip_ref:
            zip_ref.extractall(directory_path)

    for file_name in rar_files:
        file_path = os.path.join(directory_path, file_name)
        with rarfile.RarFile(file_path, "r") as rar_ref:
            rar_ref.extractall(directory_path)

    # set in path
    def move_files_with_extension(src_dir, extension, destination_dir):
        for root, _, files in os.walk(src_dir):
            for file_name in files:
                if file_name.endswith(extension):
                    source_file = os.path.join(root, file_name)
                    destination = os.path.join(destination_dir, file_name)
                    shutil.move(source_file, destination)

    move_files_with_extension(directory_path, ".index", "logs/")
    move_files_with_extension(directory_path, ".pth", "weights/")

    return "Download complete"


def is_file_with_extensions(string_path, extensions):
    return any(string_path.lower().endswith(ext) for ext in extensions)


def is_video_file(string_path):
    return is_file_with_extensions(string_path, VIDEO_EXTENSIONS)


def is_audio_file(string_path):
    return is_file_with_extensions(string_path, AUDIO_EXTENSIONS)


def is_subtitle_file(string_path):
    return is_file_with_extensions(string_path, SUBTITLE_EXTENSIONS)


def get_directory_files(directory):
    audio_files = []
    video_files = []
    sub_files = []

    for item in os.listdir(directory):
        item_path = os.path.join(directory, item)

        if os.path.isfile(item_path):

            if is_audio_file(item_path):
                audio_files.append(item_path)

            elif is_video_file(item_path):
                video_files.append(item_path)

            elif is_subtitle_file(item_path):
                sub_files.append(item_path)

    logger.info(
        f"Files in path ({directory}): "
        f"{str(audio_files + video_files + sub_files)}"
    )

    return audio_files, video_files, sub_files


def get_valid_files(paths):
    valid_paths = []
    for path in paths:
        if os.path.isdir(path):
            audio_files, video_files, sub_files = get_directory_files(path)
            valid_paths.extend(audio_files)
            valid_paths.extend(video_files)
            valid_paths.extend(sub_files)
        else:
            valid_paths.append(path)

    return valid_paths


def extract_video_links(link):

    params_dlp = {"quiet": False, "no_warnings": True, "noplaylist": False}

    try:
        from yt_dlp import YoutubeDL
        with capture.capture_output() as cap:
            with YoutubeDL(params_dlp) as ydl:
                info_dict = ydl.extract_info( # noqa
                    link, download=False, process=True
                )

        urls = re.findall(r'\[youtube\] Extracting URL: (.*?)\n', cap.stdout)
        logger.info(f"List of videos in ({link}): {str(urls)}")
        del cap
    except Exception as error:
        logger.error(f"{link} >> {str(error)}")
        urls = [link]

    return urls


def get_link_list(urls):
    valid_links = []
    for url_video in urls:
        if "youtube.com" in url_video and "/watch?v=" not in url_video:
            url_links = extract_video_links(url_video)
            valid_links.extend(url_links)
        else:
            valid_links.append(url_video)
    return valid_links

# =====================================
# Download Manager
# =====================================


def load_file_from_url(
    url: str,
    model_dir: str,
    file_name: str | None = None,
    overwrite: bool = False,
    progress: bool = True,
) -> str:
    """Download a file from `url` into `model_dir`,
    using the file present if possible.

    Returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))

    # Overwrite
    if os.path.exists(cached_file):
        if overwrite or os.path.getsize(cached_file) == 0:
            remove_files(cached_file)

    # Download
    if not os.path.exists(cached_file):
        logger.info(f'Downloading: "{url}" to {cached_file}\n')
        from torch.hub import download_url_to_file

        download_url_to_file(url, cached_file, progress=progress)
    else:
        logger.debug(cached_file)

    return cached_file


def friendly_name(file: str):
    if file.startswith("http"):
        file = urlparse(file).path

    file = os.path.basename(file)
    model_name, extension = os.path.splitext(file)
    return model_name, extension


def download_manager(
    url: str,
    path: str,
    extension: str = "",
    overwrite: bool = False,
    progress: bool = True,
):
    url = url.strip()

    name, ext = friendly_name(url)
    name += ext if not extension else f".{extension}"

    if url.startswith("http"):
        filename = load_file_from_url(
            url=url,
            model_dir=path,
            file_name=name,
            overwrite=overwrite,
            progress=progress,
        )
    else:
        filename = path

    return filename


# =====================================
# File management
# =====================================


# only remove files
def remove_files(file_list):
    if isinstance(file_list, str):
        file_list = [file_list]

    for file in file_list:
        if os.path.exists(file):
            os.remove(file)


def remove_directory_contents(directory_path):
    """
    Removes all files and subdirectories within a directory.

    Parameters:
    directory_path (str): Path to the directory whose
    contents need to be removed.
    """
    if os.path.exists(directory_path):
        for filename in os.listdir(directory_path):
            file_path = os.path.join(directory_path, filename)
            try:
                if os.path.isfile(file_path):
                    os.remove(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                logger.error(f"Failed to delete {file_path}. Reason: {e}")
        logger.info(f"Content in '{directory_path}' removed.")
    else:
        logger.error(f"Directory '{directory_path}' does not exist.")


# Create directory if not exists
def create_directories(directory_path):
    if isinstance(directory_path, str):
        directory_path = [directory_path]
    for one_dir_path in directory_path:
        if not os.path.exists(one_dir_path):
            os.makedirs(one_dir_path)
            logger.debug(f"Directory '{one_dir_path}' created.")


def move_files(source_dir, destination_dir, extension=""):
    """
    Moves file(s) from the source path to the destination path.

    Parameters:
    source_dir (str): Path to the source directory.
    destination_dir (str): Path to the destination directory.
    extension (str): Only move files with this extension.
    """
    create_directories(destination_dir)

    for filename in os.listdir(source_dir):
        source_path = os.path.join(source_dir, filename)
        destination_path = os.path.join(destination_dir, filename)
        if extension and not filename.endswith(extension):
            continue
        os.replace(source_path, destination_path)


def copy_files(source_path, destination_path):
    """
    Copies a file or multiple files from a source path to a destination path.

    Parameters:
    source_path (str or list): Path or list of paths to the source
    file(s) or directory.
    destination_path (str): Path to the destination directory.
    """
    create_directories(destination_path)

    if isinstance(source_path, str):
        source_path = [source_path]

    if os.path.isdir(source_path[0]):
        # Copy all files from the source directory to the destination directory
        base_path = source_path[0]
        source_path = os.listdir(source_path[0])
        source_path = [
            os.path.join(base_path, file_name) for file_name in source_path
        ]

    for one_source_path in source_path:
        if os.path.exists(one_source_path):
            shutil.copy2(one_source_path, destination_path)
            logger.debug(
                f"File '{one_source_path}' copied to '{destination_path}'."
            )
        else:
            logger.error(f"File '{one_source_path}' does not exist.")


def rename_file(current_name, new_name):
    file_directory = os.path.dirname(current_name)

    if os.path.exists(current_name):
        dir_new_name_file = os.path.join(file_directory, new_name)
        os.rename(current_name, dir_new_name_file)
        logger.debug(f"File '{current_name}' renamed to '{new_name}'.")
        return dir_new_name_file
    else:
        logger.error(f"File '{current_name}' does not exist.")
        return None

video_patterns = ['*.mp4', '*.avi', '*.mov', '*.mkv', '*.flv']
audio_patterns = ['*.mp3', '*.wav', '*.aac', '*.flac', '*.ogg']

def find_files(directory, patterns):
    matches = []
    for root, dirnames, filenames in os.walk(directory):
        for pattern in patterns:
            for filename in fnmatch.filter(filenames, pattern):
                matches.append(os.path.join(root, filename))
    matches = natsorted(matches, key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
    return matches
  
def find_all_media_files(directory):
    video_files = find_files(directory, video_patterns)
    audio_files = find_files(directory, audio_patterns)
    media_files = video_files + audio_files
    media_files = natsorted(media_files, key=lambda x: (x.count(os.sep), os.path.dirname(x), os.path.basename(x)))
    return media_files

def find_most_matching_prefix(path_list, path):
    matching_prefix = ""
    for prefix in path_list:
        if path.startswith(prefix) and len(prefix) > len(matching_prefix):
            matching_prefix = prefix
    return matching_prefix
  
def split_and_join_by_comma(long_string, max_length=250):
    # Split the string by commas
    parts = long_string.split(',')
    
    # Initialize the result list and a temporary buffer
    result = []
    temp_buffer = ""
    
    for part in parts:
        if len(temp_buffer) + len(part) + 1 <= max_length:
            if temp_buffer:
                temp_buffer += "," + part
            else:
                temp_buffer = part
        else:
            result.append(temp_buffer)
            temp_buffer = part
    
    # Append the remaining buffer if it's not empty
    if temp_buffer:
        result.append(temp_buffer)
    
    return result