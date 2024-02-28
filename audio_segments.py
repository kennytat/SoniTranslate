from pydub import AudioSegment
from tqdm import tqdm
import os

def split_array_odd_even(input_array):
    # Initialize empty arrays for odd and even elements
    even_array = []
    odd_array = []
    # Use a for loop to iterate through the input array
    for index, element in enumerate(input_array):
        if index % 2 == 0:
            # If the index is even, add the element to the even array
            even_array.append(element)
        else:
            # If the index is odd, add the element to the odd array
            odd_array.append(element)
    return even_array, odd_array

def create_translated_audio(result_diarize, audio_files, Output_name_file, match_start):
  if match_start:
    # Split even, odd audio files path and time segments
    even_audio_files, odd_audio_files = split_array_odd_even(audio_files)
    even_segments, odd_segments = split_array_odd_even(result_diarize['segments'])
    
    total_duration = result_diarize['segments'][-1]['end'] # in seconds
    print(round((total_duration / 60),2), 'minutes of video')

    for method in ["even", "odd", "full"]:
    # silent audio with total_duration
      combined_audio = AudioSegment.silent(duration=int(total_duration * 1000))
      output_base, output_ext = os.path.splitext(Output_name_file)
      file_array = even_audio_files if method == "even" else (odd_audio_files if method == "odd" else audio_files)
      segments = even_segments if method == "even" else (odd_segments if method == "odd" else result_diarize['segments'])
      output_path = f"{output_base}_even{output_ext}" if method == "even" else (f"{output_base}_odd{output_ext}" if method == "odd" else Output_name_file)
      print("file_array::", method, file_array)
      for line, audio_file in tqdm(zip(segments, file_array)):
        start = float(line['start'])
        # Overlay each audio at the corresponding time
        try:
          audio = AudioSegment.from_file(audio_file)
          ###audio_a = audio.speedup(playback_speed=1.5)
          start_time = start * 1000  # to ms
          combined_audio = combined_audio.overlay(audio, position=start_time)
        except:
          print(f'ERROR AUDIO FILE {audio_file}')
      # combined audio as a file
      combined_audio.export(output_path, format="wav", bitrate="16k") # best than ogg, change if the audio is anomalous
 
  else:
    concatenated_audio = AudioSegment.empty()
    for audio_file in audio_files:
        audio = AudioSegment.from_file(audio_file)
        concatenated_audio += audio
    # Export the concatenated audio to a file
    concatenated_audio.export(Output_name_file, format="wav")
  os.system("rm -rf audio/*")
