from pydub import AudioSegment
from pydub.utils import make_chunks
from tqdm import tqdm
import os

def split_by_odd_even(input_array):
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

def split_by_speaker(input_array):
    # Initialize empty arrays for odd and even elements
    speakers = list(set([segment['speaker'] for segment in input_array]))
    speaker_array = {}
    for speaker in speakers:
      speaker_array[speaker] = list(filter(lambda segment: segment['speaker'] == speaker, input_array))
    return speaker_array

def normalise_volume(audio, target_dBFS=-16, target_max_dBFS=-4.0):
    # Calculate current levels
    current_dBFS = audio.dBFS
    chunks = make_chunks(audio, 1)
    max_dBFS = max(chunk.dBFS for chunk in chunks)
    print("max_dBFS::", max_dBFS)   
    # Calculate the adjustment needed
    dB_change = target_dBFS - current_dBFS
    if dB_change > (target_max_dBFS - max_dBFS):
      dB_change = target_max_dBFS - max_dBFS 
    # Adjust volume
    adjusted_audio = audio + dB_change
    
    # # Process chunks to ensure max volume doesn't exceed original max_volume
    # adjusted_chunks = []
    # for chunk in make_chunks(adjusted_audio, 100):
    #     print(max_dBFS, chunk.dBFS)
    #     if chunk.dBFS > max_dBFS:
    #         # If chunk exceeds max_dBFS, reduce its volume
    #         db_change = max_dBFS - chunk.dBFS
            
    #         adjusted_chunks.append(chunk + db_change)
    #     else:
    #         adjusted_chunks.append(chunk)
    
    # # Combine adjusted chunks
    # final_audio = sum(adjusted_chunks)
    return adjusted_audio
   
def create_translated_audio(result_diarize, Output_name_file, match_start):
  
  if match_start:
    # Split even, odd audio files path and time segments
    split_speaker_array = split_by_speaker(result_diarize['segments'])
    if len(split_speaker_array.keys()) == 1:
      even_segments, odd_segments = split_by_odd_even(result_diarize['segments'])
      split_speaker_array['even'] = even_segments
      split_speaker_array['odd'] = odd_segments
    
    total_duration = result_diarize['segments'][-1]['end'] # in seconds
    print(round((total_duration / 60),2), 'minutes of video')
    combined_audio = AudioSegment.silent(duration=int(total_duration * 1000))

    for key, segments in split_speaker_array.items():
    # silent audio with total_duration
      segment_combined_audio = AudioSegment.silent(duration=int(total_duration * 1000))
      output_base, output_ext = os.path.splitext(Output_name_file)
      # file_array = even_audio_files if method == "even" else (odd_audio_files if method == "odd" else audio_files)
      output_path = f"{output_base}_{key}{output_ext}"
      # print("file_array::", method, len(file_array))
      for line in tqdm(segments):
        start = float(line['start'])
        audio_file = f"audio/{line['start']}.wav"
        # Overlay each audio at the corresponding time
        if os.path.isfile(audio_file):
          try:
            audio = AudioSegment.from_file(audio_file)
            ###audio_a = audio.speedup(playback_speed=1.5)
            start_time = start * 1000  # to ms
            segment_combined_audio = segment_combined_audio.overlay(audio, position=start_time)
          except:
            print(f'ERROR AUDIO FILE {audio_file}')
      # combined audio as a file
      segment_combined_audio = normalise_volume(audio=segment_combined_audio, target_dBFS=-16, target_max_dBFS=-4.0)
      segment_combined_audio.export(output_path, format="wav", bitrate="192k")
      combined_audio = combined_audio.overlay(segment_combined_audio, position=0)
      ## Export segment audio
    ## Export final audio
    combined_audio = normalise_volume(audio=combined_audio, target_dBFS=-16, target_max_dBFS=-4.0)
    combined_audio.export(Output_name_file, format="wav", bitrate="192k")
  else:
    concatenated_audio = AudioSegment.empty()
    for line in result_diarize['segments']:
      audio_file = f"audio/{line['start']}.wav"
      if os.path.isfile(audio_file):
        audio = AudioSegment.from_file(audio_file)
        concatenated_audio += audio
    # Export the concatenated audio to a file
    concatenated_audio.export(Output_name_file, format="wav")
  ## Normalise audio volume
  # name, ext = os.path.splitext(Output_name_file)
  # Output_name_tempfile=f"{name}_temp{ext}"
  # os.system(f'ffmpeg -i "{Output_name_file}" -af "loudnorm=I=-16:TP=-1.5:LRA=11:print_format=json" -ar 24000 "{Output_name_tempfile}" && mv "{Output_name_tempfile}" "{Output_name_file}"')
  # os.system(f'ffmpeg -i "{Output_name_file}" -filter:a "volume=-1.5dB" -ar 24000 "{Output_name_tempfile}" && mv "{Output_name_tempfile}" "{Output_name_file}"')
  os.system("rm -rf audio/*")
