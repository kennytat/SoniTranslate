from gtts import gTTS
import re
import edge_tts
import asyncio
import nest_asyncio
from vietTTS.vietTTS import text_to_speech
from gradio_client import Client
import shutil


LANGUAGES_ABBR = {
    'Automatic detection': None,
    'Arabic': 'ar',
    'Cantonese': 'yue',
    'Chinese': 'zh',
    'Czech': 'cs',
    'Danish': 'da',
    'Dutch': 'nl',
    'English': 'en',
    'Finnish': 'fi',
    'French': 'fr',
    'German': 'de',
    'Greek': 'el',
    'Hebrew': 'he',
    'Hungarian': 'hu',
    'Indonesian': 'id',
    'Italian': 'it',
    'Japanese': 'ja',
    'Korean': 'ko',
    'Persian': 'fa',
    'Polish': 'pl',
    'Portuguese': 'pt',
    'Russian': 'ru',
    'Spanish': 'es',
    'Tagalog': 'tl',
    'Thai': 'th',
    'Turkish': 'tr',
    'Ukrainian': 'uk',
    'Urdu': 'ur',
    'Vietnamese': 'vi',
    'Hindi': 'hi',
}

def make_voice_gradio(tts_text, tts_voice, tts_speed, filename, source_language, target_language, t2s_method):
    print("make_voice_gradio::",tts_text, tts_voice, tts_speed, filename, source_language, target_language, t2s_method)
    if t2s_method == "VietTTS" and target_language == "vi" :
      try:
        text_to_speech(tts_text, filename, tts_voice, tts_speed)
      except Exception as error:
        print("tts error:", error, tts_text)
        tts = gTTS('a', lang=target_language)
        tts.save(filename)
    elif t2s_method == "M4T":
      client = Client("https://facebook-seamless-m4t.hf.space/--replicas/0jsdd/")
      try:
        result = client.predict(
            "T2ST (Text to Speech translation)",	# str (Option from: [('S2ST (Speech to Speech translation)', 'S2ST (Speech to Speech translation)'), ('S2TT (Speech to Text translation)', 'S2TT (Speech to Text translation)'), ('T2ST (Text to Speech translation)', 'T2ST (Text to Speech translation)'), ('T2TT (Text to Text translation)', 'T2TT (Text to Text translation)'), ('ASR (Automatic Speech Recognition)', 'ASR (Automatic Speech Recognition)')]) in 'Task' Dropdown component
            "file",	# str  in 'Audio source' Radio component
            "",	# str (filepath on your computer (or URL) of file) in 'Input speech' Audio component
            "",	# str (filepath on your computer (or URL) of file) in 'Input speech' Audio component
            tts_text,	# str  in 'Input text' Textbox component
            source_language,	# str (Option from: [('Afrikaans', 'Afrikaans'), ('Amharic', 'Amharic'), ('Armenian', 'Armenian'), ('Assamese', 'Assamese'), ('Basque', 'Basque'), ('Belarusian', 'Belarusian'), ('Bengali', 'Bengali'), ('Bosnian', 'Bosnian'), ('Bulgarian', 'Bulgarian'), ('Burmese', 'Burmese'), ('Cantonese', 'Cantonese'), ('Catalan', 'Catalan'), ('Cebuano', 'Cebuano'), ('Central Kurdish', 'Central Kurdish'), ('Croatian', 'Croatian'), ('Czech', 'Czech'), ('Danish', 'Danish'), ('Dutch', 'Dutch'), ('Egyptian Arabic', 'Egyptian Arabic'), ('English', 'English'), ('Estonian', 'Estonian'), ('Finnish', 'Finnish'), ('French', 'French'), ('Galician', 'Galician'), ('Ganda', 'Ganda'), ('Georgian', 'Georgian'), ('German', 'German'), ('Greek', 'Greek'), ('Gujarati', 'Gujarati'), ('Halh Mongolian', 'Halh Mongolian'), ('Hebrew', 'Hebrew'), ('Hindi', 'Hindi'), ('Hungarian', 'Hungarian'), ('Icelandic', 'Icelandic'), ('Igbo', 'Igbo'), ('Indonesian', 'Indonesian'), ('Irish', 'Irish'), ('Italian', 'Italian'), ('Japanese', 'Japanese'), ('Javanese', 'Javanese'), ('Kannada', 'Kannada'), ('Kazakh', 'Kazakh'), ('Khmer', 'Khmer'), ('Korean', 'Korean'), ('Kyrgyz', 'Kyrgyz'), ('Lao', 'Lao'), ('Lithuanian', 'Lithuanian'), ('Luo', 'Luo'), ('Macedonian', 'Macedonian'), ('Maithili', 'Maithili'), ('Malayalam', 'Malayalam'), ('Maltese', 'Maltese'), ('Mandarin Chinese', 'Mandarin Chinese'), ('Marathi', 'Marathi'), ('Meitei', 'Meitei'), ('Modern Standard Arabic', 'Modern Standard Arabic'), ('Moroccan Arabic', 'Moroccan Arabic'), ('Nepali', 'Nepali'), ('North Azerbaijani', 'North Azerbaijani'), ('Northern Uzbek', 'Northern Uzbek'), ('Norwegian Bokmål', 'Norwegian Bokmål'), ('Norwegian Nynorsk', 'Norwegian Nynorsk'), ('Nyanja', 'Nyanja'), ('Odia', 'Odia'), ('Polish', 'Polish'), ('Portuguese', 'Portuguese'), ('Punjabi', 'Punjabi'), ('Romanian', 'Romanian'), ('Russian', 'Russian'), ('Serbian', 'Serbian'), ('Shona', 'Shona'), ('Sindhi', 'Sindhi'), ('Slovak', 'Slovak'), ('Slovenian', 'Slovenian'), ('Somali', 'Somali'), ('Southern Pashto', 'Southern Pashto'), ('Spanish', 'Spanish'), ('Standard Latvian', 'Standard Latvian'), ('Standard Malay', 'Standard Malay'), ('Swahili', 'Swahili'), ('Swedish', 'Swedish'), ('Tagalog', 'Tagalog'), ('Tajik', 'Tajik'), ('Tamil', 'Tamil'), ('Telugu', 'Telugu'), ('Thai', 'Thai'), ('Turkish', 'Turkish'), ('Ukrainian', 'Ukrainian'), ('Urdu', 'Urdu'), ('Vietnamese', 'Vietnamese'), ('Welsh', 'Welsh'), ('West Central Oromo', 'West Central Oromo'), ('Western Persian', 'Western Persian'), ('Yoruba', 'Yoruba'), ('Zulu', 'Zulu')]) in 'Source language' Dropdown component
            target_language,	# str (Option from: [('Bengali', 'Bengali'), ('Catalan', 'Catalan'), ('Czech', 'Czech'), ('Danish', 'Danish'), ('Dutch', 'Dutch'), ('English', 'English'), ('Estonian', 'Estonian'), ('Finnish', 'Finnish'), ('French', 'French'), ('German', 'German'), ('Hindi', 'Hindi'), ('Indonesian', 'Indonesian'), ('Italian', 'Italian'), ('Japanese', 'Japanese'), ('Korean', 'Korean'), ('Maltese', 'Maltese'), ('Mandarin Chinese', 'Mandarin Chinese'), ('Modern Standard Arabic', 'Modern Standard Arabic'), ('Northern Uzbek', 'Northern Uzbek'), ('Polish', 'Polish'), ('Portuguese', 'Portuguese'), ('Romanian', 'Romanian'), ('Russian', 'Russian'), ('Slovak', 'Slovak'), ('Spanish', 'Spanish'), ('Swahili', 'Swahili'), ('Swedish', 'Swedish'), ('Tagalog', 'Tagalog'), ('Telugu', 'Telugu'), ('Thai', 'Thai'), ('Turkish', 'Turkish'), ('Ukrainian', 'Ukrainian'), ('Urdu', 'Urdu'), ('Vietnamese', 'Vietnamese'), ('Welsh', 'Welsh'), ('Western Persian', 'Western Persian')]) in 'Target language' Dropdown component
            api_name="/run"
        )
        if result and result[0]:
          shutil.move(result[0], filename)
      except Exception as error:
        print("tts error:", error, tts_text)
        tts = gTTS(tts_text, lang=LANGUAGES_ABBR[target_language])
        tts.save(filename)
    elif t2s_method == 'Google':
      try:
        tts = gTTS(tts_text, lang=LANGUAGES_ABBR[target_language])
        tts.save(filename)
        print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
      except:
        tts = gTTS('a', lang=LANGUAGES_ABBR[target_language])
        tts.save(filename)
        print('Error: Audio will be replaced.')   
    else:
      try:
        asyncio.run(edge_tts.Communicate(tts_text, "-".join(tts_voice.split('-')[:-1])).save(filename))
      except:
        try:
          tts = gTTS(tts_text, lang=LANGUAGES_ABBR[target_language])
          tts.save(filename)
          print(f'No audio was received. Please change the tts voice for {tts_voice}. TTS auxiliary will be used in the segment')
        except:
          tts = gTTS('a', lang=LANGUAGES_ABBR[target_language])
          tts.save(filename)
          print('Error: Audio will be replaced.')
 
