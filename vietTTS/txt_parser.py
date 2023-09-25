from vietTTS.BibleVerseParser import BibleVerseParser
from vietTTS.replace_dict import dictOfStrings
from vietnam_number import n2w
import re
from pathlib import Path

VERSE_MAX_LENGTH = 500

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
