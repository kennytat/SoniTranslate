import re
from vietnam_number import n2w
from vietTTS.BibleVerseParser import BibleVerseParser
from vietTTS.replace_dict import dictOfStrings

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