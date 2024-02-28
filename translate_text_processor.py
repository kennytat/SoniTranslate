import re
from unicodedata import normalize
from nltk import tokenize as nltk_tokenize
import nltk
import os
from repl_dict import repl_dict_vi, clean_dict_html
from langdetect import detect_langs as detect_langs_prob

nltk.data.path.append("translate_tool/nltk")
sf = nltk.translate.bleu_score.SmoothingFunction()
# from docx import Document

def clean_html_entities(text):
  text = normalize('NFKC', text)
  text = text.strip()
  for regex, replacement in clean_dict_html.items():
    text = re.sub(regex, replacement, text, 0)
  return text

def split_sent(text):
  # list_clean_sent = list()
  list_raw_sent = list()
  para_2_list_raw_sent = list()
  list_para = text.split("\n")
  for para in list_para:
    para = para.strip()
    list_raw_sent_para = list()
    if len(para) > 0:
      list_raw_sent_para = nltk_tokenize.sent_tokenize(para)
    list_raw_sent.extend(list_raw_sent_para)
    para_2_list_raw_sent.append((para, len(list_raw_sent_para)))
  return list_raw_sent, para_2_list_raw_sent

def word_count(text):
  return len(text.strip().split(" "))

def join_sent(list_sent):
  return " eos ".join(list_sent)

def generate_gram(list_sent):
  tail_context = list()

  list_gram = list()
  for sent in list_sent:
    tail_context.append(sent)
    while (word_count(join_sent(tail_context)) > 250 and len(tail_context) > 0) or len(tail_context) > 5:
      tail_context.pop(0)
    list_gram.append(join_sent(tail_context))
  return list_gram

def post_process_text_vi(text):
  if text is None:
    return "[câu chờ được dịch]"
  text = text.strip()
  for regex, replacement in repl_dict_vi.items():
    text = re.sub(regex, replacement, text, 0)
  text = titlecase_with_dash(text)
  return text

def process_line(doc):
  text = clean_html_entities(doc)
  list_raw_sent, para_2_list_raw_sent = split_sent(text)
  return list_raw_sent, para_2_list_raw_sent


def read_file(file_dir):
  f = open(file_dir, encoding='utf-8', mode='r')
  content = f.read()
  f.close()
  return content

def sep_sent(text):
  return text.split(" eos ")

def count_sent(text):
  return len(sep_sent(text))

def is_vi(text):
  ret = False
  try:
    langs = detect_langs_prob(text)
    #     print(langs)
    if langs[0].lang == 'vi' and langs[0].prob >= 0.8:
      ret = True
  except Exception as e:
    #     print(e, text)
    pass
  return ret

def compute_bleu(s1, s2):
  t1 = s1.split(" ")
  t2 = s2.split(" ")
  return nltk.translate.bleu_score.sentence_bleu([t1], t2, smoothing_function=sf.method2)

def select_best(list_trans):
  i_2_scores = dict()
  for i in range(len(list_trans)):
    for j in range(len(list_trans)):
      if i not in i_2_scores:
        i_2_scores[i] = list()
      i_2_scores[i].append(compute_bleu(list_trans[i], list_trans[j]))
  i_2_score = dict()
  max_i = -1
  max_s = -1
  for i in i_2_scores:
    s = sum(i_2_scores[i]) / len(i_2_scores[i])
    #     print(list_trans[i])
    if is_vi(list_trans[i]):
      s += 1
    i_2_score[i] = s

    if s > max_s:
      max_s = s
      max_i = i
  return list_trans[max_i]

def not_translated(text):
  regex = r"\s*&#91; ?câu chưa dịch ?&#93;\s*"
  return re.match(pattern=regex, string=text, flags=re.MULTILINE | re.UNICODE | re.IGNORECASE)

def capitalize_para(line):
  toks = line.split(" ")
  if len(toks) == 0 or len(line) == 0:
    return line
  if not toks[0].isupper() and line[0].isalpha():
    toks[0] = toks[0].capitalize()
  return ' '.join(toks)

def titlecase_with_dash(string):
    def titlecase_word(match):
        return match.group(0).capitalize()
    return re.sub(r"(?=[\S]*['-])([A-Za-zÀÁÂÃÈÉÊÌÍÒÓÔÕÙÚĂĐĨŨƠàáâãèéêìíòóôõùúăđĩũơƯĂẠẢẤẦẨẪẬẮẰẲẴẶẸẺẼỀỀỂẾưăạảấầẩẫậắằẳẵặẹẻẽềềểếỄỆỈỊỌỎỐỒỔỖỘỚỜỞỠỢỤỦỨỪễệỉịọỏốồổỗộớờởỡợụủứừỬỮỰỲỴÝỶỸửữựỳỵỷỹ'-]+)", titlecase_word, string)
