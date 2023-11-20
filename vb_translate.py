from dotenv import load_dotenv
import tempfile
import os
import shutil
import json
import translate_text_processor
load_dotenv()

TRANS_TOOL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "translate_tool")
MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "model")
TXT_TRANS_TEMP = os.path.join(tempfile.gettempdir(), "vgm-translate", "translate_tmp")
BPEROOT = os.path.join(TRANS_TOOL_DIR ,"subword-nmt")

def vb_translate(raw_input):
  # print("raw_input::", raw_input)
  file_name = 'tmp_input'
  content = raw_input
  # print("source_lang_content::", content)
  list_raw_sent, para_2_list_raw_sent = translate_text_processor.process_line(content)
  # print("postcontent::", list_raw_sent, para_2_list_raw_sent)

  tmp_dir = os.path.join(TXT_TRANS_TEMP, file_name)
  # print("tmp_dir::", tmp_dir)
  
  if os.path.exists(tmp_dir) and os.path.isdir(tmp_dir):
    shutil.rmtree(tmp_dir)
  os.makedirs(tmp_dir, exist_ok=False)
  
  g = open(os.path.join(tmp_dir,"para_2_raw_count.json"), 'w', encoding='utf-8')
  json.dump(para_2_list_raw_sent, g, indent=3)
  g.close()

  g = open(os.path.join(tmp_dir, "text.en"), 'w', encoding="utf-8")
  g.write(str("\n".join(list_raw_sent)))
  g.close()

  file = os.path.join(tmp_dir, "text.en")
  new_file = os.path.join(tmp_dir,  "tok.en")
  lower_file = os.path.join(tmp_dir, "lower.en")

  bpe_file =os.path.join(tmp_dir, "bpe.en")
  gram_file = os.path.join(tmp_dir, "gram.en")

  trans_file = os.path.join(tmp_dir, "en_vi.txt")

  os.system(f"cat {file} \
  | perl {TRANS_TOOL_DIR}/mosesdecoder/scripts/tokenizer/remove-non-printing-char.perl \
  | perl {TRANS_TOOL_DIR}/mosesdecoder/scripts/tokenizer/tokenizer.perl -threads 8 -l en > {new_file}")

  os.system(f"perl {TRANS_TOOL_DIR}/mosesdecoder/scripts/tokenizer/lowercase.perl <{new_file}> {lower_file}")

  os.system(f"python {BPEROOT}/apply_bpe.py --input {lower_file} --output {bpe_file} -c {TRANS_TOOL_DIR}/bpe_code/code.en --num-workers 8")

  f = open(bpe_file, 'r', encoding='utf-8')
  list_sent = [l.strip() for l in f]
  f.close()
  gram = translate_text_processor.generate_gram(list_sent)

  assert len(list_sent) == len(gram)
  g = open(gram_file, 'w', encoding='utf-8')
  g.write("\n".join(gram))
  g.close()

  testpref = os.path.join( tmp_dir, "gram")
  destdir = os.path.join(tmp_dir, "en.pre")

  os.system(
    f"export MKL_SERVICE_FORCE_INTEL=1 && fairseq-preprocess \
        --only-source \
        --source-lang en --target-lang vi \
        --srcdict {TRANS_TOOL_DIR}/dict/dict.en.txt \
        --tgtdict {TRANS_TOOL_DIR}/dict/dict.vi.txt \
        --testpref {testpref} \
        --destdir {destdir} \
        --workers 8"
  )

  os.system(f"cp {TRANS_TOOL_DIR}/dict/dict.vi.txt {destdir}")

  os.system(
    f"fairseq-generate {destdir} \
    --path {os.path.join(MODEL_DIR,'vb_model','checkpoint_last.pt')} \
    --max-len-b 300 \
    --batch-size 80 --beam 4 --remove-bpe --fp16 --empty-cache-freq 10 > {trans_file}"
  )

  f = open(trans_file, 'r', encoding='utf-8')
  lines = f.readlines()
  f.close()

  en_idx_2_text = dict()
  vi_idx_2_text = dict()

  for line in lines:
    line = line.strip()

    if line.startswith("S-"):
      num = int(line.split("\t")[0].split("-")[-1])
      text = line.split("\t")[-1].strip()
      if num in en_idx_2_text:
        print("warning: id appear more than 1 time: ", line)
      en_idx_2_text[num] = text

    if line.startswith("D-"):
      num = int(line.split("\t")[0].split("-")[-1])
      text = line.split("\t")[-1].strip()
      if num in vi_idx_2_text:
        print("warning: id appear more than 1 time: ", line)
      vi_idx_2_text[num] = text

  idx_2_trans = dict()
  for idx in vi_idx_2_text:
    vi_sent = vi_idx_2_text[idx]
    en_sent = en_idx_2_text[idx]

    vi_segs = translate_text_processor.sep_sent(vi_sent)
    en_segs = translate_text_processor.sep_sent(en_sent)

    if len(vi_segs) != len(en_segs):
      continue

    for i in range(len(vi_segs)):
      pos = idx + i

      if pos not in idx_2_trans:
        idx_2_trans[pos] = list()
      idx_2_trans[pos].append(vi_segs[i])

  idx_2_best = dict()

  for idx in idx_2_trans:
    list_trans = idx_2_trans[idx]
    idx_2_best[idx] = translate_text_processor.select_best(list_trans)

  list_vi_trans = list()

  for i in range(len(list_sent) + 4):
    if i in idx_2_best:
      list_vi_trans.append(idx_2_best[i])
    else:
      list_vi_trans.append(" &#91; câu chưa dịch &#93; ")
      print("TRANSLATION FAILED")

  g = open(os.path.join(tmp_dir, 'vi_text.txt'), 'w', encoding='utf-8')
  g.write("\n".join(list_vi_trans))
  g.close()

  os.system(
    f"python {BPEROOT}/apply_bpe.py -c {TRANS_TOOL_DIR}/bpe_code_lc/code \
    -i {tmp_dir}/vi_text.txt -o {tmp_dir}/lc.clean.bpe.en --num-workers 8"
  )

  os.system(
    f"export MKL_SERVICE_FORCE_INTEL=1 && fairseq-preprocess \
        --only-source \
        --source-lang en --target-lang vi \
        --srcdict {TRANS_TOOL_DIR}/dict_lc/dict.en.txt \
        --tgtdict {TRANS_TOOL_DIR}/dict_lc/dict.vi.txt \
        --testpref {tmp_dir}/lc.clean.bpe \
        --destdir {tmp_dir}/lc.en.pre \
        --workers 8"
  )

  os.system(f"cp {TRANS_TOOL_DIR}/dict_lc/dict.vi.txt {tmp_dir}/lc.en.pre")

  os.system(
    f"fairseq-generate {tmp_dir}/lc.en.pre \
    --path {os.path.join(MODEL_DIR,'vb_model_lc','checkpoint_last.pt')} \
    --batch-size 100 --beam 4 --remove-bpe > {tmp_dir}/vi_lc.txt"
  )

  f = open(os.path.join(tmp_dir, 'vi_lc.txt'), encoding='utf-8')
  lines = f.readlines()
  f.close()

  en_idx_2_text = dict()
  vi_idx_2_text = dict()

  for line in lines:
    line = line.strip()

    if line.startswith("S-"):
      num = int(line.split("\t")[0].split("-")[-1])
      text = line.split("\t")[-1].strip()
      if num in en_idx_2_text:
        print("warning: id appear more than 1 time: ", line)
      en_idx_2_text[num] = text

    if line.startswith("D-"):
      num = int(line.split("\t")[0].split("-")[-1])
      text = line.split("\t")[-1].strip()
      if num in vi_idx_2_text:
        print("warning: id appear more than 1 time: ", line)
      vi_idx_2_text[num] = text

  f = open(os.path.join(tmp_dir, 'vi_text.txt') , 'r', encoding='utf-8')
  list_vi_trans = [l.strip() for l in f]
  f.close()

  list_vi_up = list()
  for i in range(len(list_vi_trans)):
    list_vi_up.append(vi_idx_2_text[i])

  g = open(os.path.join(tmp_dir, 'vi_trans.txt'), 'w', encoding='utf-8')
  g.write('\n'.join(list_vi_up))
  g.close()

  f = open(os.path.join(tmp_dir,'vi_trans.txt'), encoding='utf-8')
  vi_lines = [l.strip() for l in f]
  f.close()

  f = open(os.path.join(tmp_dir, 'text.en'), encoding='utf-8')
  raw_en_lines = [l.strip() for l in f]
  f.close()

  en_para_json = json.load(open(os.path.join(tmp_dir, "para_2_raw_count.json"), encoding='utf-8'))

  vi_pos = 4
  en_para_list = list()
  vi_para_list = list()

  for en_para, vi_count in en_para_json:
    vi_para = list()
    for i in range(vi_count):
      if vi_pos < len(vi_lines):
        tmp = vi_lines[vi_pos]
        lang = 'vi'

        if translate_text_processor.not_translated(tmp):
          lang = 'en'
          tmp = raw_en_lines[vi_pos - 4]
        else:
          try:
            langs = translate_text_processor.detect_langs_prob(tmp)
            if langs[0].lang == 'en' and langs[0].prob >= 0.98:
              lang = 'en'
              tmp = raw_en_lines[vi_pos - 4]
          except Exception as e:
            lang = 'vi'
            tmp = vi_lines[vi_pos]

        if lang == 'vi':
          tmp = translate_text_processor.post_process_text_vi(tmp)

          if raw_en_lines[vi_pos - 4].upper() == raw_en_lines[vi_pos - 4]:
            tmp = tmp.upper()

      elif vi_pos >= len(vi_lines):
        tmp = raw_en_lines[vi_pos - 4]

      vi_para.append(tmp)
      vi_pos += 1
    vi_text = " ".join(vi_para)
    en_para_list.append(en_para)
    vi_para_list.append(translate_text_processor.capitalize_para(vi_text))
    
  print("en_para_list::",len(en_para_list), en_para_list)
  print("vi_para_list::",len(vi_para_list),vi_para_list)
  return vi_para_list


if __name__ == '__main__':
  # raw_str = """At the first God made the heaven and the earth.
  #           And the earth was waste and without form; and it was dark on the face of the deep: and the Spirit of God was moving on the face of the waters.
  #           And God said, Let there be light: and there was light.
  #           And God, looking on the light, saw that it was good: and God made a division between the light and the dark,
  #           Naming the light, Day, and the dark, Night. And there was evening and there was morning, the first day.
  #           And God said, Let there be a solid arch stretching over the waters, parting the waters from the waters.
  #           And God made the arch for a division between the waters which were under the arch and those which were over it: and it was so.
  #           And God gave the arch the name of Heaven. And there was evening and there was morning, the second day.
  #           And God said, Let the waters under the heaven come together in one place, and let the dry land be seen: and it was so.
  #           And God gave the dry land the name of Earth; and the waters together in their place were named Seas: and God saw that it was good.
  #           And God said, Let grass come up on the earth, and plants producing seed, and fruit-trees giving fruit, in which is their seed, after their sort: and it was so."""
  
  with open('/home/vgm/Desktop/test/test.txt', 'r') as f:
    result = vb_translate(f.read())
  print("result::", result)

