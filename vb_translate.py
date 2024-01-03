from dotenv import load_dotenv
import tempfile
import os
import shutil
import json
import translate_text_processor
from langdetect import detect
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
  from translate_segments import translate_text
  for index, sentence in enumerate(vi_para_list):
    try:
      if index == 0:
        vi_para_list[index] = translate_text(en_para_list[index], "vi", "T5")
      else:
        lang = detect(sentence)
        if lang != "vi":
          print('non translated sentence, try again::', sentence)
          vi_para_list[index] = translate_text(sentence, "vi", "T5")
    except Exception as e:
      print('detect sentence language failed: skip translating sentence:', e, index, sentence)
      # print(index, sentence)
  return vi_para_list


if __name__ == '__main__':
  raw_str = """At the first God made the heaven and the earth.
            And the earth was waste and without form; and it was dark on the face of the deep: and the Spirit of God was moving on the face of the waters.
            And God said, Let there be light: and there was light.
            And God, looking on the light, saw that it was good: and God made a division between the light and the dark,
            Naming the light, Day, and the dark, Night. And there was evening and there was morning, the first day.
            And God said, Let there be a solid arch stretching over the waters, parting the waters from the waters.
            And God made the arch for a division between the waters which were under the arch and those which were over it: and it was so.
            And God gave the arch the name of Heaven. And there was evening and there was morning, the second day.
            And God said, Let the waters under the heaven come together in one place, and let the dry land be seen: and it was so.
            And God gave the dry land the name of Earth; and the waters together in their place were named Seas: and God saw that it was good.
            And God said, Let grass come up on the earth, and plants producing seed, and fruit-trees giving fruit, in which is their seed, after their sort: and it was so.
            the voice identified itself as the alpha and omega, the first and the last, and then charged him to write what he saw to seven churches in asia 
            turning to see the voice, john saw seven golden lampstands and in their midst the son of man
            describing the awesome appearance of the son of man and his own reaction, john then records how jesus comforted and then charged him to write what he has seen and will see 
            the church at smyrna is commended for being rich despite their tribulation and poverty
            unlike most churches, there are no words of condemnation directed toward it
            ״ BOOKENDS
            th! CHRISTIAN LIFE
            JERRY BRIDGES
            ROR RFVTMGTOM
            Through his many books, Jerry Bridges has been shepherding my soul since I first became a Christian sixteen years ago. He has done it again. As I have come to expect, he has provided a sea of theological matter in a drop of devotional language. Here you will find God-centered doctrine that is delectably deep and down to earth at the same time. I promise that if you read this book carefully and prayerfully, you will gain both an informed mind and an enlarged heart.
            - TULLIAN TCHIVIDJIAN, Pastor, New City Church, Fort Lauderdale, Florida; author, Unfashionable:
            Making a Difference in this World by Being Different
            Jerry Bridges and Bob Bevington have provided another marvelous instrument for guiding believers to the provisions of a gospel-blessed and gospel-driven life. In this volume the surpassing righteousness of Christ joined with the power of the Holy Spirit is clearly displayed from God's Word. Thanks from all of us who desire to make disciples that will follow Christ purposefully and passionately.
            - HARRY L. REEDER III, Senior Pastor,
            Briarwood Presbyterian Church, Birmingham, Alabama
            Forgiveness of sin and power to change - I can think of no more essential topics for a Christian to study than these twin blessings of the gospel. And I can think of no one better to write on these topics than Jerry Bridges. Jerry has provided for me over the years a constant diet of gospel-saturated writing, and here is a fresh feast. I trust you will enjoy it as much as I have.
            - C. J. MAHANEY, Sovereign Grace Ministries
            Jerry Bridges and Bob Bevington look at the Christian life through a wide-angle lens, examining the framework that supports, stabilizes and secures the believer's life in Christ. They teach elements of a distinctly biblical worldview, leaning upon the righteousness of Christ on one hand and upon the power of the Holy Spirit on the other. A wise and powerful book, one I heartily recommend.
            - TIM CHALLIES, author, The Discipline of Spiritual Discernment
            Thinking you understand the gospel but applying it only to salvation is like barely releasing your sail and slogging through the waves. Bookends will equip you to release that sail, catch the mighty wind of God, and see every 'book' in your life transformed.
            - DEE BRESTIN, author, Falling in Love with Jesus
            With their latest publication, the authors display how sound theology is transformational and how understanding enhances true piety and produces profound worship.
            - ROBERT M. NORRIS, Pastor, Fourth Presbyterian Church, Bethesda, Maryland
            Martin Luther said, 'Man in his search for truth is like a drunken peasant. You help him up on one side of his horse and he falls over the other side.' This book, perhaps more than any other, is designed to keep you on the horse living in the truth of the gospel. There are few books you will find more valuable on your shelf.
            - JOE COFFEY, Lead Pastor, Hudson Community Chapel, Hudson, Ohio
            THE BOOKENDS OF THE
            CHRISTIAN LIFE
            OTHER CROSSWAY BOOKS BY
            Jerry Bridges and Bob Bevington
            The Great Exchange:
            My Sin for His Righteousness (2007)
            THE
            BOOKENDS
            OF THE
            CHRISTIAN LIFE
            JERRY BRIDGES
            & BOB BEVINGTON
            CROSS WAY BOOKS
            WHEATON ILLINOIS
            The Bookends of the Christian Life
            Copyright © 2009 by Jerry Bridges and Bob Bevington
            Published by Crossway Books
            a publishing ministry of Good News Publishers
            1300 Crescent Street
            Wheaton, Illinois 60187
            All rights reserved. No part of this publication may be reproduced, stored in a retrieval system, or transmitted in any form by any means, electronic, mechanical, photocopy, recording, or otherwise, without the prior permission of the publisher, except as provided for by USA copyright law.
            Cover design: The DesignWorks Group, www.thedesignworksgroup.com
            First printing 2009
            Printed in the United States of America
            Unless otherwise indicated, Scripture quotations are from the ESV® Bible (The Holy
            Bible, English Standard Version®), © 2001 by Crossway Bibles, a publishing ministry of
            Good News Publishers. Used by permission. All rights reserved.
            Scripture quotations marked KJV are from the King James Version of the Bible.
            Scripture quotations marked NASB are from The New American Standard Bible, © The Lockman Foundation 1960, 1962, 1963, 1968, 1971, 1972, 1973, 1975, 1977, 1995.
            Used by permission.
            All emphases in Scripture quotations have been added by the authors.
            PDF ISBN: 978-1-4335-0551-5
            Mobipocket ISBN: 978-1-4335-0552-2
            Library of Congress Cataloging-in-Publication Data
            Bridges, Jerry.
            The bookends of the Christian life / Jerry Bridges and Bob Bevington.
            p. cm.
            ISBN 978-1-4335-0319-1 (hc)
            1. Justification (Christian theology). 2. Sanctification -
            Christianity. I. Bevington, Bob, 1956- . II. Title
            BT764.3.B75 2009
            234'.7 - dc22
            LB 18 17 16 15 14 13 12 11 10 09
            To all who, like the two of us,
            recognize the utter insufficiency of their own
            righteousness and strength,
            and thus are desperate for the gospel.
            And to our Triune God - Father, Son, and Holy Spirit -
            who provides us with an impeccable righteousness
            and an indomitable strength
            through our union with Christ.
            Only in the Lord . . . are righteousness and strength.
            ISAIAH 45:24
            FOR INDIVIDUAL OR SMALL GROUP STUDY
            We encourage you to visit:
            www.TheBookendsBook.com where you'll find a free downloadable study guide and other tools to help you get the most out of this book.
            Contents
            Preface
            Introduction: Books and Bookends
            PART
            The First Bookend:
            The Righteousness of Christ
            The Righteousness of Christ
            The Motivation of the Gospel
            Gospel Enemy #1: Self-righteousness
            Gospel Enemy #2: Persistent Guilt
            Leaning on the First Bookend
            PART
            The Second Bookend:
            The Power of the Holy Spirit
            The Power of the Holy Spirit
            Dependent Responsibility
            The Help of the Divine Encourager
            Gospel Enemy #3: Self-reliance
            Leaning on the Second Bookend
            Conclusion: The Bookends Personal Worldview
            Notes
            PREFACE
            Over the past several years, as the two of us have shared with each other what God is teaching us through his Word and our experiences, we've concluded there are two foundational truths that give stability to our Christian lives. We've chosen to use the illustration of bookends to teach these two truths.
            The Bookends of the Christian Life is a collaborative effort. So in every instance, whether the teaching or illustration is from one or both of us, we've chosen to use the plural pronouns we or us
            We would like to acknowledge Greg Plitt, Chris Thifault, Steve Myers, and Joe Coffey for their valuable assistance with the early drafts and Greg Bryan for the diagram design. Thanks also to Allan Fisher, senior vice president for book publishing at Crossway, for his support of this project, and to Lydia Brownback and Thomas Womack for their outstanding editorial work. In addition, we're grateful for each and every member of the Crossway team.
            Lastly, we would like to thank Mitch Gingrich for his excellence in providing the free study guide at www. TheBookendsBook.com.
            Jerry Bridges
            Bob Bevington
            INTRODUCTION
            BOOKS AND BOOKENDS
            Most of us have experienced the difficulty of putting books on a bookshelf without having a set of bookends to keep them in place. You know what happens. The books on the end tip over. Then the books next to those tumble over the ones already fallen. Inevitably some end up on the floor. At this point we do what we should have done in the first place. We set a couple of sturdy bookends in position to support and stabilize the books on the shelf.
            Think of your life right now as a long bookshelf. The books on it represent all the things you do - both spiritual and temporal. There's a spiritual book for each activity of your Christian growth and service, perhaps with titles such as Church Attendance, Bible Study, Daily Quiet Time, Sharing th e Gospel, or Serving Others. The temporal books might include Job Performance, Educational Pursuits, Recreation and Leisure, Grocery Shopping, Driving the Car, Doing the Laundry, Mowing the Grass, and Paying the Bills, to name a few. Our temporal books are intermingled with spiritual books on our bookshelf, since all our activities are to be informed and directed by the spiritual dimension, just as Paul indicated: Whatever you do, do all to the glory of God (1 Corinthians 10:31).
            This bookshelf of your life is a very active place. In the course of each day, as you pull one book after another off the shelf, life can get complicated. And the more committed and conscientious you are, the more frustration you might feel trying to manage all your various books simultaneously.
            Without adequate bookends, even if we succeed in getting all our books to remain upright, their stability is precarious at best. If we try removing even one book, we may jostle those next to it, disturb the delicate balance, and cause books to topple and fall. Sometimes a single tilted book can knock over every other volume on the shelf in a catastrophic domino effect. You can see why two sturdy, reliable bookends can make all the difference.
            On top of life's complexity with its demands in both the spiritual and temporal realms, we often add a sense of guilt -  guilt for what we should do but don't and guilt for what we do but shouldn't. Regrettably, many Christians struggle with one or more persistent sin patterns, often called besetting sins. They produce a sense of deep, gnawing, demoralizing guilt, which tends to hinder us from pursuing godly change. In fact, contrary to popular thinking, guilt by itself rarely, if ever, motivates a person to change. By itself, it only discourages us.
            However, when rightly handled, guilt is actually good for us. It's like pain. Pain tells us something's wrong and alerts us to do something to address its root cause. Consider leprosy, a disease that causes the loss of the sensation of pain in the hands and feet so that its victims frequently injure themselves without realizing it. In a similar way, a person without a sense of guilt can continue on a destructive path of sin without being aware of it. Such was generally the case of the self-righteous Pharisees, the ones Jesus opposed so vehemently.
            Truth be told, there's probably something of the Pharisee in all of us, but in some, numbness to guilt is the prevalent condition of their heart. And their resulting sense of self-righteousness is far more dangerous than a sense of guilt.
            On the other hand, the guilt-laden person is painfully aware of his situation. He struggles with his persistent sins, but sooner or later he fails again. He just doesn't know what to do. He has been told to try harder, but that hasn't worked either. So he continues a life of quiet desperation.
            Both the self-righteous Pharisee in his smugness and the guilt-laden person in his desperation have one thing in common: their bookshelf of life has no bookends.
            The solution for both is the same. When we become united to Christ by faith, God places a set of bookends on the bookshelf of our lives. One bookend is the righteousness of Christ; the other is the power of the Holy Spirit. Though they're provided by God, it's our responsibility to lean our books on them, relying on them to support, stabilize, and secure all our books - everything we do.
            Why are these two gracious provisions from God the bookends of the Christian life? And how do we lean our books on them? This book will answer those questions, and these:
            « How can I overcome persistent guilt?
            « How can I deal with the pressure to measure up?
            « Where can I find the motivation to grow?
            « How can I live the Christian life with my heart and not just my head?
            « How can I be sure God loves and accepts me?
            o Where do I draw the line between God's grace and my works?
            o Where can I find the strength to change in an authentic and lasting way?
            The answers start with the first bookend. So continue with us as we explore the meaning and application of the righteousness of Christ.
            PART
            The First Bookend:
            The
            of Christ
            CHAPTER ONE
            THE RIGHTEOUSNESS
            OF CHRIST
            I am not ashamed of the gospel . . . for in it the righteousness
            of God is revealed.
            ROMANS 1:16-17
            What is the righteousness of Christ, and why do we need it as the first bookend? The word righteous in the Bible basically means perfect obedience; a righteous person is one who always does what is right. This statement assumes there's an external, objective standard of right and wrong. That standard is the universal moral will of God as given to us throughout the Bible. It's the law of God written on every human heart. It's the standard by which each person will ultimately be judged.
            Our problem is that we're not righteous. As the apostle Paul put it so bluntly, None is righteous, no, not one
            No one
            does good, not even one (Romans 3:10, 12). That's strong language. We may quickly protest that we're not so bad. After all, we don't steal, murder, or engage in sexual immorality. We usually obey our civil laws and treat each other decently. So how can Paul say we're not righteous?
            We respond this way because we fail to realize how impossibly high God's standard actually is. When asked, Which is the great commandment in the Law? Jesus responded, You shall love the Lord your God with all your heart and with all your soul and with all your mind. This is the great and first commandment. And a second is like it: You shall love your neighbor as yourself. On these two commandments depend all the Law and the Prophets (Matthew 22:36-40). None of us has even come close to fulfilling either of these two commandments. Yet Paul wrote, For all who rely on works of the law are under a curse; for it is written, 'Cursed be everyone who does not abide by all things written in the Book of the Law, and do them' (Galatians 3:10). All is absolute. It means exactly what it says; not most, but all.
            If we applied this same standard in the academic world, scoring 99 percent on a final exam would mean failing the course. A term paper with a single misspelled word would earn an F. No school has a standard of grading this rigorous; if it did, no one would graduate. In fact, professors often grade on a curve, meaning all grades are relative to the best score in the class, even if that score isn't perfect. We're so accustomed to this approach we tend to think God also grades on a curve. We look at the scandalous sins of society around us, and because we don't engage in them, we assume God is pleased with us. After all, we're better than they are.
            But God doesn't grade on a curve. The effect of Galatians 3:10 is to put us all under God's curse. And while it's one thing to fail a course at the university, it's altogether something else to be eternally damned under the curse of God. The good news of the gospel, of course, is that those who have trusted in Jesus Christ as their Savior will not experience that curse. As Paul wrote just a few sentences later, Christ redeemed us from the curse of the law by becoming a curse for us (Galatians 3:13). Let this truth sink deeply into your heart and mind: apart from the saving work of Christ, every one of us still deserves God's curse every day of our lives.
            We may not commit scandalous sins. But what about our pride, our selfishness, our impatience with others, our critical spirit, and all sorts of other sins we tolerate on a daily basis? Even on our best days, we still haven't loved God or our neighbor as we should.
            So we have to agree with Paul. None of us is righteous, not even one.
            We know we need a Savior, so we trust in Christ to redeem us from the curse of God's law. But though we believe we're saved as far as our eternal destiny is concerned, we may not be sure about our day-to-day standing with God. Many of us embrace a vague but very real notion that God's approval has to be earned by our conduct. We know we're saved by grace, but we believe God blesses us according to our level of personal obedience. Consequently, our confidence that we abide in God's favor ebbs and flows according to how we gauge our performance. And since we each sin every single day, this approach is ultimately discouraging and even devastating. This is exactly why we need the first bookend. The righteousness of Christ changes all this.
            JESUS CHRIST THE
            RIGHTEOUS ONE
            What exactly is the righteousness of Christ? And how will it give us a sense of assurance in our day-to-day relationship with God? To begin answering those questions, let's go to one of our favorite verses of Scripture:
            For our sake he made him to be sin who knew no sin, so that in him we might become the righteousness of God. (2 Corinthians 5:21)
            The first thing we need to consider in this verse is the sinlessness - the perfect obedience - of Jesus as a man living among us for thirty-three years. The Scriptures consistently testify to this. All four of the major writers of the New Testament letters attest to the sinless, perfect obedience of Jesus throughout his life on earth. In addition to Paul's words that Jesus knew no sin, we have the testimony of Peter, John, and the writer of Hebrews: He committed no sin (1 Peter 2:22); In him there is no sin (1 John 3:5); Jesus was in every respect tempted as we are, yet without sin (Hebrews 4:15).
            One of the most powerful indications of the sinlessness of Jesus came from his own mouth. To a group of hostile Jews to whom he'd just said, You are of your father the devil, Jesus dared to ask the question, Which one of you convicts me of sin? (John 8:44-46). He could ask this question because he knew the answer - he was sinless. Jesus could confidently say of the Father, I always do the things that are pleasing to him (John 8:29). Every moment of his life, from birth to death, Jesus perfectly obeyed the law of God, the same law that is applicable to all of us.
            Christ's obedience was tested by temptation (Matthew 4:1­11; Hebrews 4:15), and the intensity of his temptation was greater than any we'll ever experience or even imagine. When we succumb to temptation, the pressure is relieved for awhile; but unlike us, Jesus never gave in.
            As astounding as that is, it wasn't the epitome of Christ's obedience. The pinnacle of his obedience came when he humbled himself by becoming obedient to the point of death, even death on a cross (Philippians 2:8). The obedient death of Christ is the very apex of the righteousness of Christ.
            Let's not miss the implications of this. At the cross, Jesus paid the penalty we should have paid, by enduring the wrath of God we should have endured. And this required him to do something unprecedented. It required him to provide the ultimate level of obedience - one that we'll never be asked to emulate. It required him to give up his relationship with the Father so that we could have one instead. The very thought of being torn away from the Father caused him to sweat great drops of blood (Luke 22:44). And at the crescendo of his obedience, he screamed, My God, my God, why have you forsaken me? (Mark 15:34). The physical pain he endured was nothing compared to the agony of being separated from the Father. In all of history, Jesus is the only human being who was truly righteous in every way; and he was righteous in ways that are truly beyond our comprehension.
            OUR SIN TRANSFERRED TO CHRIST
            The second truth to note in 2 Corinthians 5:21 is that for our sake he made him to be sin. This is Paul's way of saying God caused Jesus to bear our sin. Peter wrote something similar: He himself bore our sins in his body on the tree (1 Peter 2:24). So did the prophet Isaiah: All we like sheep have gone astray; we have turned - every one - to his own way; and the Lord has laid on him the iniquity of us all (Isaiah 53:6). Paul is telling us that God the Father took our sin and charged it to God the Son in such a way that Christ was made to be sin for our sake.
            Now we can see what Paul meant in Galatians 3:13 when he said, Christ redeemed us from the curse of the law by becoming a curse for us. He became a curse for us because he'd become sin for us. And by those words for us, Paul indicates that Christ did this in our place and as our substitute.
            Imagine there's a moral ledger recording every event of your entire life - all your thoughts, words, actions, even your motives. You might think of it as a mixture of good and bad deeds, with hopefully more good than bad. The Scriptures, however, tell us that even our righteous deeds are unclean in the sight of God (Isaiah 64:6). So Jesus has a perfectly righteous moral ledger, and we have a completely sinful one. However, God took our sins and charged them to Christ, leaving us with a clean sheet.
            The biblical word for this is forgiveness. In and of itself, forgiveness is a monumental blessing. Paul echoed David on this when he wrote, Blessed are those whose lawless deeds are forgiven, and whose sins are covered; blessed is the man against whom the Lord will not count his sin (Romans 4:7-8; Psalm 32:1-2). But how did God do this and yet remain perfectly holy and just?
            He did it by causing the sinless Son to bear our sins, including everything that goes with them: our guilt, our condemnation, our punishment. That's what it took for God to wipe our moral ledger sheet perfectly clean and at the same time preserve his holiness and justice - the price had to be paid on our behalf; so the sentence was executed on our Substitute.
            CHRIST'S RIGHTEOUSNESS CREDITED
            TO US
            But it wasn't enough for us to have a clean, but empty, ledger sheet. God also credits us with the perfect righteousness of Christ so that in him we might become the righteousness of God. This happens the same way Jesus was made to be sin -  by transfer. Just as God charged our sin to Christ, so he credits the perfect obedience of Jesus to all who trust in him. In what is often called the Great Exchange, God exchanges our sin for Christ's righteousness. As a result, all who have trusted in Christ as Savior stand before God not with a clean-but-empty ledger, but one filled with the very righteousness of Christ!
            The theological term for what we've just described is one of Paul's favorite words, justification. The word justified in Paul's usage means to be counted righteous by God. Even though in ourselves we're completely unrighteous, God counts us as righteous because he has appointed Christ to be our representative and substitute. Therefore when Christ lived a perfect life, in God's sight we lived a perfect life. When Christ died on the cross to pay for our sins, we died on the cross. All that Christ did in his sinless life and his sin-bearing death, he did as our representative, so that we receive the credit for it. It's in this representative union
            with Christ that he presents us before the Father, holy and blameless and above reproach (Colossians 1:22).
            There's an old play on the word justified: just-as-if-I'd never sinned. But here's another way of saying it: just-as-if- I'd always obeyed. Both are true. The first refers to the transfer of our moral debt to Christ so we're left with a clean ledger, just as if we'd never sinned. The second tells us our ledger is now filled with the perfect righteousness of Christ, so it's just as if we'd always obeyed. That's why we can come confidently into the very presence of God (Hebrews 4:16; 10:19) even though we're still sinners - saved sinners to be sure, but still practicing sinners every day in thought, word, deed, and motive.
            The perfect righteousness of Christ, which is credited to us, is the first bookend of the Christian life. The news of this righteousness is the gospel. Christ's righteousness is given to us by God when we genuinely trust in Christ as our Savior. From that moment on, from God's point of view, the first bookend is permanently in place. We're justified; we're credited with his righteousness. Or to say it differently, we're clothed with his righteousness (Isaiah 61:10) so that as God looks at us in union with Christ, he always sees us to be as righteous as Christ himself.
            And that changes everything.
            THE PRESENT REALITY OF OUR JUSTIFICATION
            From our point of view, however, we sometimes handle our books as though the bookend of Christ's righteousness is not in place on our bookshelf. We do this when we depend on our own performance, whether good or bad in our estimate, as the basis of God's approval or disapproval. And when we take this approach, our assurance that we stand before God as justified sinners inevitably fades.
            How can we experience the righteousness of Christ as it was meant to apply to our daily lives? In Galatians 2:15-21, Paul provided much insight on this, beginning with this sentence:
            We know that a person is not justified by works of the law but through faith in Jesus Christ, so we . . . have believed in Christ Jesus, in order to be justified by faith in Christ and not by works of the law, because by works of the law no one will be justified. (Galatians 2:16)
            In this single sentence Paul uses the word justified three times. The repetition emphasizes that we're justified not by our personal obedience to the law but by faith in Christ.
            In this context, faith involves both a renunciation and a reliance. First, we must renounce any trust in our own performance as the basis of our acceptance before God. We trust in our own performance when we believe we've earned God's acceptance by our good works. But we also trust in our own performance when we believe we've lost God's acceptance by our bad works - by our sin. So we must renounce any consideration of either our bad works or our good works as the means of relating to God.
            Second, we must place our reliance entirely on the perfect obedience and sin-bearing death of Christ as the sole basis of our standing before God - on our best days as well as our worst.
            Just a few sentences later Paul wrote, The life I now live in the flesh I live by faith in the Son of God, who loved me and gave himself for me (Galatians 2:20). In the context of Galatians 2:15-21, it's clear Paul is still talking about justification, yet he's using the present tense. He writes of the life he lives now in the flesh. This raises an apparent problem. We know justification is a past event - the moment we genuinely trusted in Christ we were justified, declared righteous by God. That's why Paul wrote, We have been justified [past tense] by faith (Romans 5:1). So if justification was a point-in-time past event for Paul, why in Galatians 2:20 does he speak in the present tense: The life I now live [today] . . . I live by faith in the Son of God?
            The answer to this question is important. It tells us how to experience the application of the first bookend to our daily lives. For Paul, justification was not only a past event; it was also a daily, present reality. So every day of his life, by faith in Christ, Paul realized he stood righteous in the sight of God - he was counted righteous and accepted by God as righteous -  because of the perfectly obedient life and death Christ provided for him. He stood solely on the rock-solid righteousness of Christ alone, which is our first bookend.
            We must learn to live like the apostle Paul, looking every day outside ourselves to Christ and seeing ourselves standing before God clothed in his perfect righteousness. Every day we must re-acknowledge the fact that there's nothing we can do to make ourselves either more acceptable to God o r less acceptable. Regardless of how much we grow in our Christian lives, we're accepted for Christ's sake or not accepted at all. It's this reliance on Christ alone, apart from any consideration of our good or bad deeds, that enables us to experience the daily reality of the first bookend, in which the believer finds peace and joy and comfort and gratitude.
            Before battery-powered watches were invented, wristwatches had to be wound every day. A watch's stem was used not only to adjust the hands but also to wind up the mainspring. The gradual unwinding of the mainspring throughout the day drove the mechanism of the watch to keep time. The gospel of justification by faith in Christ is the mainspring of the Christian life. And like the mainspring in old watches, it must be wound every day. Because we have a natural tendency to look within ourselves for the basis of God's approval or disapproval, we must make a conscious daily effort to look outside ourselves to the righteousness of Christ, then to stand in the present reality of our justification. Only then will we experience the stability that the first bookend is meant to provide.
            But if it's true God's acceptance of me and his blessing on my life is based entirely on the righteousness of Christ, what difference does it make how I live? Why should I make any effort? Why should I put myself through the pain of dealing with sin and seeking to grow in Christlike character if it doesn't affect my standing with God? We'll answer these questions in the next chapter.
            CHAPTER TWO
            THE MOTIVATION OF
            THE GOSPEL
            For the love of Christ controls us.
            2 CORINTHIANS 5:14
            To explore the gospel's motivating power, we'll look at the experience of three Bible characters: a sinful woman who met Jesus, a highly respected Jew who encountered the holiness of God, and a self-righteous Pharisee who discovered he was dead wrong.
            THE SINFUL WOMAN
            One of the most profound examples of how the gospel motivates and transforms us is seen in the story in Luke 7:36­50 of a sinful woman who encountered Jesus. The story begins with a Pharisee named Simon inviting Jesus to his house for dinner. While Jesus and the other guests reclined at the table with their feet behind them in the manner of that time, a woman who was a sinner came with a flask of expensive ointment to anoint the feet of Jesus.
            In those days, at a dinner for a special guest, it wasn't unusual for uninvited visitors to enter and sit around the edge of the room, listening to the table conversation. What made this incident remarkable was that a woman whose ill-repute was well known would dare to enter the house of a highly religious Pharisee.
            But this woman did not come merely to hear the conversation. She was on a mission. Rather than take a seat at the edge of the room, she went straight to Jesus. As she stood at his feet, she began to weep - not just a few trickling tears, but so profusely that his feet became wet with them. Kneeling down, the woman loosened her long tresses - a shameful act according to the custom of the day - and began to dry Jesus' feet with her hair. Bending lower, she kissed his feet, then anointed them with the expensive ointment she had brought.
            She hadn't preplanned her actions. Jesus was already at Simon's house when she learned (verse 37) he was there. She must have rushed home, grabbed the flask of expensive ointment, and hastened to the dinner. She wanted only to anoint his feet. The tears and the drying of his feet with her hair were spontaneous. The question naturally arises: why would she dare to do this? To answer, let's continue the story.
            The woman's actions and the lack of offense from Jesus were duly noted by Simon, a self-righteous Pharisee. He concluded that Jesus couldn't possibly be a prophet, or else he would have known what sort of sinner she was and wouldn't have allowed her to touch him. Reading Simon's mind, Jesus told him a parable of a moneylender who had two debtors. One owed five hundred denarii, the other fifty. When they couldn't pay, the moneylender cancelled both debts. Jesus asked Simon, Which of them will love him more? Simon replied, The one, I suppose, for whom he cancelled the larger debt.
            Jesus said, You have judged rightly.
            Jesus then compared the uncaring treatment he'd received from Simon to the woman's lavish acts of adoration. What prompted the difference? Jesus went right to the root: He who is forgiven little, loves little. Since Simon sensed no need of forgiveness from Jesus, he showed little if any love for him.
            By contrast, the sinful woman lavished her love on the Savior because she realized she'd been forgiven much. When was she forgiven? Though Luke doesn't tell us, the only way this story makes sense is to assume the woman had previously encountered Jesus and been forgiven for her sins at that time. She wasn't forgiven because she loved much; rather, she loved much because she'd been forgiven much.
            We can trace three steps in the woman's experience. She'd become deeply convicted of her many sins through her initial encounter with Jesus. She then received from him the assurance that her sins were forgiven. These two steps - deep conviction of sin and assurance of forgiveness - prompted the third: love and gratitude on her part. The dinner at Simon's house provided an occasion for her to publicly display these feelings. She displayed much love because she'd been forgiven so much.
            There's an important lesson here for all of us. Genuine love for Christ comes through (1) an ever-growing consciousness of our own sinfulness and unworthiness, coupled with (2) the assurance that our sins, however great, have been forgiven through his death on the cross. Only love that's founded on both of these foundations can be authentic and permanent. If we find we lack love for the Savior, one or both of these prerequisites are deficient.
            You may be wondering why Jesus told the woman, Your sins are forgiven, if he'd already forgiven them earlier. He wanted to make her forgiveness public. Remember, she was a well-known sinner with a bad reputation. Simon and his other guests needed to hear Jesus' words. And imagine what those words meant to the woman. She well knew what Simon and his guests thought of her. To be publicly reassured of her forgiveness must have sent her home with an even greater sense of love and gratitude for the Savior.
            What about the ointment? Isn't sacrificial giving the point of the story? Yes and no. It's true the ointment was very expensive; it may have been worth nearly a year's wages. But don't miss the fact that her acts of worship included evidence of heartfelt gratitude and deep affection. The ointment was merely an outward symbol of a life now dedicated to Jesus. She was forgiven much, and she loved much; she gave not only her ointment but her heart as well.
            When we've truly experienced the gospel, far from producing a why bother to grow? attitude, it has just the opposite effect. It motivates us to lay down our lives in humble and loving service out of gratitude for grace.
            A HIGHLY RESPECTED JEW
            Little is known about the prophet Isaiah except that he ministered in and around Jerusalem and had ready access to Judah's kings. As such, he was undoubtedly a highly
            respected and very moral man. Isaiah recorded the details of an encounter he had one day with God:
            I saw the Lord sitting upon a throne, high and lifted up; and the train of his robe filled the temple. Above him stood the seraphim. Each had six wings: with two he covered his face, and with two he covered his feet, and with two he flew. And one called to another and said: Holy, holy, holy is the Lord of hosts; the whole earth is full of his glory! And the foundations of the thresholds shook at the voice of him who called, and the house was filled with smoke. (Isaiah 6:1-4)
            The threefold ascription by the angelic seraphim, Holy, holy, holy, meant that they attributed an infinite degree of holiness to God. The entire scene, especially this revelation of God's holiness, had a devastating impact on Isaiah. Overwhelmed by acute awareness of his own sinfulness, he cried out in desperation, Woe is me! For I am lost; for I am a man of unclean lips, and I dwell in the midst of a people of unclean lips; for my eyes have seen the King, the Lord of hosts! (verse 5). This is remarkable considering Isaiah was a member of the religious elite, totally on the opposite end of the moral spectrum from the sinful woman of Luke 7. But righteous though he was in outward morality, in light of God's infinite holiness Isaiah essentially placed himself on the same plane as the woman.
            As Isaiah anguished over his newly discovered sinfulness, God sent one of the seraphim with a burning coal from the altar. As he touched Isaiah's mouth with it, the seraph said, Behold, this has touched your lips; your guilt is taken away, and your sin atoned for (verse 7). In this good news, Isaiah heard the gospel. Like the sinful woman, Isaiah also experienced both the deep conviction of his sin and the assurance of God's gracious forgiveness. Isaiah's response was also similar. When he heard the voice of the Lord saying, Whom shall I send, and who will go for us? he responded, Here am I! Send me (verse 8). Isaiah gave his life in service to God. He essentially offered himself as a blank check, to be filled in as God saw fit.
            Isaiah's experience parallels that of the sinful woman. Though we don't know exactly what brought about her deep consciousness of sin, it was undoubtedly connected with being in the presence of Jesus and sensing the vast gulf between his holiness and her sinfulness. With Isaiah we see the same three-step process: first, acute realization of one's own sinfulness in the light of God's holiness; second, hearing the gospel that one's sins are forgiven; and finally the response of gratitude, love, and surrender leading to action.
            We may not think we are as sinful as the woman in Luke 7; we're certainly no more righteous than Isaiah. But wherever we are on the moral spectrum, we all need to experience this same three-step process deep in our souls. For the sinful woman and Isaiah, these steps came suddenly and dramatically. For many of us, such realizations may come in stages as we gradually grow in the Christian life. But whether suddenly or slowly, we should aim to increase our awareness of God's holiness and our sinfulness, coupled with an ever-deepening understanding of the meaning and application of the gospel. As we do, we, too, will respond with genuine gratitude and commitment to God; we'll experience the motivating power of the gospel, and our lives will be progressively transformed.
            THE
            SELF-RIGHTEOUS
            PHARISEE
this chapter appears designed to reinforce the idea that christ in his kingly rule will defeat those who have been persecuting his people
            the book ends with a fearful warning not to add to or take away from the book, a final promise of the lord's coming, and a two-fold prayer calling for the lord jesus to come, and for his grace to be with all the brethren         
            """
  
  result = vb_translate(raw_str)
  # with open('/home/vgm/Desktop/test/test.txt', 'r') as f:
  #   result = vb_translate(f.read())
  print("result::", result)

