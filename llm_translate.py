from dotenv import load_dotenv
import requests
import time
import os
import re
# import shutil
# import json
import random
from tqdm import tqdm
import joblib
from joblib import Parallel, delayed
from langdetect import detect
# from vietTTS.utils import concise_srt
# from utils.utils import srt_to_segments, segments_to_srt
from utils.language_configuration import LANGUAGES
from langchain_openai import ChatOpenAI
import concurrent.futures
from requests.exceptions import RequestException
import threading
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.output_parsers import StrOutputParser

load_dotenv()

fault_words = [
  "im_start",
  "im_end",
  "<skip_think>"
  "<think>",
  "</think>",
  "```",
  "English:",
  "Vietnamese:",
  "English–Vietnamese"
]

default_endpoints = [
    "http://172.27.188.32:8082/v1"
    # "http://192.168.2.12:8081/v1",
    # "http://192.168.2.13:8081/v1",
    # "http://192.168.2.14:8081/v1",
    # "http://192.168.2.14:8082/v1",
]

def cleanup_text(text):
    if '</think>' in text:
      text = text.split('</think>')[1]
    if '<skip_think>' in text:
      text = text.split('<skip_think>')[1]
    return text
  
def is_valid_input(source_text):
  return True

def is_valid_response(source_text, target_text):
    source_text = str(source_text).replace("-", " ").strip()
    target_text = str(target_text).replace("-", " ").strip()
    source_len = len(source_text)
    target_len = len(target_text)

    print(f"----- length count ----- source: {source_len} - target {target_len} | {target_len/source_len}")

    if target_text == '':
        print("-----invalid response ---- : empty target_text")
        return False
    elif any(word in target_text.lower() for word in fault_words):
        print("-----invalid response ---- : fault_words")
        return False
    elif target_len/source_len > 2 or source_len/target_len > 2:
        print("-----invalid response ---- : length not match::" ,source_len, target_len)
        return False

    return True

def post_process(target_text):
  return target_text  
 
class LLM():
  def __init__(self, systemPrompt = "") -> None:
    self._monitor_thread = None
    self.running = False
    self.llm_chain = {}
    self.endpoints = default_endpoints
    self.interval = 5
    self.timeout = 2
    self.model = ""
    self.api_key = ""
    self.temp = 0.3
    self.k = 10
    self.available_endpoints = set(default_endpoints) 
    self.systemPrompt = systemPrompt if systemPrompt != "" else "This GPT functions as a translation tool that processes text from {source_language}, translating it into {target_language}. The output is a plain text content with a full translation in {target_language}. It accepts input in the form of {source_language} text, ensuring the texts are accurately digitized and represent the original manuscripts. The translation engine interprets and translates words into modern {target_language}, incorporating linguistic analysis to handle idiomatic expressions and cultural nuances. Response only translated text."
    self.prompt = ChatPromptTemplate(
          messages=[
              SystemMessagePromptTemplate.from_template(self.systemPrompt),
              # The `variable_name` here is what must align with memory
              MessagesPlaceholder(variable_name="history"),
              HumanMessagePromptTemplate.from_template("{input}"),
          ]
      )

  def check_endpoint(self, endpoint: str):
      url = f"{endpoint}/models"
      try:
          response = requests.get(url, timeout=self.timeout)
          
          if response.status_code == 200:
              self.available_endpoints.add(endpoint)  # Add to available endpoints
              chain = ChatOpenAI(
                          model=self.model,
                          openai_api_key=self.api_key,
                          openai_api_base=endpoint,
                          # max_tokens=2048,
                          temperature=self.temp,
                          top_p= 0.95,
                          frequency_penalty=1.3,
                          stop=["<|im_end|>"],
                      )
              self.llm_chain[endpoint] = RunnableWithMessageHistory(
                  self.prompt | chain | StrOutputParser(),
                  lambda session_id: ChatMessageHistory(),  # Factory for creating history storage
                  input_messages_key="input",               # Key for input messages
                  history_messages_key="history",      # Key for history in the chain
                  window_size=self.k                            # This is equivalent to the 'k' parameter - keep last 2 exchanges
              )
          else:
              self.available_endpoints.discard(endpoint)  # Remove from available endpoints
              if endpoint in self.llm_chain:
                del self.llm_chain[endpoint]
      except RequestException as e:
          self.available_endpoints.discard(endpoint)  # Remove from available endpoints
          if endpoint in self.llm_chain:
            del self.llm_chain[endpoint]
      return url
      
  def monitor(self):
      while self.running:
          with concurrent.futures.ThreadPoolExecutor() as executor:
              # Check all endpoints concurrently
              future_to_endpoint = {
                  executor.submit(self.check_endpoint, endpoint): endpoint 
                  for endpoint in self.endpoints
              }
              
              for future in concurrent.futures.as_completed(future_to_endpoint):
                  result = future.result()
                  print("Available llm endpoints::\n", result)
          if len(self.available_endpoints) >= 1:
            self.interval = 30
          else:
            self.interval = 5
          time.sleep(self.interval)
          
  def start(self):
      """Start the monitoring in a separate thread"""
      if not self.running:
          self.running = True
          self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
          self._monitor_thread.start()
          print("Endpoint monitoring started in background thread")
      else:
          print("Monitoring is already running")
                
  def stop(self):
      """Stop the monitoring thread"""
      if self.running:
          self.running = False
          if self._monitor_thread:
              self._monitor_thread.join(timeout=1)
          print("Endpoint monitoring stopped")
      else:
          print("Monitoring is not running")

  def initLLM(self, endpoints="", model="", api_key="", temp=0.5, k=5):
    print("Initializing LLM::")
    endpoints = endpoints.split(',')
    self.endpoints = list(set(default_endpoints + endpoints))
    self.endpoints = self.endpoints if len(self.endpoints) > 0 else ["https://openrouter.ai/api/v1"]
    self.temp = temp
    self.k = k
    self.model = model if model != "" else "openai/gpt-4o"
    self.api_key = api_key if api_key != "" else os.getenv("OR_API_KEY", "")
    self.start()
    return True
        
  def process(self, text, source_lang="en", target_lang="vn"):
    if is_valid_input(text):
      max_attempts = 5
      attempts = 0
      source_language = next((key for key, value in LANGUAGES.items() if value == source_lang), None)
      target_language = next((key for key, value in LANGUAGES.items() if value == target_lang), None)

      while attempts < max_attempts:
        try:
          if len(list(self.llm_chain.values())) >= 1:
            llm_chain = random.choice(list(self.llm_chain.values()))
            print('translate inferencing::', source_language, target_language)
            result = llm_chain.invoke({
                      "input": text,
                      "source_language": source_language,
                      "target_language": target_language,
                  }, config={"configurable": {"session_id": "default_session"}
            })
            if is_valid_response(text, cleanup_text(result)) and target_lang in detect(result):
                return result, post_process(cleanup_text(result))
        except Exception as e:
          print("error::", e)
          result = ""
        print(f"re-run {attempts}:\n")
        time.sleep(2)
        attempts += 1
      return text, text
    else:
      return text, text

  def translate(self, segments, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      N_JOBS = len(self.available_endpoints) * 7 if len(self.available_endpoints) else 20
      print("Start LLM Translate:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(N_JOBS)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(segments[line]['text'], source_lang, target_lang) for (line) in tqdm(range(len(segments))))
      for index in tqdm(range(len(segments))):
        segments[index]['source'] = segments[index]['text']
        segments[index]['think'] = t2t_results[index][0]
        segments[index]['text'] = t2t_results[index][1]
        segments[index]['systemPrompt'] = self.systemPrompt
      return segments
    
  def predict(self, text, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      result = self.process(text, source_lang, target_lang)
      return result  
    
# if __name__ == '__main__':
  
#   # systemPrompt="""Sửa lỗi chính tả từ bản gốc sang bảng mới"""
#   # llm = LLM(systemPrompt=systemPrompt)
#   # llm.initLLM(
#   #   endpoints="https://openrouter.ai/api/v1", ## http://172.27.188.32:8081/v1
#   #   model="openai/gpt-4o", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
#   #   api_key=os.getenv("OR_API_KEY", ""),
#   #   temp=0.3,
#   #   k=10
#   # )
  
#   # systemPrompt="""Think and translate English accurately into clear, natural, appropriate Vietnamese."""
#   # llm = LLM(systemPrompt=systemPrompt)
#   llm = LLM()
#   llm.initLLM(
#     endpoints="http://172.27.188.40:8082/v1", ## http://172.27.188.31:8081/v1
#     model="trast-ai/trust-translator-0525", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
#     api_key="EMPTY",
#     temp=0.1,
#     k=10
#   )
#   # text = "Reason and science are gifts from god that help us discern these patterns, and for this reason evangelicals write, value rational and scientific research into the pentateuch"
#   # result = llm.predict(text, "en", "vi")
#   # print("source::", text)
#   # print("target::", result)
    
#   # ## Translate segments
#   # input_file = '/home/vgm/Desktop/en.srt'
#   # segments = srt_to_segments(input_file)
#   # # segments = concise_srt(segments)
#   # # segments_to_srt(segments, '/home/vgm/Desktop/en.srt')
#   # # print(segments, len(segments))
#   # segments = llm.translate(segments=segments, source_lang="en", target_lang="vi")
#   # # print("results::",  segments, len(segments))
#   # segments_to_srt(segments, '/home/vgm/Desktop/vi.srt')
  
#   texts = """
# Imagine for a moment growing up with a close sibling or friend . . you play together , you learn together , you reach adulthood together . For most of your life this person has been right by your side and then one day your friend or sibling claims to be God 's chosen one .
# Well , for James , the brother of Jesus , this was not just an imaginary scenario . In his younger years , he doubted that Jesus was the Savior . But later in his life , he not only became a follower of Jesus , he became the leader of the church at Jerusalem and wrote the New Testament book that bears his name .
# This is the first lesson in our series on the Epistle of James , and we 've entitled it Introduction to James . In this lesson , we 'll touch on a number of introductory issues that will enable us to pursue a faithful interpretation of this portion of the New Testament .
# We 'll approach our introduction to James in two ways . First , we 'll explore the background of the book . And second , we 'll examine its structure and content . Let 's begin with the background of the book of James . With any biblical book , it 's important to understand the context surrounding its writing as much as possible .
# The various books of the Bible were written in real historical settings by people with particular motivations and concerns . So , studying these kinds of background issues can help us understand the books themselves . When we consider the settings and motivations associated with the book of James , we 're better equipped to understand what the eclipse meant when it was first written .
# and we can apply James ' words more effectively to our lives today . To understand the background of James , we 'll consider first the authorship of the book . Then we 'll look at the original audience . And finally , we 'll examine the occasion on which the letter of James was written . Let 's begin with the authorship of the Epistle of James .
# Although we know that the Holy Spirit inspired the Scriptures , many books in the Bible , like James , also identify their human authors . And the more we know about biblical authors , then the better prepared we are to understand and interpret what they wrote . So for this reason , we must learn all we can about who wrote the Epistle of James .
# To investigate the authorship of James , we 'll consider two subjects . First , we 'll explore the traditional outlook that James , the younger brother of Jesus , wrote the eclipse . Second , we 'll explore the author 's personal history . Let 's start by looking at the traditional outlook on these matters .
# The letter opens in James chapter 1 verse 1 with this simple statement , James , a servant of God and of the Lord Jesus Christ to the 12 tribes scattered among the nations . Greetings . As we see here , the letter clearly identifies a man named James as the author . But this greeting does not settle precisely who this man was .
# Five different men in the New Testament , including two of Jesus ' twelve disciples , were named James . But only two of these five men would have had enough authority in the early Church to write a letter like this .
# The first of these two was James the son of Zebedee and the brother of John . But according to Acts chapter 12 verse 2 , this James was martyred under Howard Agrippa the first around AD 44 . As we 'll see later , there are good reasons for believing that the book of James was written after Herod 's death . So it 's highly unlikely that James the son of Zimbabwe was the author .
# The second James was the younger brother of Jesus . He was also the leader of the early church in Jerusalem . This James was by far the more prominent of the two and the one most theologians have attributed this eclipse to through the centuries .
# There is a great deal of support for the traditional outlook that Jesus ' brother James wrote this article . But there are also a few objections . Let 's begin with the support for this view .
# In the first place , in chapter 1 verse 1 , the writer did not give any credentials beyond saying that he was a servant of God and of the Lord Jesus Christ . He simply assumed that his name alone would be recognized and would carry sufficient authority . And based on this authority , his letter contains one strong command after another .
# This opening greeting , then , makes a strong case for Jesus ' brother James because of his status in the early church in Jerusalem .
# Well , in the days of the Apostolic Church , the whole question of authority was very significant . Who has the authority to teach and lead this new community of followers of Jesus Christ ?
# There were various writings that were circulating , various claims to have authority . And one of the criteria that emerged is very significant , was that of being an eyewitness to the ministry .
# of Jesus , those who were eyewitnesses of his ministry , who spent the time with the Lord himself , were considered to have a righteous claim to the authority to teach in the early church .
# James , the brother of Jesus , of course , was an eyewitness to his ministry , but more than that had been eyewitness really to the whole of his life . And that did play a significant role in the weight that the teaching of James and the weight that James ' letter was given in the early church .
# In the second place , the testimony of the early Church confirms this outlook on the authorship of the book .
# The first eclipse of Clement , written around AD 96 , and the Shepherd of Hammers , written around AD 140 , both either refer to or quote from James ' eclipse . And Oregon , who died in AD 254 , quoted the book of James several times in his commentary on the opposite to the Romans .
# Origin 's use of James is particularly significant because in Book 4 Chapter 8 , Oregon identified the author of James as the brother of the Lord . We also know that the Church in the East , and later the Church in the West , accepted this letter as the work of Jesus ' brother .
# Now , despite this strong support for the traditional outlook that Jesus ' brother James was the author , there have been some objections . Critical interpreters have suggested at least two alternatives . Some interpreters have looked for an unknown James in the early Church .
# They say that the person who wrote the letter was indeed named James , but he was not the son of Zebedee or the brother of Jesus . He remains obscure because he was not mentioned in any other writings of the infant church .
# However , this theory is unlikely . As we 've already noted , the simplicity of the author 's identification at the beginning of the letter indicates that he was well known . It 's highly doubtful that there would have been nothing else written about him . A second theory offered by critical interpreters is that of pseudonymity .
# Pseudonymity refers to the practice of assigning written works to someone other than the actual author . This practice took place among Jews in the first century for a variety of reasons . One prominent reason for pseudonymity was to give weight or authority to a book or letter .
# In the case of James ' eclipse , critical interpreters have argued that someone other than James used his name to gain wider acceptance for their letter in the church . Now , according to passages like 2 Thessalonians chapter 2 verse 2 , this practice was scorned in the first century church as deceit . But critical scholars still offer at least three arguments for this objection .
# First , they say there is no mention of the author 's relation to Jesus . They say it 's unthinkable that a brother of Jesus would write to the churches and not reveal this familial bond when he identified himself .
# But Jude , the author of the eclipse of Jude , was also Jesus ' brother , and he never mentioned his blood ties to Jesus in his letter . So this argument for pseudonymity is weak at best .
# Second , some critical scholars assume pseudonymity because the book gives evidence that the author was aware of Hellenistic or Greek culture . And James was a Jew from Palestine . It 's true that the writer of James had some awareness of Greek culture .
# For instance , in James chapter 3 verse 6 , he used the phrase , the whole course of one 's life . This phrase was commonly used in Greek philosophy and religion . But at the time James later was written , many well -- educated Jews in Palestine had more than a passing knowledge of Hellenistic philosophy and religion .
# In addition , while the Greek of James is more sophisticated than what we find in other portions of the New Testament , it is not by any means the most sophisticated Greek in the New Testament . In fact , the letter is quite similar in style to books such as Testaments of the Twelve characters and other Hellenistic Jewish writings of that time .
# A third argument for pseudonymity points to inconsistencies with the theological portrait of James in the books of Acts and Galatians .
# This view suggests that some of the ideas expressed in the Epistle of James do not match theological outlooks attributed to James in these other New Testament books . For instance , critical interpreters point to passages like Acts chapter 21 verses 17 through 25 and Galatians chapter 2 verse 12 .
# They argue that in these verses , James appears to be a spokesman for a rather conservative Jewish Christian position on the law . But in James chapter 1 verse 25 and James chapter 2 verse 12 , the author seems to take a somewhat lenient view of the law , calling it the law that gives freedom .
# But these differences simply are not as great as critical scholars make them out to be . On closer review , the verses cited in Acts and Galatians do not portray an extreme Jewish Christian point of view . And James ' position on the law in Acts and Galatians is , actually , very consistent with the theology of the letter of James .
# As we can see , the arguments against James , the brother of Jesus , being the author of this book are weak at best . The arguments in favor of James ' authorship are much more compelling . And because of this , most evangelical scholars rightly affirm that James , the brother of Jesus , was the author of the letter that bears his name .
# We 've considered the authorship of James by looking at the traditional outlook . Now let 's look more closely at James ' personal history . Matthew chapter 13 verse 55 identifies James as one of Mary 's sons and one of Jesus ' half -- brothers .
# This family connection may account for the many similarities between James ' eclipse and Jesus ' teachings recorded in the Gospels . But Scripture makes it clear that when James and his other brothers were growing up , they did not recognize who their oldest sibling really was . As John chapter 7 verse 5 tells us , even Jesus ' own brothers did not believe in him .
# But at some point in his life , James came to have saving faith in Jesus as his Lord . In fact , James rose to such prominence in the early church that Paul called him , in Galatians chapter 2 verse 9 , one of the pillars of the church . In addition , we know that , according to 1 Corinthians chapter 15 verse 7 , Jesus appeared to James after his resurrection .
# James ' position of authority is well documented in the New Testament . For instance , he appears three times in the Book of Acts as the leader of the Jerusalem church . And in Acts chapter 15 , we see him as the spokesman for the apostolic council . Even none -- Christians acknowledge James ' importance in the church .
# One of the most well -- known accounts of James ' violent death in AD 62 comes from the Jewish historian Josephs . Listen to Antiquities , Book 20 , Chapter 9 , Section 1 , written in AD 93 , where Josephs described the circumstances surrounding James ' death .
# Anninus convened the judges of the Sheridan and brought before them the brother of Jesus , the one called Christ , whose name was James , and certain others , and accusing them of having transgressed the law , delivered them up to be stoned .
# While growing up , James may not have understood who his older brother really was . But we can see from Josephus ' account and from scripture and other historical records that later in his adult life , James had an unwavering commitment to Jesus as the Christ .
# As Eusebius wrote in his Ecclesiastical History , Book 2 , Chapter 23 , quoting the early Christian historian , Ijesipus , James became a true witness , both to Jews and Greeks , that Jesus is the Christ .
# Now that we 've considered the background of James ' eclipse by looking at some of the issues surrounding authorship , let 's explore the original audience of this letter . Technologies often spend a great deal of time and energy trying to learn as much as possible about the author of a particular biblical book .
# But discovering the identity of the original audience is just as important . If we want to interpret correctly what a biblical writer was saying , it helps us to know who the writer 's original readers were and what they were facing at that particular time in history .
# As we saw earlier , in James chapter 1 verse 1 , James identified his readers as the 12 tribes scattered among the nations . This seems to be a reference to Jews who lived outside of Israel . And in James chapter 2 verse 1 , James addressed his audience as believers in our glorious Lord Jesus Christ .
# Taken together , these verses indicate that James ' original audience was made up , primarily , of Jewish Christians who lived outside of Palestine . On several occasions in his book , James addressed his audience affectionately as brothers . But how did James , living in Jerusalem , know his audience well enough to speak to them in this way ?
# Well , in Acts chapter 8 verses 1 through 4 , we learn that in the wave of persecution following Stephen 's martyrdom , members of the Jerusalem church were scattered throughout Judea and Samaria .
# It 's possible then that James , as the leader of the Jerusalem church , was writing to these scattered members of the 12 tribes . But even if the eclipse was not addressed specifically to these believers , it seems that James ' audience was made up of Jewish Christians in similar circumstances . The vocabulary James used also supports the idea that his original readers were Jewish followers of Jesus .
# For example , in chapter two , verse two , James chose the word synagogue , or synagogue , to describe his audience 's meetings .
# This was a typical way to refer to Jewish gatherings . And in chapter 5 verse 4 , James used the phrase , Lord Almighty , or Kurias Sabbath . This phrase comes from a common Old Testament name for the God of Israel , Yahweh Sabbath . Language of this kind makes much more sense if the recipients have strong Jewish roots .
# Knowing the background to James ' audience is extremely important because it helps us set a trajectory as to how we understand the message that he 's trying to articulate to his audience . James ' audience as a Jewish community are recipients of a long tradition of
# the Torah of Moses , the message of the prophets and the writings . James draws on this rich tradition as he talks to them about the life of faith , the wise life , and they need to understand how they should apply it into their own lives in light of the resurrection of Jesus Christ .
# Now , when we say that James was writing to Jewish Christians , we do not mean that there were no Gentile believers in the churches James addressed . As early as Acts chapter 8 , we know of an Ethiopian convert .
# And as we learn in Acts chapter 10 , there were many Gentile God -- fearing converts to Judaism who attended synagogues . So it would not have been surprising to find at least some Gentile believers in these churches as well . Still , according to Romans chapter 9 verse 8 , Gentile believers were regarded as Abraham 's offspring .
# And ideally , they were considered just as much as part of the 12 tribes of Israel as any who were Jews by bloodline . We 've looked at the background of James by considering the Epistle 's authorship and its original audience . Now we 're ready to examine the occasion of its writing .
# We 'll explore the occasion of the writing of James in three steps . First , we 'll touch on the location of both the author and audience . Second , we 'll consider the date of composition . Third , we 'll think about the purpose of James ' eclipse . Let 's begin by looking at the location of both the author and the audience of this letter .
# The location of the author is not difficult to discern . Both the New Testament and early Church Fathers suggest that James lived his life of ministry in Jerusalem . And he remained in Jerusalem until he was martyred in AD 62 . Because of this , there 's no reason to think that he wrote the respite from any other location .
# The location of the original audience is also somewhat straightforward . As we just mentioned , the letter 's recipients were most likely Jewish believers who had been scattered throughout Judea and Samaria after the murder of Stephen .
# Acts chapter 11 verse 19 tells us that these displaced believers traveled as far as Phoenicia , Antioch , and Cyprus in search of a safe place to live . We can not be positive that James wrote to believers in these specific locations .
# Yet , based on James ' initial greeting to the 12 tribes scattered among the nations , these areas are strong possibilities for the location of James ' original audience .
# We really think that these are truly dispersed tribes , that is , the parishioners of Jerusalem who were scattered into Phoenicia and Cyprus and Antioch by the persecution after Stephen 's martyrdom .
# that it 's quite possible , in fact I think it likely , that James was writing to these folks as his own parishioners . And the reason I think that is that he surprisingly gives us no theology or virtually none overtly . He does not talk in terms of the structure of the gospel . There are quite a few things that he does not mention .
# And as a pastor I 'm thinking well he probably covered those things earlier in his ministry and now he 's speaking to his well -- known audience in the way that a pastor would . And so it has great effect on our sense of James that we look at this audience scattered , this audience already under his ministry .
# and see him building in that way . Keeping in mind this first aspect of the occasion of James Epstein , the location of the author and audience , now let 's consider the date of the letter 's composition .
# The earliest and latest likely dates for this letter are fairly easy to establish . First , the earliest likely date for the letter 's composition is AD 44 . We know that James wrote his exploits as the leader of the early church in Jerusalem .
# Acts chapter 12 verse 17 indicates that James became a significant leader of the Jerusalem church by the time of Peter 's release from prison . According to Acts chapter 12 verses 19 through 23 , Peter was released in the year Herold Agrippa I died in AD 44 . This makes it most likely that the eclipse was not written much before this date .
# Second , the latest possible date of composition for the eclipse is AD 62 , the year of James ' martyrdom . As we saw earlier , according to Josephs , James died at the hands of the priest Anninus near this time . This provides a brief window for the letter 's composition .
# The letter itself does not include specific references to historical events that would date it more specifically . But there are at least two reasons to think that the date of composition was earlier rather than later .
# For one , as we mentioned before , in chapter 2 , verse 2 , James used the word synagogue , or synagogue , to describe his audience 's meetings . The use of synagogue seems to indicate an early stage in the development of the Christian movement .
# James may have written before Christians were forced out of the synagogues . Or , at the very least , he wrote at a time when Christians were still calling their gatherings a synagogue . In addition , there 's no mention in James ' eclipse of the Jewish -- Gentile controversies that received so much attention in the writings of Peter and Paul .
# In the early church , as Gentiles came to faith in Christ in large numbers , conflicts arose over whether or not these new believers should be required to conform to Jewish customs . Perhaps James simply chose not to deal with these controversies , but more likely , they had not yet become a major factor in the life of the young churches that James addressed .
# Having looked at the letter 's occasion both in its location and its date , let 's examine James ' purpose in writing this letter .
# One of the most helpful ways to summarize the overarching purpose of James is to look at James chapter 1 verses 2 through 4 . In his opening words , James told his readers , Consider it pure joy , my brothers and sisters , whenever you face trials of many kinds , because you know that the testing of your faith produces perseverance .
# Let perseverance finish its work so that you may be mature and complete , not lacking anything . As this passage indicates , James ' audience was facing trials of many kinds . But James called them to have pure joy in their trials .
# Trials , he explained , produce perseverance . And those who persevere will become mature and complete , not lacking anything . But the real key to James ' message comes in the very next verse .
# In verse 5 , James completed his thoughts with these words , If any of you lacks wisdom , you should ask God , who gives generously to all without finding fault , and it will be given to you .
# We 'll discuss these verses in more detail later in the lesson . But for now , this passage gives us a window into the heart of the entire eclipse . To experience pure joy in the midst of trials , ask God for wisdom , and it will be given to you .
# With this in mind , we can summarize the main purpose of James ' letter in this way . James called his audience to pursue wisdom from God so that they would have joy in their trials . It was important for James ' audience to hear this message .
# As we said earlier , James ' audience was no longer in Palestine . They were living scattered among the nations , far from their homes . No doubt , it was not easy for them to find joy in their trials . This appears to have led some of them to abandon their loyalty to Christ . Instead , they were pursuing what James called friendship with the world .
# Listen to James chapter 4 verse 4 where James used these strong words , You adulterous people , do not you know that friendship with the world is hatred toward God ? Anyone who chooses to be a friend of the world becomes an enemy of God .
# Clearly , there were some in James ' audience who had strayed far from the faith , and James warned them that being friends with the world made them an enemy of God . It 's no wonder , then , that James exerted his authority as a leader of the church .
# Repeatedly , James commanded his readers to live in a manner consistent with a sincere profession of faith . He used more than 50 imperatives or direct commands in his 108 verses .
# and he often used other grammatical forms that functioned just like imperatives within their contexts . But James ' principal solution to the problems his audience faced was not merely to command them to do this or that . For him , the heart of the matter was that they needed to pursue wisdom from God . Wisdom from God was the key to receiving joy as they endured their many trials .
# Listen to these well -- known words of chapter four , verses eight through 10 , where James told his readers , come near to God and he will come near to you . Humble yourselves before the Lord and he will lift you up .
# James directed believers to humble themselves so that God would lift them up . He taught that humility before God is a path to wisdom . And when Christ 's followers draw near to God in humble submission , the wisdom they receive brings joy , even as they persevere through trials .
# So far in our introduction to James , we 've looked at the background of James . Now we 're ready to examine the Epistle 's structure and content . We 've just suggested that the book of James focuses a great deal of attention on wisdom as the way to find joy in times of trial .
# But this emphasis on wisdom helps us understand something more than just the purpose of this book . Many interpreters have spoken of the book of James as the New Testament book of wisdom . And this perspective also helps us grasp the unusual structure and content of the eclipse .
# By the time James wrote his letter , there had been a long history of wisdom literature stemming from the Old Testament . Old Testament wisdom writings include Job and Ecclesiastes , as well as the Book of Petrobras and a number of so -- called wisdom palms and prophetic wisdom sayings .
# James ' indebtedness to this Old Testament literature is evident in a number of ways . For instance , in chapter 5 verse 11 , James used the example of Job , the main character in the book of Job , to promote perseverance . Beyond this , James touched on topics such as speech , the treatment of widows and orphans , poverty , and favoritism .
# These topics reflect numerous parallels with the content of the Book of Petrobras .
# When we read through the eclipse of James , one of the things that we see as a common thread is the word wisdom . He obviously values greatly wisdom , the wisdom from above as opposed to the wisdom from below . That very value in wisdom and the structure of the eclipse makes us think that there 's a great influence in his life on wisdom literature that 's come before him .
# Now I think we see that most explicitly in his citation and use of the book of Petrobras and also in the way that he remembers the words of our Lord , of Jesus , who also spoke often in a wisdom context . Alongside that , there was a development of wisdom thought and wisdom writing , a genre really of wisdom writing .
# in the intertestamental time . And I think we see some of the same themes through that wisdom literature in James . Occasionally we see the same structure . But I think a lot of the themes also were really started with the book of Petrobras and also with Jesus .
# And so I think that the bigger influence on James is probably going to come out of Jesus in Petrobras but that genre and the importance of proverbial wisdom throughout Second Temple Judaism around the time of Jesus is also very important in James .
# The letter of James also reflects the content of influential wisdom books outside of scripture , like The Wisdom of Sirach , also known simply as Sirach , and The Wisdom of Solomon . These books were well known in James ' day , and there are striking parallels to both in his letter .
# As just one example , in chapter 1 verse 26 from Sirach , we read , And James chapter 1 verse 5 tells us ,
# In addition to these types of wisdom literature , much of Jesus ' instruction recorded in the Gospels is characteristic of wisdom teaching in Israel , and interpreters have noted a number of similarities between James ' writing and Jesus ' instruction .
# Consider , for instance , Matthew chapter 5 verse 10 , where Jesus said , blessed are those who are persecuted because of righteousness , for theirs is the kingdom of heaven .
# Compare this with James chapter 1 verse 12 , where James wrote , Blessed is the man who preserves under trial , because when he has stood the test , he will receive the crown of life that God has promised to those who love him . The wisdom literature of Judaism in the first century
# a little bit before then , had considerable influence on James , especially in terms of the cultural and literary milieu that he was working with . In fact , there are dozens of allusions and parallels between James and other literature , both in the Old Testament and in other Jewish literature .
# You know that James quotes from Proverbs twice , at least once and probably twice , and he has many allusions , particularly to the wisdom of Jesus bin Sirach , a work that was written in about a century before the time of the New Testament . But there is one thing that is unique to James in terms of wisdom , and that is
# He links his wisdom very closely with the teaching of Jesus .
# James is probably one of the most colorful illustrators in the New Testament with depictions of ships being guided by little rudders and farmers that are patiently waiting and merchants that are traveling . There 's many , many illustrations . That 's all wisdom influence , but the content of James
# is really carrying forward the way in which Jesus presents the kingdom and the way the presence of the kingdom changes your life . Because of James ' close ties to wisdom literature , the structure of the eclipse is quite different from what we might expect . Even a brief look at this letter tells us that its organization is not simple .
# In fact , from our modern point of view , it can seem quite disorganized . Much like the book of Proverbs , the book of James deals with a variety of important themes , and it often spends only a few verses on one theme before moving on to another . Occasionally , it returns to one or more of its themes later in the letter , but not with any consistency .
# Some commentators have even concluded that there is no structure to James . They 've suggested that it 's only a collection of wisdom sayings with no real order or flow of thought . But we have to be careful here . This letter is not just a chaotic jumble of unrelated verses thrown together without any order at all .
# Although the book of James resembles wisdom literature in both form and content , it also differs from that genre in a variety of ways . Unlike other wisdom literature , James is a letter written to specific churches . And for this reason , it does reflect some of the organizational features of other New Testament exploits .
# There 's little agreement among interpreters on the organization or structure of James . But for the purposes of this lesson , we 've divided the book into seven sections . The eclipse opens with James ' greeting in James chapter 1 verse 1 .
# The first major division is an introduction to the main themes of the book that we might call wisdom and joy in James chapter 1 verses 2 through 18 . The second major division expresses James ' concern for wisdom and obedience in James chapter 1 verse 19 through chapter 2 verse 26 .
# The third major division deals with wisdom and peace in the Christian community in James 3 -- 1 -- 4 - 12 . The fourth major division focuses on wisdom and the future in James 4 - 13 - 5 - 12 .
# The fifth and final major division is devoted to what we may describe as wisdom and prayer in James chapter 5 verses 13 through 18 . After these five major divisions , there is a concluding authorization in chapter 5 verses 19 and 20 . Let 's take a closer look at each of these divisions , beginning with the greeting in James chapter 1 verse 1 .
# Listen again to chapter 1 verse 1 , James short situation . We should not miss how James described himself here . He called himself a servant of God and of the Lord Jesus Christ .
# James could have introduced himself as the leader of the church , or even as the brother of Jesus . Instead , he chose to make the point that he was the servant of God and Christ . This dual reference may be James ' personal statement of humility , a theme he touches on later in the book .
# Here , he exemplified that humility by making it clear that he was the servant of his brother Jesus . Following the greeting , the first major division centers on what we 've called wisdom and joy .
# James wrote his letter to Christians who 'd been driven out of Jerusalem and were scattered around the Mediterranean world . They were facing different kinds of trials that no doubt discouraged them . And for this reason , James ' first words about the importance of wisdom began with a call to joy .
# Listen again to James chapter 1 verse 2 , where James told his audience , This passage may seem odd to us , especially because it addresses people who were facing trials of many kinds .
# but James ' appeal to consider trials pure joy is not as unusual as we might think . The phrase pure joy comes from the Greek expression , passion karan , that may be translated complete unmitigated joy . This kind of encouragement fits well with other wisdom literature of James Day .
# Many times , wisdom writings encouraged those who suffered to consider themselves blessed . Jesus , for instance , closed the Beatitudes in Matthew chapter 5 verse 12 with the call to rejoice and to be glad in the face of persecution .
# As we said earlier , in chapter 1 verses 3 and 4 , James taught that perseverance through trials makes it possible for believers to be mature and complete . In other words , when God 's people endure hardship , they grow into the fullness of all that God intends for them .
# But in reality , it 's often difficult for even the most sincere believer to see how this is true in the midst of suffering . This is why , in the very next verse , James told his readers to pursue wisdom from God .
# You 'll recall that James chapter one verse five days , if any of you lacks wisdom , you should ask God who gives generously to all .
# Those who want to have pure joy as they suffer trials must ask God for insight . They need wisdom to help them understand how their trials lead to their betterment . And if we ask for this kind of wisdom from God , He will give it to us . As James went on to say in chapter 1 verse 17 , God gives good and perfect gifts to His people .
# James closed this section in chapter 1 verse 18 with this reassurance .
# When we receive the wisdom to understand how God works through trials , we can be joyful . Wisdom strengthens our confidence that God has ordained for us the blessing of eternal salvation . After his discussion on wisdom and joy , James moved to the relationship between wisdom and obedience .
# In this section , James discussed wisdom and obedience in three basic steps . To begin with , chapter 1 verses 19 through 27 introduces the importance of taking action rather than just listening or talking . In chapter 1 verse 22 , we read this .
# Do not merely listen to the word and so deceive yourselves . Do what it says . To hear the word is simply not good enough . The word of wisdom from God must also lead to faithful obedience . Otherwise , we are deceiving ourselves .
# When you read James 's letter you understand that he 's really emphasizing the need to put into practice the things that we say we believe . It 's a very prominent theme throughout the whole eclipse . You ask the question why is James emphasizing that and the first answer seems to be James lives in the real world .
# He ministers to real people and the world in which we live is a world where talk is cheap , where it 's very easy to say we believe in God and much harder to follow through on what that belief might look like in action .
# This seems to have been a challenge not just for James but also for Jesus . Talking is not the same as doing . Jesus knows that . James knows that . We 're trying to reach real people in the real world with a real problem . James expected his readers to do more than just hear God 's word . He expected them to put their faith into action .
# This theme was so important to James that , although he mainly discussed it in chapters 1 and 2 , he returned to it periodically throughout his website . For instance , in chapter 3 verse 13 , James ' basic perspective on the relationship between wisdom and obedience appears again .
# James wrote , Who is wise and understanding among you ? Let them show it by their good life , by deeds done in the humility that comes from wisdom . As this verse indicates , wisdom and understanding of God 's purposes in trials and suffering is no mere intellectual matter .
# Those who have it will show it by their good life , by deeds done in humility that comes from the wisdom that God gives . So , in chapter 1 verse 27 , James closed this section on the need for action by summing up true piety or religion in this way .
# Religion that God our Father accepts as pure and faultless is this , to look after orphans and widows in their distress and to keep oneself from being polluted by the world . James speaks very frankly about religion that what he calls pure and faultless being this , to look after orphans and widows in their distress
# and to keep oneself from being polluted by the world . And in our culture , which is so materialistic in many ways , those are two sides of the same coin , that one of the ways in which we get polluted by the world is not caring for the poor around us , or attributing their poverty to something only within them .
# and not looking at the systemic causes of it . Or looking at ourselves who have means as meaning that that means that somehow we 're superior or we have God 's blessing and poor people do not . When the reality is that oftentimes what you find is the faith of the poor is stronger and more authentic than folks
# who have not suffered the same things that they have . Following this introductory call to action , James elaborated on the connection between wisdom and obedience by focusing on the problem of favoritism in chapter 2 verses 1 through 13 .
# Some people within James ' audience had apparently been showing preference to the wealthy and neglecting the poor . And in this section , James addressed this problem by calling them to give proper attention to what he called the Royal Law .
# In chapter 2 verse 8 , James said , Essentially , neglecting the poor in favor of the rich is a failure to love your neighbor . And James taught that they must avoid the sign of favoritism by keeping the royal law .
# We see in James ' teaching about the rich and their relationship to the poor a real reflection of the Savior 's teaching in Luke chapter 16 . In chapter 2 of James he talks about how do not you know that God has chosen the poor , those who love him , to be heirs of his kingdom , the rich
# are being shown partially as they come into the Christian meetings , they 're being showing deference , you can take my seat , you can have the best seat in the assembly . And James warns those who are acting that way to remember that the poor have full standing in the kingdom of God , full inheritance rights and therefore they should be shown dignity and respect .
# and full membership among the people of God as well . As we 've seen , the book of James has a very positive focus on the law of God . In James ' view , the law teaches us to care for one another , to have compassion on the poor , to avoid favoritism and the like . But this positive outlook can be misused if we are not careful .
# Modern Christians often point out how the law of God has been used in vain as a way to try and justify ourselves before God by our own righteous deeds . And we 're right to reject this abuse of God 's law . But by contrast , the book of James stresses a different facet of the law .
# James taught that although no one can be justified by the law , the law of God is our source of wisdom , and we should live in obedience to it . Of course , we do not obey the law as if we still lived in Old Testament times . We must always apply God 's law in the light of Christ and the teachings of the New Testament .
# But those who 've trusted Christ for salvation obey the law out of gratitude to God because it 's the revelation of God 's wisdom .
# In this sense , James echoes Palm 19 verses 7 and 8 , where we read this ,
# After introducing the importance of action in response to the word of wisdom and resisting favoritism by obeying the royal law of God , James addressed the relationship between faith and obedience in chapter 2 verses 14 through 26 .
# In chapter 2 verse 14 , James posed this question ,
# James answered this question with a resounding no . He did this in a number of ways . First , he pointed out that even the devil believes true things about God , but it does him no good . Then he noted how Abraham 's faith led to obedience , and he described how Rehab demonstrated her faith through good works .
# So , in chapter 2 verse 26 , James drew this well -- known conclusion . According to James , having the right beliefs is not enough . A faith that does not show itself in obedience is dead . It is not true saving faith .
# After exhorting his audience to live a life of obedience , James focused his attention on the relationship between wisdom and peace among followers of Christ . Listen to James ' question in chapter 4 verse 1 . What causes fights and quarrels among you ?
# Although this verse comes in the middle of this section , in a variety of ways , the entire section deals with this question . In this section , James noted three main issues associated with wisdom and peace among believers . First , in chapter 3 verses 1 through 12 , James focused on the tongue , or our use of words .
# In chapter 3 verses 4 and 5 , James compared the tongue to a ship 's rudder . He explained it this way . Ships are so large and are driven by strong winds , but they are steered by a very small rudder . Likewise , the tongue is a small part of the body , but it makes great boasts .
# Then , in verse 6 , he went further , telling the audience , James ' warning against the tongue 's capacity for evil is very similar to what we find in the book of Proverbs .
# Proverbs also deals with the dangers associated with the tongue , or speech , a number of times . We find this in places like Proverbs 10 verse 31 , chapter 11 verse 12 , chapter 15 verse 4 , and many other verses . Both James and Petrobras pointed out that words can lead to all kinds of trouble among God 's people .
# To avoid conflict and live in peace , we must control . instructions for the church how we are to live in the light of Christ 's coming in anticipation of his future return . One of the ways that James gives us to measure our hearts is focusing on our words .
# In other words , James views the words of a person , the tongue , which is shorthand for the words , as a barometer of a person 's whole moral being . It gives the temperature of one 's heart . And so , just as Jesus says , out of the overflow of the heart , the mouth speaks , when James says that a man must bridle his tongue ,
# And it should not be that from the same mouth come blessing and curses . He 's telling us that our heart must be fully committed to God . We must not be a double minded man but we must in faith hold fast to the teaching of Christ . And as we do that our words should bless our brothers and sisters instead of cursing them .
# The second issue tied to wisdom and peace involves two kinds of wisdom . We find this in chapter 3 verses 13 through 18 . In James chapter 3 verses 14 through 17 , we read these words .
# If you harbor bitter envy and selfish ambition in your hearts , such wisdom does not come down from heaven but is earthly , inspirational , demonic . But the wisdom that comes from heaven is first of all pure , then peace -- loving , considerate , submissive , full of mercy and good fruit , impartial and sincere .
# As we see here , to explain the relationship between wisdom and peace , James distinguished between earthly , even demonic wisdom , and wisdom that comes from heaven . Earthly wisdom leads to bitter envy and selfish ambition , but wisdom from God brings peace in the Christian community .
# James called for his readers to let go of their fights and quarrels . He explained that when we cling to our own selfish desires there can be no peace among us . Worldly wisdom , he taught , only leads to disorder and every evil practice .
# So James instructed his readers to rely on the wisdom that comes from God . When we do this , we find peace . As James put it in chapter 3 , verse 18 , peacemakers who saw in peace raise a harvest of righteousness .
# The third issue in this section , in chapter 4 verses 1 through 12 , looks at wisdom and peace in relationship to the inward conflict that followers of Christ experience . James traced strife among Christians to selfish desires , wrong motives , and discontent within us . From James ' point of view ,
# The evil desires within his audience had caused great damage in the Christian community . They were ruled by their wants . And because of this , they were fighting , and covering , and even destroying each other . So , James sternly told them what they must do to bring peace .
# In chapter 4 verses 7 through 10 , James said , Only humble submission to God would put an end to their fights and quarrels and give them peace with one another .
# Now let 's consider the relationship between wisdom and the future . James ' discussion of wisdom and the future can be divided into three parts . The first part is found in chapter 4 verses 13 through 17 and deals with those who were making plans for the future as if God were not in control .
# These verses indicate that many in James ' audience were attempting to determine their own futures . They focused on accumulating wealth , and they bragged about what they would do and where they would go . In response to this , James reminded them that their lives were fleeting . They could not possibly know what their futures held .
# Listen to chapter 4 verses 15 and 16 , where James told them , Only God controls the future , and those who are wise will acknowledge this .
# In the second part of this section , James gave attention to wisdom and the future in a slightly different way . In chapter 5 verses 1 through 6 , he warned against hoarding wealth because of the future day of judgment .
# James spoke at great length about the treatment of the poor in many places , and he repeatedly condemned the wealthy for taking advantage of those less fortunate . In these verses , James strongly cautioned the rich who had gained wealth at the expense of the poor , and he informed them that they would soon suffer for it .
# As he put it in chapter 5 verse 3 , your gold and silver are corroded . Their corrosion will testify against you and eat your flesh like fire . You have hoarded wealth in the last days . As this passage indicates , accumulating wealth at the expense of others will bring severe judgment .
# What James basically says is something that would have been mined -- blowing to many of the Jews who heard him . He basically reverses the understanding that many in Israel had about the relationship of rich and poor . And he actually calls the poor blessed and speaks about , he warns the rich .
# to actually be ready to repent and to expect judgment . The basis for that judgment is these people are hoarding their wealth , which basically , if you 've been blessed with wealth , God 's will is that you would share this with your neighbor , use it to bless your neighbor . But they 're hoarding it up for themselves . They 're defrauding their workers by not paying them a fair wage .
# Wealth is a gift of God that is then to be used as God wills , not for yourself , but ultimately for your neighbor . In other words , every business should be guided by the principle , love your neighbor as yourself . The third part of James discussion on wisdom in the future in chapter 5 verses 7 through 12 turns to waiting patiently for God 's plan for the future to unfold .
# James had criticized those who 'd made plans without relying on God for wisdom . And he 'd warned those who ignored God 's wisdom by hoarding wealth and abusing the poor that they would see God 's judgment . But following this , James encouraged those who were suffering to wait patiently for God to bring the communication of history to pass .
# Listen to chapter 5 verses 7 and 8 where James used this analogy . Be patient then , brothers and sisters , until the Lord 's coming . See how the farmer waits for the land to yield its valuable crop , patiently waiting for the autumn and spring rains . You to be patient and stand firm , because the Lord 's coming is near .
# As we 've just pointed out , James ' words in this section did more than just admonish the wealthy . They also encouraged the poor and oppressed . James ' strong rebuke reminded his audience that the Day of Judgment was coming . And at that time , those who had faithfully depended on God would be rewarded .
# In this way , he encouraged the faithful to continue on the path of gold wisdom , living out their profession of faith , obedient to God in the light of the grand finale of God 's plan for the future .
# After explaining to his readers how wisdom is related to joy , to obedience , to peace , and to the future , the book of James closes with a short practical application of wisdom and prayer . James ' audience was dealing with a number of issues .
# They 'd been scattered from their homes . The rich were oppressing the poor . They were arguing and hurting one another . Many , it seems , were being ruled by their selfish desires , and they were finding it difficult to live in ways that matched their profession of faith . So , in this last section , James taught them what to do in the Christian community as they faced these struggles .
# similar to what he taught at the beginning of the eclipse . Here , James instructed them to devote themselves to prayer . In times of trouble or joy , when dealing with sickness , even sickness caused by the individual sin , those who have wisdom will pray .
# Listen to chapter 5 verses 13 and 14 , where James told his readers , Is anyone of you in trouble ? He should pray . Is anyone happy ? Let him sing songs of praise . Is anyone of you sick ? He should call the elders of the church to pray over him .
# Clearly , James expected his readers to draw near to God for wisdom in every situation . The reason for this is clear enough in verse 16 , where James said ,
# After finishing the main body of his eclipse with his call to patience and prayers and trails , James ended the letter with an exhilaration . In chapter 5 verses 19 and 20 , James urged his audience to watch out for each other and bring back those who had wandered away from the truth .
# He reminded them that , as brothers and sisters in the community of faith , they have the obligation and privilege to lead people back to a faith that truly saves .
# In this introduction to James , we 've looked at the background of the book and noted the author , the audience , and the occasion of writing . We 've also explored the letter structure and content and seen how this book serves as the New Testament book of wisdom for believers facing the discouragement of trials through joy , obedience , peace , the future , and prayer .
# The book of James challenged first century Christians to seek God for wisdom so that they could have joy as they endured trials . Of course , you and I live in very different circumstances than the original audience of James , but we also do face trials and we also need wisdom from God to help us deal with those trials .
# Just like James ' first audience , we need the pure joy that God 's wisdom brings . Although in this lesson we 've only touched on what this book offers , one thing should be clear . The Epistle of James charts a path for wise living in every age .
# And the more we apply this book to our own lives , the more we 'll receive the blessing of pure joy that God offers His people , no matter what trials or difficulties we may face .
# """

#   for index, text in enumerate(texts.split("\n")):
#     print(index)
#     # if text:
#     #   result = llm.predict(text, "en", "vi")
#       # print("source::\n", text)
#       # print(f"target::\n {result}\n\n")
  

