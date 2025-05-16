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
from typing import List
import concurrent.futures
from requests.exceptions import RequestException
import threading
# from langchain import ConversationChain, LLMChain, PromptTemplate
# from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

load_dotenv()

fault_words = [
  "im_start",
  "im_end",
  "<skip_think>"
  "<think>",
  "</think>"
]

default_endpoints = [
    #"http://172.27.188.32:8082/v1",
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
  
def is_valid_response(source_text, target_text):
    source_text = str(source_text).replace("-", " ").strip()
    target_text = str(target_text).replace("-", " ").strip()
    source_len = len(source_text)
    target_len = len(target_text)

    print(f"----- length count ----- source: {source_len} - target {target_len} | {target_len/source_len}")

    if source_text == "" or target_text == '' :
        print("-----invalid response ---- : empty source_text or target_text")
        return False
    elif any(word in target_text.lower() for word in fault_words):
        print("-----invalid response ---- : fault_words")
        return False
    elif target_len/source_len > 2 or source_len/target_len > 2:
        print("-----invalid response ---- : length not match::\n", f"source:\n{source_text}\ntarget:\n{target_text}")
        return False

    return True
   
class LLM():
  def __init__(self, systemPrompt = "") -> None:
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
              HumanMessagePromptTemplate.from_template("""Enlish:\n```{source_text}```Vietnamese:\n```{target_text}```"""),
          ]
      )

  def check_endpoint(self, endpoint: str):
      url = f"{endpoint}/models"
      try:
          response = requests.get(url, timeout=self.timeout)
          
          if response.status_code == 200:
              self.available_endpoints.add(endpoint)  # Add to available endpoints
              self.llm_chain[endpoint] = ChatOpenAI(
                          model=self.model,
                          openai_api_key=self.api_key,
                          openai_api_base=endpoint,
                          # max_tokens=2048,
                          temperature=self.temp,
                          top_p= 0.95,
                          frequency_penalty=1.3,
                          stop=["<|im_end|>"],
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
      while True:
          with concurrent.futures.ThreadPoolExecutor() as executor:
              # Check all endpoints concurrently
              future_to_endpoint = {
                  executor.submit(self.check_endpoint, endpoint): endpoint 
                  for endpoint in self.endpoints
              }
              
              for future in concurrent.futures.as_completed(future_to_endpoint):
                  result = future.result()
                  print("Available llm endpoints::\n", result)
          self.interval = 60
          time.sleep(self.interval)
                
  def initLLM(self, endpoints="", model="", api_key="", temp=0.3, k=5):
    print("Initializing LLM::")
    # self.memory = ConversationBufferWindowMemory(memory_key="history", return_messages=True, k=k)
    endpoints = endpoints.split(',')
    self.endpoints = list(set(default_endpoints + endpoints))
    self.endpoints = self.endpoints if len(self.endpoints)>0 else ["https://openrouter.ai/api/v1"]
    self.temp = temp
    self.k = k
    self.model = model if model != "" else "openai/gpt-4o"
    self.api_key = api_key if api_key != "" else os.getenv("OR_API_KEY", "")
    for endpoint in self.endpoints:
      self.check_endpoint(endpoint)
    time.sleep(self.interval)
    self._monitor_thread = threading.Thread(target=self.monitor, daemon=True)
    self._monitor_thread.start()
    return True

        
  def process(self, source_text, target_text, source_lang="en", target_lang="vn"):
    max_attempts = 5
    attempts = 0
    source_language = next((key for key, value in LANGUAGES.items() if value == source_lang), None)
    target_language = next((key for key, value in LANGUAGES.items() if value == target_lang), None)
    llms = [v for k, v in self.llm_chain.items()]

    while attempts < max_attempts:
      try:
        llm = random.choice(llms)
        print('correction inferencing::', source_language, target_language)
        chain = self.prompt | llm
        llm_chain = RunnableWithMessageHistory(
            chain,
            lambda session_id: ChatMessageHistory(),  # Factory for creating history storage
            input_messages_key="target_text",               # Key for input messages
            history_messages_key="history",      # Key for history in the chain
            window_size=self.k                            # This is equivalent to the 'k' parameter - keep last 2 exchanges
        )
        result = llm_chain.invoke({
                  "source_text": source_text,
                  "target_text": target_text,
                  "source_language": source_language,
                  "target_language": target_language,
				}, config={"configurable": {"session_id": "default_session"}})
        if result.content and is_valid_response(target_text, cleanup_text(result.content)) and target_lang in detect(result.content):
            return cleanup_text(result.content)
      except Exception as e:
        print("error::", e)
        result = {"content": ""}
      print(f"re-run {attempts}:")
      attempts += 1
    return target_text

  def translate(self, source_segments, target_segments, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      N_JOBS = len(self.available_endpoints) * 7 if len(self.available_endpoints) else 20
      print("Start LLM Correct:: concurrency =", N_JOBS)
      with joblib.parallel_config(backend="threading", prefer="threads", n_jobs=int(N_JOBS)):
        t2t_results = Parallel(verbose=100)(delayed(self.process)(source_segments[line]['text'], target_segments[line]['text'], source_lang, target_lang) for (line) in tqdm(range(len(target_segments))))
      for index in tqdm(range(len(target_segments))):
        target_segments[index]['text'] = t2t_results[index]
      return target_segments
    
  def predict(self, source_text, target_text, source_lang="en", target_lang="vi"):
      print("start llm_translate::")
      result = self.process(source_text, target_text, source_lang, target_lang)
      return result  
    
if __name__ == '__main__':
  
  # systemPrompt="""Sửa lỗi chính tả từ bản gốc sang bảng mới"""
  # llm = LLM(systemPrompt=systemPrompt)
  # llm.initLLM(
  #   endpoints="https://openrouter.ai/api/v1", ## http://172.27.188.32:8081/v1
  #   model="openai/gpt-4o", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
  #   api_key=os.getenv("OR_API_KEY", ""),
  #   temp=0.3,
  #   k=10
  # )
  
  # systemPrompt="""Think and translate English accurately into clear, natural, appropriate Vietnamese."""
  # llm = LLM(systemPrompt=systemPrompt)
  llm = LLM()
  llm.initLLM(
    endpoints="http://172.27.188.40:8082/v1", ## http://172.27.188.31:8081/v1
    model="trast-ai/trust-translator-0525", ## "trast-ai/trust-translator-llama3-5b4e" "nampdn-ai/vietmistral-bible-translation"
    api_key="EMPTY",
    temp=0.6,
    k=5
  )
  
  source_texts = """
Imagine for a moment growing up with a close sibling or friend . . you play together , you learn together , you reach adulthood together . For most of your life this person has been right by your side and then one day your friend or sibling claims to be God 's chosen one .
Well , for James , the brother of Jesus , this was not just an imaginary scenario . In his younger years , he doubted that Jesus was the Savior . But later in his life , he not only became a follower of Jesus , he became the leader of the church at Jerusalem and wrote the New Testament book that bears his name .
This is the first lesson in our series on the Epistle of James , and we 've entitled it Introduction to James . In this lesson , we 'll touch on a number of introductory issues that will enable us to pursue a faithful interpretation of this portion of the New Testament .
We 'll approach our introduction to James in two ways . First , we 'll explore the background of the book . And second , we 'll examine its structure and content . Let 's begin with the background of the book of James . With any biblical book , it 's important to understand the context surrounding its writing as much as possible .
The various books of the Bible were written in real historical settings by people with particular motivations and concerns . So , studying these kinds of background issues can help us understand the books themselves . When we consider the settings and motivations associated with the book of James , we 're better equipped to understand what the eclipse meant when it was first written .
and we can apply James ' words more effectively to our lives today . To understand the background of James , we 'll consider first the authorship of the book . Then we 'll look at the original audience . And finally , we 'll examine the occasion on which the letter of James was written . Let 's begin with the authorship of the Epistle of James .
Although we know that the Holy Spirit inspired the Scriptures , many books in the Bible , like James , also identify their human authors . And the more we know about biblical authors , then the better prepared we are to understand and interpret what they wrote . So for this reason , we must learn all we can about who wrote the Epistle of James .
To investigate the authorship of James , we 'll consider two subjects . First , we 'll explore the traditional outlook that James , the younger brother of Jesus , wrote the eclipse . Second , we 'll explore the author 's personal history . Let 's start by looking at the traditional outlook on these matters .
The letter opens in James chapter 1 verse 1 with this simple statement , James , a servant of God and of the Lord Jesus Christ to the 12 tribes scattered among the nations . Greetings . As we see here , the letter clearly identifies a man named James as the author . But this greeting does not settle precisely who this man was .
Five different men in the New Testament , including two of Jesus ' twelve disciples , were named James . But only two of these five men would have had enough authority in the early Church to write a letter like this .
The first of these two was James the son of Zebedee and the brother of John . But according to Acts chapter 12 verse 2 , this James was martyred under Howard Agrippa the first around AD 44 . As we 'll see later , there are good reasons for believing that the book of James was written after Herod 's death . So it 's highly unlikely that James the son of Zimbabwe was the author .
The second James was the younger brother of Jesus . He was also the leader of the early church in Jerusalem . This James was by far the more prominent of the two and the one most theologians have attributed this eclipse to through the centuries .
There is a great deal of support for the traditional outlook that Jesus ' brother James wrote this article . But there are also a few objections . Let 's begin with the support for this view .
In the first place , in chapter 1 verse 1 , the writer did not give any credentials beyond saying that he was a servant of God and of the Lord Jesus Christ . He simply assumed that his name alone would be recognized and would carry sufficient authority . And based on this authority , his letter contains one strong command after another .
This opening greeting , then , makes a strong case for Jesus ' brother James because of his status in the early church in Jerusalem .
Well , in the days of the Apostolic Church , the whole question of authority was very significant . Who has the authority to teach and lead this new community of followers of Jesus Christ ?
There were various writings that were circulating , various claims to have authority . And one of the criteria that emerged is very significant , was that of being an eyewitness to the ministry .
of Jesus , those who were eyewitnesses of his ministry , who spent the time with the Lord himself , were considered to have a righteous claim to the authority to teach in the early church .
James , the brother of Jesus , of course , was an eyewitness to his ministry , but more than that had been eyewitness really to the whole of his life . And that did play a significant role in the weight that the teaching of James and the weight that James ' letter was given in the early church .
In the second place , the testimony of the early Church confirms this outlook on the authorship of the book .
The first eclipse of Clement , written around AD 96 , and the Shepherd of Hammers , written around AD 140 , both either refer to or quote from James ' eclipse . And Oregon , who died in AD 254 , quoted the book of James several times in his commentary on the opposite to the Romans .
Origin 's use of James is particularly significant because in Book 4 Chapter 8 , Oregon identified the author of James as the brother of the Lord . We also know that the Church in the East , and later the Church in the West , accepted this letter as the work of Jesus ' brother .
Now , despite this strong support for the traditional outlook that Jesus ' brother James was the author , there have been some objections . Critical interpreters have suggested at least two alternatives . Some interpreters have looked for an unknown James in the early Church .
They say that the person who wrote the letter was indeed named James , but he was not the son of Zebedee or the brother of Jesus . He remains obscure because he was not mentioned in any other writings of the infant church .
However , this theory is unlikely . As we 've already noted , the simplicity of the author 's identification at the beginning of the letter indicates that he was well known . It 's highly doubtful that there would have been nothing else written about him . A second theory offered by critical interpreters is that of pseudonymity .
Pseudonymity refers to the practice of assigning written works to someone other than the actual author . This practice took place among Jews in the first century for a variety of reasons . One prominent reason for pseudonymity was to give weight or authority to a book or letter .
In the case of James ' eclipse , critical interpreters have argued that someone other than James used his name to gain wider acceptance for their letter in the church . Now , according to passages like 2 Thessalonians chapter 2 verse 2 , this practice was scorned in the first century church as deceit . But critical scholars still offer at least three arguments for this objection .
First , they say there is no mention of the author 's relation to Jesus . They say it 's unthinkable that a brother of Jesus would write to the churches and not reveal this familial bond when he identified himself .
But Jude , the author of the eclipse of Jude , was also Jesus ' brother , and he never mentioned his blood ties to Jesus in his letter . So this argument for pseudonymity is weak at best .
Second , some critical scholars assume pseudonymity because the book gives evidence that the author was aware of Hellenistic or Greek culture . And James was a Jew from Palestine . It 's true that the writer of James had some awareness of Greek culture .
For instance , in James chapter 3 verse 6 , he used the phrase , the whole course of one 's life . This phrase was commonly used in Greek philosophy and religion . But at the time James later was written , many well -- educated Jews in Palestine had more than a passing knowledge of Hellenistic philosophy and religion .
In addition , while the Greek of James is more sophisticated than what we find in other portions of the New Testament , it is not by any means the most sophisticated Greek in the New Testament . In fact , the letter is quite similar in style to books such as Testaments of the Twelve characters and other Hellenistic Jewish writings of that time .
A third argument for pseudonymity points to inconsistencies with the theological portrait of James in the books of Acts and Galatians .
This view suggests that some of the ideas expressed in the Epistle of James do not match theological outlooks attributed to James in these other New Testament books . For instance , critical interpreters point to passages like Acts chapter 21 verses 17 through 25 and Galatians chapter 2 verse 12 .
They argue that in these verses , James appears to be a spokesman for a rather conservative Jewish Christian position on the law . But in James chapter 1 verse 25 and James chapter 2 verse 12 , the author seems to take a somewhat lenient view of the law , calling it the law that gives freedom .
But these differences simply are not as great as critical scholars make them out to be . On closer review , the verses cited in Acts and Galatians do not portray an extreme Jewish Christian point of view . And James ' position on the law in Acts and Galatians is , actually , very consistent with the theology of the letter of James .
As we can see , the arguments against James , the brother of Jesus , being the author of this book are weak at best . The arguments in favor of James ' authorship are much more compelling . And because of this , most evangelical scholars rightly affirm that James , the brother of Jesus , was the author of the letter that bears his name .
We 've considered the authorship of James by looking at the traditional outlook . Now let 's look more closely at James ' personal history . Matthew chapter 13 verse 55 identifies James as one of Mary 's sons and one of Jesus ' half -- brothers .
This family connection may account for the many similarities between James ' eclipse and Jesus ' teachings recorded in the Gospels . But Scripture makes it clear that when James and his other brothers were growing up , they did not recognize who their oldest sibling really was . As John chapter 7 verse 5 tells us , even Jesus ' own brothers did not believe in him .
But at some point in his life , James came to have saving faith in Jesus as his Lord . In fact , James rose to such prominence in the early church that Paul called him , in Galatians chapter 2 verse 9 , one of the pillars of the church . In addition , we know that , according to 1 Corinthians chapter 15 verse 7 , Jesus appeared to James after his resurrection .
James ' position of authority is well documented in the New Testament . For instance , he appears three times in the Book of Acts as the leader of the Jerusalem church . And in Acts chapter 15 , we see him as the spokesman for the apostolic council . Even none -- Christians acknowledge James ' importance in the church .
One of the most well -- known accounts of James ' violent death in AD 62 comes from the Jewish historian Josephs . Listen to Antiquities , Book 20 , Chapter 9 , Section 1 , written in AD 93 , where Josephs described the circumstances surrounding James ' death .
Anninus convened the judges of the Sheridan and brought before them the brother of Jesus , the one called Christ , whose name was James , and certain others , and accusing them of having transgressed the law , delivered them up to be stoned .
While growing up , James may not have understood who his older brother really was . But we can see from Josephus ' account and from scripture and other historical records that later in his adult life , James had an unwavering commitment to Jesus as the Christ .
As Eusebius wrote in his Ecclesiastical History , Book 2 , Chapter 23 , quoting the early Christian historian , Ijesipus , James became a true witness , both to Jews and Greeks , that Jesus is the Christ .
Now that we 've considered the background of James ' eclipse by looking at some of the issues surrounding authorship , let 's explore the original audience of this letter . Technologies often spend a great deal of time and energy trying to learn as much as possible about the author of a particular biblical book .
But discovering the identity of the original audience is just as important . If we want to interpret correctly what a biblical writer was saying , it helps us to know who the writer 's original readers were and what they were facing at that particular time in history .
As we saw earlier , in James chapter 1 verse 1 , James identified his readers as the 12 tribes scattered among the nations . This seems to be a reference to Jews who lived outside of Israel . And in James chapter 2 verse 1 , James addressed his audience as believers in our glorious Lord Jesus Christ .
Taken together , these verses indicate that James ' original audience was made up , primarily , of Jewish Christians who lived outside of Palestine . On several occasions in his book , James addressed his audience affectionately as brothers . But how did James , living in Jerusalem , know his audience well enough to speak to them in this way ?
Well , in Acts chapter 8 verses 1 through 4 , we learn that in the wave of persecution following Stephen 's martyrdom , members of the Jerusalem church were scattered throughout Judea and Samaria .
It 's possible then that James , as the leader of the Jerusalem church , was writing to these scattered members of the 12 tribes . But even if the eclipse was not addressed specifically to these believers , it seems that James ' audience was made up of Jewish Christians in similar circumstances . The vocabulary James used also supports the idea that his original readers were Jewish followers of Jesus .
For example , in chapter two , verse two , James chose the word synagogue , or synagogue , to describe his audience 's meetings .
This was a typical way to refer to Jewish gatherings . And in chapter 5 verse 4 , James used the phrase , Lord Almighty , or Kurias Sabbath . This phrase comes from a common Old Testament name for the God of Israel , Yahweh Sabbath . Language of this kind makes much more sense if the recipients have strong Jewish roots .
Knowing the background to James ' audience is extremely important because it helps us set a trajectory as to how we understand the message that he 's trying to articulate to his audience . James ' audience as a Jewish community are recipients of a long tradition of
the Torah of Moses , the message of the prophets and the writings . James draws on this rich tradition as he talks to them about the life of faith , the wise life , and they need to understand how they should apply it into their own lives in light of the resurrection of Jesus Christ .
Now , when we say that James was writing to Jewish Christians , we do not mean that there were no Gentile believers in the churches James addressed . As early as Acts chapter 8 , we know of an Ethiopian convert .
And as we learn in Acts chapter 10 , there were many Gentile God -- fearing converts to Judaism who attended synagogues . So it would not have been surprising to find at least some Gentile believers in these churches as well . Still , according to Romans chapter 9 verse 8 , Gentile believers were regarded as Abraham 's offspring .
And ideally , they were considered just as much as part of the 12 tribes of Israel as any who were Jews by bloodline . We 've looked at the background of James by considering the Epistle 's authorship and its original audience . Now we 're ready to examine the occasion of its writing .
We 'll explore the occasion of the writing of James in three steps . First , we 'll touch on the location of both the author and audience . Second , we 'll consider the date of composition . Third , we 'll think about the purpose of James ' eclipse . Let 's begin by looking at the location of both the author and the audience of this letter .
The location of the author is not difficult to discern . Both the New Testament and early Church Fathers suggest that James lived his life of ministry in Jerusalem . And he remained in Jerusalem until he was martyred in AD 62 . Because of this , there 's no reason to think that he wrote the respite from any other location .
The location of the original audience is also somewhat straightforward . As we just mentioned , the letter 's recipients were most likely Jewish believers who had been scattered throughout Judea and Samaria after the murder of Stephen .
Acts chapter 11 verse 19 tells us that these displaced believers traveled as far as Phoenicia , Antioch , and Cyprus in search of a safe place to live . We can not be positive that James wrote to believers in these specific locations .
Yet , based on James ' initial greeting to the 12 tribes scattered among the nations , these areas are strong possibilities for the location of James ' original audience .
We really think that these are truly dispersed tribes , that is , the parishioners of Jerusalem who were scattered into Phoenicia and Cyprus and Antioch by the persecution after Stephen 's martyrdom .
that it 's quite possible , in fact I think it likely , that James was writing to these folks as his own parishioners . And the reason I think that is that he surprisingly gives us no theology or virtually none overtly . He does not talk in terms of the structure of the gospel . There are quite a few things that he does not mention .
And as a pastor I 'm thinking well he probably covered those things earlier in his ministry and now he 's speaking to his well -- known audience in the way that a pastor would . And so it has great effect on our sense of James that we look at this audience scattered , this audience already under his ministry .
and see him building in that way . Keeping in mind this first aspect of the occasion of James Epstein , the location of the author and audience , now let 's consider the date of the letter 's composition .
The earliest and latest likely dates for this letter are fairly easy to establish . First , the earliest likely date for the letter 's composition is AD 44 . We know that James wrote his exploits as the leader of the early church in Jerusalem .
Acts chapter 12 verse 17 indicates that James became a significant leader of the Jerusalem church by the time of Peter 's release from prison . According to Acts chapter 12 verses 19 through 23 , Peter was released in the year Herold Agrippa I died in AD 44 . This makes it most likely that the eclipse was not written much before this date .
Second , the latest possible date of composition for the eclipse is AD 62 , the year of James ' martyrdom . As we saw earlier , according to Josephs , James died at the hands of the priest Anninus near this time . This provides a brief window for the letter 's composition .
The letter itself does not include specific references to historical events that would date it more specifically . But there are at least two reasons to think that the date of composition was earlier rather than later .
For one , as we mentioned before , in chapter 2 , verse 2 , James used the word synagogue , or synagogue , to describe his audience 's meetings . The use of synagogue seems to indicate an early stage in the development of the Christian movement .
James may have written before Christians were forced out of the synagogues . Or , at the very least , he wrote at a time when Christians were still calling their gatherings a synagogue . In addition , there 's no mention in James ' eclipse of the Jewish -- Gentile controversies that received so much attention in the writings of Peter and Paul .
In the early church , as Gentiles came to faith in Christ in large numbers , conflicts arose over whether or not these new believers should be required to conform to Jewish customs . Perhaps James simply chose not to deal with these controversies , but more likely , they had not yet become a major factor in the life of the young churches that James addressed .
Having looked at the letter 's occasion both in its location and its date , let 's examine James ' purpose in writing this letter .
One of the most helpful ways to summarize the overarching purpose of James is to look at James chapter 1 verses 2 through 4 . In his opening words , James told his readers , Consider it pure joy , my brothers and sisters , whenever you face trials of many kinds , because you know that the testing of your faith produces perseverance .
Let perseverance finish its work so that you may be mature and complete , not lacking anything . As this passage indicates , James ' audience was facing trials of many kinds . But James called them to have pure joy in their trials .
Trials , he explained , produce perseverance . And those who persevere will become mature and complete , not lacking anything . But the real key to James ' message comes in the very next verse .
In verse 5 , James completed his thoughts with these words , If any of you lacks wisdom , you should ask God , who gives generously to all without finding fault , and it will be given to you .
We 'll discuss these verses in more detail later in the lesson . But for now , this passage gives us a window into the heart of the entire eclipse . To experience pure joy in the midst of trials , ask God for wisdom , and it will be given to you .
With this in mind , we can summarize the main purpose of James ' letter in this way . James called his audience to pursue wisdom from God so that they would have joy in their trials . It was important for James ' audience to hear this message .
As we said earlier , James ' audience was no longer in Palestine . They were living scattered among the nations , far from their homes . No doubt , it was not easy for them to find joy in their trials . This appears to have led some of them to abandon their loyalty to Christ . Instead , they were pursuing what James called friendship with the world .
Listen to James chapter 4 verse 4 where James used these strong words , You adulterous people , do not you know that friendship with the world is hatred toward God ? Anyone who chooses to be a friend of the world becomes an enemy of God .
Clearly , there were some in James ' audience who had strayed far from the faith , and James warned them that being friends with the world made them an enemy of God . It 's no wonder , then , that James exerted his authority as a leader of the church .
Repeatedly , James commanded his readers to live in a manner consistent with a sincere profession of faith . He used more than 50 imperatives or direct commands in his 108 verses .
and he often used other grammatical forms that functioned just like imperatives within their contexts . But James ' principal solution to the problems his audience faced was not merely to command them to do this or that . For him , the heart of the matter was that they needed to pursue wisdom from God . Wisdom from God was the key to receiving joy as they endured their many trials .
Listen to these well -- known words of chapter four , verses eight through 10 , where James told his readers , come near to God and he will come near to you . Humble yourselves before the Lord and he will lift you up .
James directed believers to humble themselves so that God would lift them up . He taught that humility before God is a path to wisdom . And when Christ 's followers draw near to God in humble submission , the wisdom they receive brings joy , even as they persevere through trials .
So far in our introduction to James , we 've looked at the background of James . Now we 're ready to examine the Epistle 's structure and content . We 've just suggested that the book of James focuses a great deal of attention on wisdom as the way to find joy in times of trial .
But this emphasis on wisdom helps us understand something more than just the purpose of this book . Many interpreters have spoken of the book of James as the New Testament book of wisdom . And this perspective also helps us grasp the unusual structure and content of the eclipse .
By the time James wrote his letter , there had been a long history of wisdom literature stemming from the Old Testament . Old Testament wisdom writings include Job and Ecclesiastes , as well as the Book of Petrobras and a number of so -- called wisdom palms and prophetic wisdom sayings .
James ' indebtedness to this Old Testament literature is evident in a number of ways . For instance , in chapter 5 verse 11 , James used the example of Job , the main character in the book of Job , to promote perseverance . Beyond this , James touched on topics such as speech , the treatment of widows and orphans , poverty , and favoritism .
These topics reflect numerous parallels with the content of the Book of Petrobras .
When we read through the eclipse of James , one of the things that we see as a common thread is the word wisdom . He obviously values greatly wisdom , the wisdom from above as opposed to the wisdom from below . That very value in wisdom and the structure of the eclipse makes us think that there 's a great influence in his life on wisdom literature that 's come before him .
Now I think we see that most explicitly in his citation and use of the book of Petrobras and also in the way that he remembers the words of our Lord , of Jesus , who also spoke often in a wisdom context . Alongside that , there was a development of wisdom thought and wisdom writing , a genre really of wisdom writing .
in the intertestamental time . And I think we see some of the same themes through that wisdom literature in James . Occasionally we see the same structure . But I think a lot of the themes also were really started with the book of Petrobras and also with Jesus .
And so I think that the bigger influence on James is probably going to come out of Jesus in Petrobras but that genre and the importance of proverbial wisdom throughout Second Temple Judaism around the time of Jesus is also very important in James .
The letter of James also reflects the content of influential wisdom books outside of scripture , like The Wisdom of Sirach , also known simply as Sirach , and The Wisdom of Solomon . These books were well known in James ' day , and there are striking parallels to both in his letter .
As just one example , in chapter 1 verse 26 from Sirach , we read , And James chapter 1 verse 5 tells us ,
In addition to these types of wisdom literature , much of Jesus ' instruction recorded in the Gospels is characteristic of wisdom teaching in Israel , and interpreters have noted a number of similarities between James ' writing and Jesus ' instruction .
Consider , for instance , Matthew chapter 5 verse 10 , where Jesus said , blessed are those who are persecuted because of righteousness , for theirs is the kingdom of heaven .
Compare this with James chapter 1 verse 12 , where James wrote , Blessed is the man who preserves under trial , because when he has stood the test , he will receive the crown of life that God has promised to those who love him . The wisdom literature of Judaism in the first century
a little bit before then , had considerable influence on James , especially in terms of the cultural and literary milieu that he was working with . In fact , there are dozens of allusions and parallels between James and other literature , both in the Old Testament and in other Jewish literature .
You know that James quotes from Proverbs twice , at least once and probably twice , and he has many allusions , particularly to the wisdom of Jesus bin Sirach , a work that was written in about a century before the time of the New Testament . But there is one thing that is unique to James in terms of wisdom , and that is
He links his wisdom very closely with the teaching of Jesus .
James is probably one of the most colorful illustrators in the New Testament with depictions of ships being guided by little rudders and farmers that are patiently waiting and merchants that are traveling . There 's many , many illustrations . That 's all wisdom influence , but the content of James
is really carrying forward the way in which Jesus presents the kingdom and the way the presence of the kingdom changes your life . Because of James ' close ties to wisdom literature , the structure of the eclipse is quite different from what we might expect . Even a brief look at this letter tells us that its organization is not simple .
In fact , from our modern point of view , it can seem quite disorganized . Much like the book of Proverbs , the book of James deals with a variety of important themes , and it often spends only a few verses on one theme before moving on to another . Occasionally , it returns to one or more of its themes later in the letter , but not with any consistency .
Some commentators have even concluded that there is no structure to James . They 've suggested that it 's only a collection of wisdom sayings with no real order or flow of thought . But we have to be careful here . This letter is not just a chaotic jumble of unrelated verses thrown together without any order at all .
Although the book of James resembles wisdom literature in both form and content , it also differs from that genre in a variety of ways . Unlike other wisdom literature , James is a letter written to specific churches . And for this reason , it does reflect some of the organizational features of other New Testament exploits .
There 's little agreement among interpreters on the organization or structure of James . But for the purposes of this lesson , we 've divided the book into seven sections . The eclipse opens with James ' greeting in James chapter 1 verse 1 .
The first major division is an introduction to the main themes of the book that we might call wisdom and joy in James chapter 1 verses 2 through 18 . The second major division expresses James ' concern for wisdom and obedience in James chapter 1 verse 19 through chapter 2 verse 26 .
The third major division deals with wisdom and peace in the Christian community in James 3 -- 1 -- 4 - 12 . The fourth major division focuses on wisdom and the future in James 4 - 13 - 5 - 12 .
The fifth and final major division is devoted to what we may describe as wisdom and prayer in James chapter 5 verses 13 through 18 . After these five major divisions , there is a concluding authorization in chapter 5 verses 19 and 20 . Let 's take a closer look at each of these divisions , beginning with the greeting in James chapter 1 verse 1 .
Listen again to chapter 1 verse 1 , James short situation . We should not miss how James described himself here . He called himself a servant of God and of the Lord Jesus Christ .
James could have introduced himself as the leader of the church , or even as the brother of Jesus . Instead , he chose to make the point that he was the servant of God and Christ . This dual reference may be James ' personal statement of humility , a theme he touches on later in the book .
Here , he exemplified that humility by making it clear that he was the servant of his brother Jesus . Following the greeting , the first major division centers on what we 've called wisdom and joy .
James wrote his letter to Christians who 'd been driven out of Jerusalem and were scattered around the Mediterranean world . They were facing different kinds of trials that no doubt discouraged them . And for this reason , James ' first words about the importance of wisdom began with a call to joy .
Listen again to James chapter 1 verse 2 , where James told his audience , This passage may seem odd to us , especially because it addresses people who were facing trials of many kinds .
but James ' appeal to consider trials pure joy is not as unusual as we might think . The phrase pure joy comes from the Greek expression , passion karan , that may be translated complete unmitigated joy . This kind of encouragement fits well with other wisdom literature of James Day .
Many times , wisdom writings encouraged those who suffered to consider themselves blessed . Jesus , for instance , closed the Beatitudes in Matthew chapter 5 verse 12 with the call to rejoice and to be glad in the face of persecution .
As we said earlier , in chapter 1 verses 3 and 4 , James taught that perseverance through trials makes it possible for believers to be mature and complete . In other words , when God 's people endure hardship , they grow into the fullness of all that God intends for them .
But in reality , it 's often difficult for even the most sincere believer to see how this is true in the midst of suffering . This is why , in the very next verse , James told his readers to pursue wisdom from God .
You 'll recall that James chapter one verse five days , if any of you lacks wisdom , you should ask God who gives generously to all .
Those who want to have pure joy as they suffer trials must ask God for insight . They need wisdom to help them understand how their trials lead to their betterment . And if we ask for this kind of wisdom from God , He will give it to us . As James went on to say in chapter 1 verse 17 , God gives good and perfect gifts to His people .
James closed this section in chapter 1 verse 18 with this reassurance .
When we receive the wisdom to understand how God works through trials , we can be joyful . Wisdom strengthens our confidence that God has ordained for us the blessing of eternal salvation . After his discussion on wisdom and joy , James moved to the relationship between wisdom and obedience .
In this section , James discussed wisdom and obedience in three basic steps . To begin with , chapter 1 verses 19 through 27 introduces the importance of taking action rather than just listening or talking . In chapter 1 verse 22 , we read this .
Do not merely listen to the word and so deceive yourselves . Do what it says . To hear the word is simply not good enough . The word of wisdom from God must also lead to faithful obedience . Otherwise , we are deceiving ourselves .
When you read James 's letter you understand that he 's really emphasizing the need to put into practice the things that we say we believe . It 's a very prominent theme throughout the whole eclipse . You ask the question why is James emphasizing that and the first answer seems to be James lives in the real world .
He ministers to real people and the world in which we live is a world where talk is cheap , where it 's very easy to say we believe in God and much harder to follow through on what that belief might look like in action .
This seems to have been a challenge not just for James but also for Jesus . Talking is not the same as doing . Jesus knows that . James knows that . We 're trying to reach real people in the real world with a real problem . James expected his readers to do more than just hear God 's word . He expected them to put their faith into action .
This theme was so important to James that , although he mainly discussed it in chapters 1 and 2 , he returned to it periodically throughout his website . For instance , in chapter 3 verse 13 , James ' basic perspective on the relationship between wisdom and obedience appears again .
James wrote , Who is wise and understanding among you ? Let them show it by their good life , by deeds done in the humility that comes from wisdom . As this verse indicates , wisdom and understanding of God 's purposes in trials and suffering is no mere intellectual matter .
Those who have it will show it by their good life , by deeds done in humility that comes from the wisdom that God gives . So , in chapter 1 verse 27 , James closed this section on the need for action by summing up true piety or religion in this way .
Religion that God our Father accepts as pure and faultless is this , to look after orphans and widows in their distress and to keep oneself from being polluted by the world . James speaks very frankly about religion that what he calls pure and faultless being this , to look after orphans and widows in their distress
and to keep oneself from being polluted by the world . And in our culture , which is so materialistic in many ways , those are two sides of the same coin , that one of the ways in which we get polluted by the world is not caring for the poor around us , or attributing their poverty to something only within them .
and not looking at the systemic causes of it . Or looking at ourselves who have means as meaning that that means that somehow we 're superior or we have God 's blessing and poor people do not . When the reality is that oftentimes what you find is the faith of the poor is stronger and more authentic than folks
who have not suffered the same things that they have . Following this introductory call to action , James elaborated on the connection between wisdom and obedience by focusing on the problem of favoritism in chapter 2 verses 1 through 13 .
Some people within James ' audience had apparently been showing preference to the wealthy and neglecting the poor . And in this section , James addressed this problem by calling them to give proper attention to what he called the Royal Law .
In chapter 2 verse 8 , James said , Essentially , neglecting the poor in favor of the rich is a failure to love your neighbor . And James taught that they must avoid the sign of favoritism by keeping the royal law .
We see in James ' teaching about the rich and their relationship to the poor a real reflection of the Savior 's teaching in Luke chapter 16 . In chapter 2 of James he talks about how do not you know that God has chosen the poor , those who love him , to be heirs of his kingdom , the rich
are being shown partially as they come into the Christian meetings , they 're being showing deference , you can take my seat , you can have the best seat in the assembly . And James warns those who are acting that way to remember that the poor have full standing in the kingdom of God , full inheritance rights and therefore they should be shown dignity and respect .
and full membership among the people of God as well . As we 've seen , the book of James has a very positive focus on the law of God . In James ' view , the law teaches us to care for one another , to have compassion on the poor , to avoid favoritism and the like . But this positive outlook can be misused if we are not careful .
Modern Christians often point out how the law of God has been used in vain as a way to try and justify ourselves before God by our own righteous deeds . And we 're right to reject this abuse of God 's law . But by contrast , the book of James stresses a different facet of the law .
James taught that although no one can be justified by the law , the law of God is our source of wisdom , and we should live in obedience to it . Of course , we do not obey the law as if we still lived in Old Testament times . We must always apply God 's law in the light of Christ and the teachings of the New Testament .
But those who 've trusted Christ for salvation obey the law out of gratitude to God because it 's the revelation of God 's wisdom .
In this sense , James echoes Palm 19 verses 7 and 8 , where we read this ,
After introducing the importance of action in response to the word of wisdom and resisting favoritism by obeying the royal law of God , James addressed the relationship between faith and obedience in chapter 2 verses 14 through 26 .
In chapter 2 verse 14 , James posed this question ,
James answered this question with a resounding no . He did this in a number of ways . First , he pointed out that even the devil believes true things about God , but it does him no good . Then he noted how Abraham 's faith led to obedience , and he described how Rehab demonstrated her faith through good works .
So , in chapter 2 verse 26 , James drew this well -- known conclusion . According to James , having the right beliefs is not enough . A faith that does not show itself in obedience is dead . It is not true saving faith .
After exhorting his audience to live a life of obedience , James focused his attention on the relationship between wisdom and peace among followers of Christ . Listen to James ' question in chapter 4 verse 1 . What causes fights and quarrels among you ?
Although this verse comes in the middle of this section , in a variety of ways , the entire section deals with this question . In this section , James noted three main issues associated with wisdom and peace among believers . First , in chapter 3 verses 1 through 12 , James focused on the tongue , or our use of words .
In chapter 3 verses 4 and 5 , James compared the tongue to a ship 's rudder . He explained it this way . Ships are so large and are driven by strong winds , but they are steered by a very small rudder . Likewise , the tongue is a small part of the body , but it makes great boasts .
Then , in verse 6 , he went further , telling the audience , James ' warning against the tongue 's capacity for evil is very similar to what we find in the book of Proverbs .
Proverbs also deals with the dangers associated with the tongue , or speech , a number of times . We find this in places like Proverbs 10 verse 31 , chapter 11 verse 12 , chapter 15 verse 4 , and many other verses . Both James and Petrobras pointed out that words can lead to all kinds of trouble among God 's people .
To avoid conflict and live in peace , we must control . instructions for the church how we are to live in the light of Christ 's coming in anticipation of his future return . One of the ways that James gives us to measure our hearts is focusing on our words .
In other words , James views the words of a person , the tongue , which is shorthand for the words , as a barometer of a person 's whole moral being . It gives the temperature of one 's heart . And so , just as Jesus says , out of the overflow of the heart , the mouth speaks , when James says that a man must bridle his tongue ,
And it should not be that from the same mouth come blessing and curses . He 's telling us that our heart must be fully committed to God . We must not be a double minded man but we must in faith hold fast to the teaching of Christ . And as we do that our words should bless our brothers and sisters instead of cursing them .
The second issue tied to wisdom and peace involves two kinds of wisdom . We find this in chapter 3 verses 13 through 18 . In James chapter 3 verses 14 through 17 , we read these words .
If you harbor bitter envy and selfish ambition in your hearts , such wisdom does not come down from heaven but is earthly , inspirational , demonic . But the wisdom that comes from heaven is first of all pure , then peace -- loving , considerate , submissive , full of mercy and good fruit , impartial and sincere .
As we see here , to explain the relationship between wisdom and peace , James distinguished between earthly , even demonic wisdom , and wisdom that comes from heaven . Earthly wisdom leads to bitter envy and selfish ambition , but wisdom from God brings peace in the Christian community .
James called for his readers to let go of their fights and quarrels . He explained that when we cling to our own selfish desires there can be no peace among us . Worldly wisdom , he taught , only leads to disorder and every evil practice .
So James instructed his readers to rely on the wisdom that comes from God . When we do this , we find peace . As James put it in chapter 3 , verse 18 , peacemakers who saw in peace raise a harvest of righteousness .
The third issue in this section , in chapter 4 verses 1 through 12 , looks at wisdom and peace in relationship to the inward conflict that followers of Christ experience . James traced strife among Christians to selfish desires , wrong motives , and discontent within us . From James ' point of view ,
The evil desires within his audience had caused great damage in the Christian community . They were ruled by their wants . And because of this , they were fighting , and covering , and even destroying each other . So , James sternly told them what they must do to bring peace .
In chapter 4 verses 7 through 10 , James said , Only humble submission to God would put an end to their fights and quarrels and give them peace with one another .
Now let 's consider the relationship between wisdom and the future . James ' discussion of wisdom and the future can be divided into three parts . The first part is found in chapter 4 verses 13 through 17 and deals with those who were making plans for the future as if God were not in control .
These verses indicate that many in James ' audience were attempting to determine their own futures . They focused on accumulating wealth , and they bragged about what they would do and where they would go . In response to this , James reminded them that their lives were fleeting . They could not possibly know what their futures held .
Listen to chapter 4 verses 15 and 16 , where James told them , Only God controls the future , and those who are wise will acknowledge this .
In the second part of this section , James gave attention to wisdom and the future in a slightly different way . In chapter 5 verses 1 through 6 , he warned against hoarding wealth because of the future day of judgment .
James spoke at great length about the treatment of the poor in many places , and he repeatedly condemned the wealthy for taking advantage of those less fortunate . In these verses , James strongly cautioned the rich who had gained wealth at the expense of the poor , and he informed them that they would soon suffer for it .
As he put it in chapter 5 verse 3 , your gold and silver are corroded . Their corrosion will testify against you and eat your flesh like fire . You have hoarded wealth in the last days . As this passage indicates , accumulating wealth at the expense of others will bring severe judgment .
What James basically says is something that would have been mined -- blowing to many of the Jews who heard him . He basically reverses the understanding that many in Israel had about the relationship of rich and poor . And he actually calls the poor blessed and speaks about , he warns the rich .
to actually be ready to repent and to expect judgment . The basis for that judgment is these people are hoarding their wealth , which basically , if you 've been blessed with wealth , God 's will is that you would share this with your neighbor , use it to bless your neighbor . But they 're hoarding it up for themselves . They 're defrauding their workers by not paying them a fair wage .
Wealth is a gift of God that is then to be used as God wills , not for yourself , but ultimately for your neighbor . In other words , every business should be guided by the principle , love your neighbor as yourself . The third part of James discussion on wisdom in the future in chapter 5 verses 7 through 12 turns to waiting patiently for God 's plan for the future to unfold .
James had criticized those who 'd made plans without relying on God for wisdom . And he 'd warned those who ignored God 's wisdom by hoarding wealth and abusing the poor that they would see God 's judgment . But following this , James encouraged those who were suffering to wait patiently for God to bring the communication of history to pass .
Listen to chapter 5 verses 7 and 8 where James used this analogy . Be patient then , brothers and sisters , until the Lord 's coming . See how the farmer waits for the land to yield its valuable crop , patiently waiting for the autumn and spring rains . You to be patient and stand firm , because the Lord 's coming is near .
As we 've just pointed out , James ' words in this section did more than just admonish the wealthy . They also encouraged the poor and oppressed . James ' strong rebuke reminded his audience that the Day of Judgment was coming . And at that time , those who had faithfully depended on God would be rewarded .
In this way , he encouraged the faithful to continue on the path of gold wisdom , living out their profession of faith , obedient to God in the light of the grand finale of God 's plan for the future .
After explaining to his readers how wisdom is related to joy , to obedience , to peace , and to the future , the book of James closes with a short practical application of wisdom and prayer . James ' audience was dealing with a number of issues .
They 'd been scattered from their homes . The rich were oppressing the poor . They were arguing and hurting one another . Many , it seems , were being ruled by their selfish desires , and they were finding it difficult to live in ways that matched their profession of faith . So , in this last section , James taught them what to do in the Christian community as they faced these struggles .
similar to what he taught at the beginning of the eclipse . Here , James instructed them to devote themselves to prayer . In times of trouble or joy , when dealing with sickness , even sickness caused by the individual sin , those who have wisdom will pray .
Listen to chapter 5 verses 13 and 14 , where James told his readers , Is anyone of you in trouble ? He should pray . Is anyone happy ? Let him sing songs of praise . Is anyone of you sick ? He should call the elders of the church to pray over him .
Clearly , James expected his readers to draw near to God for wisdom in every situation . The reason for this is clear enough in verse 16 , where James said ,
After finishing the main body of his eclipse with his call to patience and prayers and trails , James ended the letter with an exhilaration . In chapter 5 verses 19 and 20 , James urged his audience to watch out for each other and bring back those who had wandered away from the truth .
He reminded them that , as brothers and sisters in the community of faith , they have the obligation and privilege to lead people back to a faith that truly saves .
In this introduction to James , we 've looked at the background of the book and noted the author , the audience , and the occasion of writing . We 've also explored the letter structure and content and seen how this book serves as the New Testament book of wisdom for believers facing the discouragement of trials through joy , obedience , peace , the future , and prayer .
The book of James challenged first century Christians to seek God for wisdom so that they could have joy as they endured trials . Of course , you and I live in very different circumstances than the original audience of James , but we also do face trials and we also need wisdom from God to help us deal with those trials .
Just like James ' first audience , we need the pure joy that God 's wisdom brings . Although in this lesson we 've only touched on what this book offers , one thing should be clear . The Epistle of James charts a path for wise living in every age .
And the more we apply this book to our own lives , the more we 'll receive the blessing of pure joy that God offers His people , no matter what trials or difficulties we may face .
"""
  target_texts = """
Hãy tưởng tượng một khoảnh khắc lớn lên cùng một người anh em hoặc bạn thân. Bạn chơi cùng nhau, bạn học cùng nhau, bạn cùng nhau đến tuổi trưởng thành. Trong hầu hết cuộc đời, người này đã ở bên cạnh bạn và rồi một ngày bạn hoặc người anh em của bạn tuyên bố xưng mình là người được Chúa chọn.
Đối với James, em trai của Chúa Giê-xu, đây không chỉ là một kịch bản tưởng tượng. Trong thời niên thiếu, ông nghi ngờ Chúa Giê-xu là Đấng Cứu Rỗi. Nhưng sau này trong cuộc đời, ông không chỉ trở thành người theo Chúa Giê-xu, mà còn trở thành người lãnh đạo hội thánh tại Jerusalem và viết cuốn sách Tân Ước mang tên ông.
Đây là bài học đầu tiên trong loạt bài của chúng tôi về thư Gia-cơ, và chúng tôi đã đặt tên là Giới thiệu về Gia-cơ. Trong bài học này, chúng ta sẽ đề cập đến một số vấn đề giới thiệu sẽ cho phép chúng ta theo đuổi cách giải thích trung tín về phần này của Tân Ước.
Chúng ta sẽ tiếp cận phần giới thiệu về sách Gia-cơ theo hai cách. Đầu tiên, chúng ta sẽ khám phá bối cảnh của sách. Và thứ hai, chúng ta sẽ xem xét cấu trúc và nội dung của nó. Chúng ta hãy bắt đầu với bối cảnh viết sách Gia-cơ. Với bất kỳ sách Kinh Thánh nào, điều quan trọng là phải hiểu bối cảnh viết sách càng nhiều càng tốt.
Các sách khác nhau trong Kinh Thánh được viết trong bối cảnh lịch sử cụ thể bởi những người với động lực và mối quan tâm đặc biệt. Vì vậy, việc nghiên cứu những vấn đề nền tảng này có thể giúp chúng ta hiểu rõ hơn về các sách đó. Khi xem xét bối cảnh và động lực liên quan đến sách Gia-cơ, chúng ta sẽ được chuẩn bị kỹ hơn để hiểu ý nghĩa của thư tín khi nó được viết đầu tiên.
Và chúng ta có thể áp dụng những lời của Gia-cơ một cách hiệu quả hơn vào cuộc sống của chúng ta ngày nay. Để hiểu bối cảnh của Gia-cơ, trước tiên chúng ta sẽ xem xét quyền tác giả của sách. Sau đó, chúng ta sẽ xem xét thính giả nguyên thuỷ. Và cuối cùng, chúng ta sẽ xem xét dịp mà thư của Gia-cơ được viết. Hãy bắt đầu với quyền tác giả của Thư Gia-cơ.
Mặc dù chúng ta biết rằng Đức Thánh Linh đã soi dẫn Kinh Thánh, nhưng nhiều sách trong Kinh Thánh, như Thư Gia-cơ, cũng xác định các tác giả con người của chúng. Và chúng ta càng biết nhiều về các tác giả Kinh Thánh, thì chúng ta càng được chuẩn bị kỹ càng hơn để hiểu và giải thích những gì họ đã viết. Vì vậy, vì lý do này, chúng ta phải tìm hiểu tất cả những gì có thể về người đã viết thư Gia-cơ.
Để điều tra về quyền tác giả của sách Gia-cơ, chúng ta sẽ xem xét hai chủ đề. Đầu tiên, chúng ta sẽ khám phá quan điểm truyền thống rằng Gia-cơ, em trai của Chúa Giê-xu, đã viết thư tín này. Thứ hai, chúng ta sẽ khám phá lịch sử cá nhân của tác giả. Chúng ta hãy bắt đầu bằng cách xem xét quan điểm truyền thống về những vấn đề này.
Bức thư mở đầu trong Gia-cơ chương 1 câu 1 với lời tuyên bố đơn giản này: 'Gia-cơ, tôi tớ Thiên Chúa Hằng Hữu và Chúa Cứu Thế Giê-xu, gửi cho 12 chi phái tản lạc giữa các dân tộc. Gửi các anh em.' Như chúng ta thấy ở đây, bức thư xác định rõ một người tên là Gia-cơ là tác giả. Nhưng lời chào này không giải quyết được chính xác người đàn ông này là ai.
Năm người đàn ông khác nhau trong Tân Ước, bao gồm hai trong mười hai môn đồ của Chúa Giê-xu, được đặt tên là James. Nhưng chỉ có hai trong số năm người này sẽ có đủ thẩm quyền trong Hội Thánh đầu tiên để viết một lá thư như thế này.
Người đầu tiên trong hai người này là Gia-cơ con trai Xê-bê-đê và anh em cùng cha khác mẹ với Giăng. Nhưng theo Công vụ chương 12 câu 2, Gia-cơ này đã tử đạo dưới thời Hoàng hậu Agrippa I vào khoảng năm 44 sau Chúa. Như chúng ta sẽ thấy sau, có những lý do chính đáng để tin rằng Sách Gia-cơ được viết sau cái chết của Hoàng hậu. Vì vậy, rất khó có khả năng Gia-cơ con trai Xê-bê-đê là tác giả.
Gia-cơ thứ hai là em trai của Chúa Giê-xu. Ông cũng là người lãnh đạo hội thánh ban đầu tại Giê-ru-sa-lem. Gia-cơ này rõ ràng là nổi bật hơn trong hai người và là người mà có lẽ các nhà thần học đã quy sự nổi bật này cho suốt nhiều thế kỷ.
Có rất nhiều sự ủng hộ cho quan điểm truyền thống rằng anh em của Chúa Giê-xu đã viết sách này. Nhưng cũng có một vài sự phản đối. Hãy bắt đầu với sự ủng hộ cho quan điểm này.
Trước hết, trong Chương 1 Câu 1, tác giả không đưa ra bất kỳ thẩm quyền nào ngoài việc xác nhận mình là tôi tớ của Thiên Chúa và Chúa Cứu Thế Giê-xu. Ông chỉ đơn giản cho rằng việc tên mình được biết đến sẽ gây ấn tượng đủ uy quyền. Và dựa trên thẩm quyền này, thư tín của ông chứa đựng mỗi mệnh lệnh đều mạnh mẽ.
Lời chào mở đầu thánh này, sau đó, lập luận mạnh mẽ cho Gia-cơ - em trai Chúa Giê-xu, bởi vì địa vị của ông trong Hội Thánh sơ khai tại Giê-ru-sa-lem.
Vâng, trong thời của Giáo hội các sứ đồ, toàn bộ câu hỏi về thẩm quyền rất quan trọng. Ai có thẩm quyền dạy dỗ và dẫn dắt cùng cộng đồng mới của những người theo Chúa Giê-xu Christ?
Có nhiều tác phẩm khác nhau đang lưu hành, nhiều tuyên bố khác nhau về thẩm quyền. Và một trong những tiêu chí xuất hiện rất quan trọng, đó là được chứng kiến chức vụ của Chúa Giê-xu.
của Chúa Giê-xu, những người tận mắt chứng kiến chức vụ của Người, những người đã dành thời gian bên chính Chúa, được xem là có căn cứ chính đáng để dạy dỗ trong Hội Thánh đầu tiên.
Gia-cơ, anh em của Chúa Giê-xu, tất nhiên là nhân chứng tận mắt về chức vụ Ngài, nhưng hơn thế nữa là nhân chứng tận mắt về toàn bộ cuộc đời Ngài. Điều đó đã đóng vai trò quan trọng trong năng lực của lời dạy của Gia-cơ và năng lực mà thư tín của Gia-cơ được ghi nhận trong hội thánh ban đầu.
Thứ hai, lời chứng của Hội Thánh sơ khai xác nhận quan điểm này về quyền tác giả của cuốn sách.
Cuốn sách đầu tiên của Clement, được viết vào khoảng năm 96 sau Công nguyên, và cuốn sách Người Chăn Chiên, được viết vào khoảng năm 140 sau Công nguyên, cả hai đều đề cập hoặc trích dẫn từ sách Gia-cơ. Và Origen, người đã chết vào năm 254 sau Công nguyên, đã trích dẫn sách Gia-cơ nhiều lần trong phần chú giải của ông về sách Rô-ma.
Việc Origen sử dụng thư tín Gia-cơ đặc biệt có ý nghĩa vì trong Sách 4 Chương 8, Origen đã xác định tác giả thư Gia-cơ là anh em của Chúa Cứu Thế. Chúng ta cũng biết rằng Giáo hội ở phương Đông, và sau đó là Giáo hội ở phương Tây, đã chấp nhận bức thư này là tác phẩm của anh em của Chúa Cứu Thế.
Bây giờ, bất chấp sự ủng hộ mạnh mẽ này cho quan điểm truyền thống rằng anh em của Chúa Giê-xu là tác giả, đã có một số phản đối. Các nhà giải kinh phê bình đã đề xuất ít nhất hai lựa chọn. Một số nhà giải kinh đã tìm kiếm một James không ai biết đến trong Hội Thánh đầu tiên.
Họ nói rằng, người viết bức thư thực sự có tên là Gia-cơ, nhưng ông không phải là con trai của Xê-bê-đê hay anh em của Chúa Giê-xu. Ông vẫn còn mờ nhạt vì không được nhắc đến trong bất kỳ văn bản nào khác của Hội Thánh sơ khai.
Tuy nhiên, lý thuyết này không đáng tin. Như chúng ta đã lưu ý, sự đơn giản trong việc xác định tác giả ở đầu thư cho thấy ông đã được biết đến rộng rãi. Rất khó có thể tưởng tượng rằng sẽ không có gì khác được viết về ông. Lý thuyết thứ hai được đưa ra bởi những người theo chủ nghĩa phê bình là thuyết về việc viết lén (pseudonymity).
Giả danh đề cập đến việc gán cho tác phẩm thành văn một tác giả khác ngoài người thực sự viết. Việc này diễn ra giữa người Do Thái thế kỷ thứ nhất vì nhiều lý do. Một lý do quan trọng là để đặt trọng lượng hoặc uy quyền cho một sách hay thư tín.
Trong trường hợp của bức thư của Gia-cơ, những người phê bình đã lập luận rằng có người khác đã sử dụng tên của Gia-cơ để được chấp nhận rộng rãi hơn trong hội thánh. Bây giờ, theo các phân đoạn như 2 Tê-sa-lô-ni-ca chương 2 câu 2, việc này bị coi là sự lừa dối trong hội thánh thế kỷ thứ nhất. Nhưng các học giả phê bình vẫn đưa ra ít nhất ba lập luận để phản đối điều này.
Đầu tiên, họ nói rằng không có đề cập đến mối quan hệ của tác giả với Chúa Giê-xu. Họ nói rằng không thể tưởng tượng được một người anh em của Chúa Giê-xu sẽ viết thư cho các hội thánh mà không bày tỏ mối quan hệ thân tộc này khi ông tự xác định mình.
Nhưng Giu-đe, tác giả sách Giu-đe, cũng là anh em của Chúa Giê-xu, và ông không bao giờ đề cập đến mối quan hệ huyết thống với Chúa Giê-xu trong bức thư của mình. Vì vậy, lập luận này về giả danh là rất yếu.
Thứ hai, một số học giả phê bình cho rằng sự giả danh vì sách này cho thấy tác giả biết văn hóa Hy Lạp. Và Gia-cơ là một người Do Thái từ Pha-lét-tin. Đúng là tác giả thư Gia-cơ có nhận thức nhất định về văn hóa Hy Lạp.
Ví dụ, trong Gia-cơ chương 3 câu 6, sứ đồ Gia-cơ sử dụng cụm từ 'toàn bộ đời sống'. Cụm từ này thường được dùng trong triết học và tôn giáo Hy Lạp. Nhưng vào thời điểm sách Gia-cơ được viết, nhiều người Do Thái có nền giáo dục cao ở Palestine đã có hiểu biết sâu sắc hơn về triết học và tôn giáo Hy Lạp.
Ngoài ra, mặc dù tiếng Hy Lạp của Gia-cơ phức tạp hơn những gì chúng ta tìm thấy trong các phần khác của Tân Ước, nhưng nó không phải là tiếng Hy Lạp phức tạp nhất trong Tân Ước. Thực tế, bức thư khá giống với các sách như Tín điều của Mười Hai Nhân Chứng và các tác phẩm Do Thái Hy Lạp khác của thời đó.
Lập luận thứ ba về tính giả danh chỉ ra sự không nhất quán với bức chân dung thần học của Gia-cơ trong sách Công vụ và Ga-la-ti.
Quan điểm này cho rằng một số ý tưởng được thể hiện trong thư Gia-cơ không phù hợp với quan điểm thần học được gán cho Gia-cơ trong các sách Tân Ước khác. Ví dụ, các nhà chú giải phê bình nhấn mạnh các đoạn như sách Công vụ chương 21 câu 17 đến 25 và sách Ga-la-ti chương 2 câu 12.
Họ cho rằng trong những câu này, Gia-cơ dường như là người phát ngôn cho một lập trường Cơ Đốc giáo Do Thái khá bảo thủ về luật pháp. Nhưng trong Gia-cơ chương 1 câu 25 và Gia-cơ chương 2 câu 12, tác giả dường như có quan điểm khá khoan dung về luật pháp, gọi nó là luật pháp ban cho sự tự do.
Nhưng những khác biệt này đơn giản không lớn như các học giả phê bình thường xuyên trình bày. Khi xem xét kỹ lưỡng, những câu Kinh Thánh được trích dẫn trong sách Công vụ và Ga-la-ti không mô tả một quan điểm Cơ Đốc cực đoan của người Do Thái. Và lập trường của Gia-cơ về luật pháp trong sách Công vụ và Ga-la-ti thực sự rất phù hợp với thần học trong bức thư của Gia-cơ.
Như chúng ta có thể thấy, những lập luận chống lại Giăng, em của Chúa Giê-xu, là tác giả của sách này có vẻ yếu ớt nhất. Những lập luận ủng hộ quyền tác giả của Giăng lại thuyết phục hơn nhiều. Và vì thế, hầu hết các học giả Tin Lành thần học khẳng định một cách đúng đắn rằng Giăng, em của Chúa Giê-xu, là tác giả của bức thư gọi theo tên ông.
Chúng ta đã xem xét quyền tác giả của Gia-cơ bằng cách xem xét quan điểm truyền thống. Giờ hãy xem xét kỹ hơn về lịch sử cá nhân của Gia-cơ. Sách Ma-thi-ơ chương 13 câu 55 xác định Gia-cơ là một trong những người con trai của Ma-ri và là một trong những anh em cùng mẹ khác cha của Chúa Giê-xu.
Mối liên hệ gia đình này có thể giải thích cho nhiều điểm tương đồng giữa Giăng và những lời dạy của Chúa Giê-xu được ghi lại trong các Tin Mừng. Nhưng Kinh Thánh nói rõ rằng khi Giăng và những anh em khác của ông lớn lên, họ không nhận ra anh cả của họ thực sự là ai. Như Giăng chương 7 câu 5 ghi lại, ngay cả các anh em của Chúa Giê-xu cũng không tin Ngài.
Nhưng tại một thời điểm nào đó trong cuộc đời mình, Gia-cơ đã có đức tin cứu rỗi nơi Chúa Giê-xu là Chúa của mình. Trên thực tế, Gia-cơ đã trở nên nổi bật đến nỗi, tức là Phao-lô, trong Ga-la-ti chương 2 câu 9, đã gọi ông là một trong những người đứng đầu của hội thánh. Ngoài ra, chúng ta biết rằng, theo 1 Cô-rinh-tô chương 15 câu 7, Chúa Giê-xu đã hiện ra với Gia-cơ sau khi Ngài phục sinh.
Chức vụ của Gia-cơ được ghi lại đầy đủ trong Tân Ước. Chẳng hạn, ông xuất hiện ba lần trong sách Công vụ như là lãnh đạo hội thánh Giê-ru-sa-lem. Và trong Công vụ chương 15, chúng ta thấy ông là phát ngôn viên của hội đồng các sứ đồ. Ngay cả những người không phải Cơ Đốc nhân cũng công nhận tầm quan trọng của Gia-cơ trong hội thánh.
Một trong những ghi chép nổi tiếng nhất về cái chết bạo lực của Gia-cơ vào năm 62 SCN đến từ sử gia Do Thái Josephs. Hãy lắng nghe sách Antiquities, Sách 20, Chương 9, Chương 1, được viết vào năm 93 SCN, nơi Josephs mô tả hoàn cảnh xung quanh cái chết của Gia-cơ.
Anninus triệu tập các quan tòa ở Sheridan và đưa ra trước họ người anh em của Chúa Giê-xu, vị được gọi là Đấng Christ, vị tên là Gia-cơ, cùng một số người khác, cáo buộc họ vi phạm luật pháp, rồi giao nộp họ cho đám đông ném đá.
Khi lớn lên, James có thể không hiểu anh trai mình thực sự là ai. Nhưng chúng ta có thể thấy từ ghi chép của Josephus, từ Kinh Thánh và các hồ sơ lịch sử khác rằng sau đó trong cuộc sống trưởng thành của mình, James đã có cam kết không lay chuyển với Chúa Giê-xu là Đấng Cứu Thế.
Như Eusebius đã viết trong tác phẩm Sử Giáo Hội của mình, cuốn 2, Chương 23, trích dẫn nhà sử học Cơ Đốc thời kỳ đầu là Ijesipus, James đã trở thành nhân chứng chân chính, cả cho người Do Thái lẫn người Hy Lạp, rằng Chúa Giê-xu là Đấng Christ.
Bây giờ chúng ta đã xem xét bối cảnh của bức thư Gia-cơ bằng cách xem xét một số vấn đề xung quanh quyền tác giả, chúng ta hãy khám phá độc giả ban đầu của bức thư này. Các nhà nghiên cứu thường dành rất nhiều thời gian và năng lượng để tìm hiểu càng nhiều càng tốt về tác giả của một sách Kinh Thánh cụ thể.
Nhưng việc khám phá bản chất của độc giả nguyên thủy cũng quan trọng như vậy. Nếu chúng ta muốn giải thích chính xác những gì một tác giả Kinh Thánh đang nói, điều sẽ giúp chúng ta là biết độc giả nguyên thủy của tác giả là ai và họ đang đối mặt điều gì vào thời điểm cụ thể đó trong lịch sử.
Như chúng ta đã thấy trước đó, trong Gia-cơ chương 1 câu 1, Gia-cơ xác định độc giả của mình là 12 chi phái bị tan lạc giữa các dân tộc. Điều này dường như ám chỉ người Do Thái sống bên ngoài Y-sơ-ra-ên. Và trong Gia-cơ chương 2 câu 1, Gia-cơ nói với độc giả rằng họ là những người tin nhận Chúa Cứu Thế Giê-xu vinh hiển của chúng ta.
Tổng hợp lại, các câu Kinh Thánh này cho thấy độc giả ban đầu của Gia-cơ được tạo thành, chủ yếu, của các Cơ Đốc nhân gốc Do Thái sống bên ngoài Palestine. Trong một số trường hợp trong sách của mình, Gia-cơ đã nhắc nhở các tín hữu của mình một cách trìu mến như anh em. Nhưng làm sao Gia-cơ, sống tại Jerusalem, lại biết rõ các tín hữu của mình đến mức có thể nói với họ theo cách này?
Vâng, trong Công vụ chương 8 câu 1 đến 4, chúng ta biết rằng trong làn sóng bắt bớ sau khi Ê-tiên tử đạo, các thành viên của hội thánh Jerusalem đã phân tán khắp Judea và Samaria.
Vậy thì có thể là sứ đồ Gia-cơ, với tư cách là người lãnh đạo Hội Thánh Giê-ru-sa-lem, đang viết thư cho những tín hữu ở rải rác của 12 chi phái. Nhưng ngay cả khi thư tín không được viết cụ thể cho những tín hữu này, dường như đối tượng đọc của Gia-cơ là các Cơ Đốc nhân Do Thái trong những hoàn cảnh tương tự. Từ vựng Gia-cơ sử dụng cũng ủng hộ ý tưởng rằng độc giả ban đầu của ông là những tín hữu Do Thái theo Chúa Giê-xu.
Ví dụ, trong chương hai câu hai, Gia-cơ đã chọn từ synagogue, hoặc synagogue, để mô tả các cuộc họp của bạn đọc mình.
Đây là cách điển hình để đề cập đến các cuộc tụ họp của người Do Thái. Trong chương 5 câu 4, sách Gia-cơ đã dùng cụm từ 'Chúa toàn năng', hoặc kurias Sabbath. Cụm từ này xuất phát từ danh xưng phổ biến trong Cựu Ước về Thiên Chúa Hằng Hữu Israel, Yahweh Sabbath. Ngôn ngữ loại này có ý nghĩa hơn nhiều nếu những người nhận có gốc Do Thái sâu sắc.
Biết rõ bối cảnh đối tượng nghe của Gia-cơ là cực kỳ quan trọng vì giúp chúng ta hiểu được lộ trình để thấu hiểu thông điệp ông truyền đạt cho họ. Đối tượng ông hướng đến là cộng đồng Do Thái nhận được những truyền thống lâu đời
Luật pháp Môi-se, sứ điệp từ các tiên tri và các tác phẩm. Gia-cơ đã trích dẫn truyền thống phong phú này khi hướng dẫn họ về đời sống đức tin, cuộc sống khôn ngoan, và họ cần hiểu cách áp dụng vào đời sống mình trong bối cảnh sự phục sinh của Chúa Cứu Thế Giê-xu.
Bây giờ, khi chúng ta nói rằng James đang viết cho các Cơ Đốc nhân gốc Do Thái, chúng ta không có nghĩa là không có những tín đồ người ngoại bang trong các hội thánh mà James viết thư. Ngay từ Công vụ chương 8, chúng ta biết về một người cải đạo người Ethiopia.
Và như chúng ta học được trong sách Công vụ chương 10, có nhiều người ngoại bang kính sợ Thiên Chúa Hằng Hữu đã cải đạo sang Do Thái giáo, đến các nhà hội. Vì vậy, không có gì ngạc nhiên khi tìm thấy ít nhất một số tín hữu ngoại bang trong các hội thánh này. Tuy nhiên, theo Rô-ma chương 9 câu 8, các tín hữu ngoại bang được coi là dòng dõi của Áp-ra-ham.
Và lý tưởng nhất, họ được coi là một phần của 12 chi tộc Israel như bất kỳ người Do Thái nào theo dòng tộc. Chúng tôi đã xem xét bối cảnh của James bằng cách xem xét tác giả thư tín và độc giả ban đầu. Giờ đây chúng ta đã sẵn sàng để xem xét nguồn gốc viết thư.
Chúng ta sẽ khám phá bối cảnh viết thư của Gia-cơ theo ba bước. Đầu tiên, chúng ta sẽ đề cập đến vị trí của cả tác giả và độc giả. Thứ hai, chúng ta sẽ xem xét ngày viết. Thứ ba, chúng ta sẽ xem xét mục đích của thư tín này. Hãy bắt đầu bằng cách xem xét vị trí của cả tác giả và độc giả của bức thư này.
Vị trí của tác giả không khó để nhận ra. Cả Tân Ước và các Giáo phụ Hội Thánh đầu tiên đều gợi ý rằng Gia-cơ sống cuộc sống chức vụ tại Jerusalem. Và ông vẫn ở tại Jerusalem cho đến khi ông bị tử đạo vào năm 62 sau Công nguyên. Vì điều này, không có lý do gì để nghĩ rằng ông đã viết Thư Gia-cơ từ bất kỳ địa điểm nào khác.
Vị trí của người đọc ban đầu cũng khá rõ ràng. Như chúng tôi vừa đề cập, những người nhận thư rất có thể là những tín đồ Do Thái đã bị phân tán khắp Judea và Samaria sau vụ giết Stephen.
Công vụ chương 11 câu 19 cho chúng ta biết rằng những tín đồ bị di cư này đã đi xa đến Phoenicia, Antioch và Cyprus để tìm nơi an toàn để sinh sống. Chúng ta không thể khẳng định chắc chắn rằng Gia-cơ đã viết cho các tín đồ ở những địa điểm cụ thể này.
Tuy nhiên, dựa trên lời chào ban đầu của Gia-cơ dành cho 12 chi phái rải rác giữa các quốc gia, đây là những khả năng cao cho người đọc ban đầu của Gia-cơ.
Chúng tôi thực sự nghĩ rằng đây là những bộ lạc thực sự bị tản lạc, tức là giáo dân của Jerusalem, những người đã bị rải rác thành Phoenicia và Cyprus rồi đến Antioch bởi cuộc bách hại sau khi Stephen tử đạo.
rằng có thể, thực tế tôi nghĩ là rất có khả năng, rằng James đang viết cho những người này với tư cách là giáo dân của ông. Lý do tôi nghĩ vậy là vì ông không đề cập thần học nào hoặc hầu như không đề cập công khai. Ông không bàn về cấu trúc của Phúc Âm. Có khá nhiều điều ông không đề cập.
Và với tư cách là một mục sư, tôi nghĩ rằng ông ấy có lẽ đã giảng dạy những điều đó trước đó trong chức vụ, và giờ đây đang nói với độc giả quen thuộc theo cách mà một mục sư thường nói. Điều này tạo nên ấn tượng mạnh mẽ trong nhận thức của chúng ta về thư tín này - khi chúng ta nhìn thấy độc giả này đang rải rác khắp nơi, những người đã được ông tiếp nhận trong chức vụ.
và xem cách ông xây dựng theo cách ấy. Hãy ghi nhớ khía cạnh đầu tiên này về bối cảnh thư Gia-cơ, vị trí của tác giả và độc giả, bây giờ, hãy xem xét ngày viết thư.
Những ngày tháng sớm nhất và muộn nhất có khả năng xác định của bức thư này khá dễ dàng. Đầu tiên, ngày tháng sớm nhất có khả năng xác định khi viết thư là năm 44 sau Chúa Giê-xu. Chúng ta biết rằng Gia-cơ đã viết các thư tín của mình với tư cách là người lãnh đạo hội thánh đầu tiên tại Giê-ru-sa-lem.
Công vụ chương 12 câu 17 cho thấy Gia-cơ đã trở thành một nhà lãnh đạo quan trọng của Hội Thánh Giê-ru-sa-lem vào thời điểm Phi-e-rơ được phóng thích khỏi ngục tù. Theo Công vụ chương 12 câu 19 đến 23, Phi-e-rơ được thả vào năm hoàng đế Ạc-ríp-ba I qua đời (sau Chúa Giê-xu năm 44). Điều này có khả năng cao nhất khiến tác phẩm này không được viết sớm hơn khoảng đó.
Thứ hai, ngày viết gần nhất có thể là năm 62 SCN, năm James tử đạo. Như chúng ta đã thấy trước đó, theo Josephus, James đã chết dưới tay giáo sĩ Annius vào thời điểm này. Điều này cung cấp một khoảng thời gian ngắn gọn cho việc soạn thảo thư tín.
Bản thân thư tín không chứa những tham chiếu cụ thể về các sự kiện lịch sử làm xác định niên đại cụ thể hơn. Nhưng có ít nhất hai lý do để nghĩ rằng niên đại của thư tín có lẽ sớm hơn là muộn hơn.
Một là, như chúng ta đã đề cập trước đó, trong đoạn Chương 2, câu 2, Gia-cơ đã dùng từ 'nhà hội' (synagogue) để mô tả những buổi nhóm của các tín hữu. Việc dùng từ 'nhà hội' dường như chỉ ra giai đoạn đầu trong sự phát triển của phong trào Cơ Đốc.
Gia-cơ có thể đã viết trước khi các Cơ Đốc nhân bị buộc phải rời khỏi hội đường Do Thái. Hoặc, ít nhất là ông đã viết vào thời điểm các Cơ Đốc nhân vẫn còn gọi những cuộc tụ họp của họ là một hội đường Do Thái. Ngoài ra, không có đề cập nào trong Sách Gia-cơ về các cuộc tranh cãi giữa người Do Thái và dân ngoại đã nhận được rất nhiều sự chú ý trong các tác phẩm của Phi-e-rơ và Phao-lô.
Trong Hội Thánh đầu tiên, khi những người ngoại bang đến với đức tin nơi Chúa Cứu Thế với số lượng lớn, xung đột nảy sinh về việc liệu những tín hữu mới này có nên được đòi hỏi phải tuân theo phong tục Do Thái hay không. Có lẽ Gia-cơ chỉ đơn giản chọn cách không giải quyết những tranh cãi này, nhưng có lẽ hơn thế nữa, chúng chưa trở thành một yếu tố chính trong đời sống của các hội thánh non trẻ mà Gia-cơ đang nói đến.
Sau khi xem xét cách thức viết bức thư trong cả vị trí và ngày tháng của nó, hãy xem xét mục đích của tác giả James trong việc viết thư này.
Một trong những cách hữu ích nhất để tóm tắt mục đích bao quát của Gia-cơ là xem xét Gia-cơ chương 1 câu 2 đến 4. Trong phần mở đầu, Gia-cơ đã nói với độc giả: 'Hãy coi đó là niềm vui thánh, hỡi anh chị em, khi anh chị em đối diện với đủ loại thử thách, bởi anh chị em biết rằng sự thử nghiệm đức tin anh chị em sinh ra sự kiên trì.'
Hãy để sự kiên trì hoàn thành công việc của nó để anh em có thể trưởng thành và trọn vẹn, không thiếu thốn điều gì. Như câu Kinh Thánh này cho thấy, thính giả của sứ đồ Gia-cơ đang đối diện với nhiều thử thách. Nhưng sứ đồ Gia-cơ kêu gọi họ có niềm vui thanh sạch trong những thử thách ấy dù biết rằng thử thách là điều không thể tránh khỏi.
Những thử thách, ông giải thích, tạo ra sự kiên trì. Và những người kiên trì sẽ trở nên trưởng thành và trọn vẹn, không thiếu gì. Nhưng chìa khóa then chốt trong thông điệp của anh em Gia-cơ đến trong câu tiếp theo.
Trong câu 5, Gia-cơ đã hoàn thành ý tưởng của mình bằng những lời này: 'Nếu ai trong anh em thiếu sự khôn ngoan, hãy cầu xin Thiên Chúa, Đấng ban cho mọi người cách rộng rãi mà không thấy lỗi lầm, thì sự khôn ngoan sẽ được ban cho anh em.'
Chúng ta sẽ thảo luận chi tiết hơn về những câu này trong bài học sau. Nhưng bây giờ, đoạn văn này cho chúng ta một miếng mành nhìn vào tâm can của toàn bộ sự biến đổi. Để trải nghiệm niềm vui tuyệt đối giữa những thử thách, hãy cầu xin Thiên Chúa ban cho sự khôn ngoan, và nó sẽ được ban cho bạn.
Với điều này trong đầu, chúng ta có thể tóm tắt mục đích chính của bức thư của Gia-cơ theo cách này. Gia-cơ đã kêu gọi người đọc của mình theo đuổi sự khôn ngoan đến từ Thiên Chúa để họ có niềm vui trong những thử thách của mình. Điều quan trọng đối với người đọc của Gia-cơ là nghe được thông điệp này.
Như chúng tôi đã nói trước đó, thính giả của James không còn ở Palestine. Họ đang sống rải rác giữa các dân tộc, xa nhà. Không nghi ngờ gì, họ không dễ dàng tìm thấy niềm vui trong thử thách của mình. Điều này dường như đã khiến một số người từ bỏ lòng trung thành với Chúa Cứu Thế. Thay vào đó, họ đang theo đuổi tình bạn với thế gian.
Hãy nghe Gia-cơ chương 4 câu 4, nơi Gia-cơ dùng những từ ngữ mạnh mẽ: 'Hỡi những kẻ ngoại tình, anh em không biết rằng tình bạn với thế gian là sự thù ghét Thiên Chúa sao? Bất cứ ai chọn làm bạn của thế gian thì trở thành kẻ thù của Thiên Chúa.'
Rõ ràng, có một số người trong thính giả của Gia-cơ đã lạc xa khỏi đức tin, và Gia-cơ cảnh báo họ rằng bận tâm đến thế gian khiến họ trở thành kẻ thù của Thiên Chúa. Không có gì ngạc nhiên, vì vậy, Gia-cơ đã thể hiện thẩm quyền của mình với tư cách là một nhà lãnh đạo của hội thánh.
Đã nhiều lần, sứ đồ Gia-cơ dạy dỗ độc giả sống theo cách phù hợp với lời tuyên xưng đức tin chân thành. Ông đã dùng hơn 50 lời khuyên răn hoặc mệnh lệnh trực tiếp trong 108 câu Kinh Thánh.
và ông thường sử dụng các hình thức ngữ pháp khác có chức năng tương đương như mệnh lệnh trong bối cảnh của chúng. Nhưng giải pháp chính của Gia-cơ đối với những vấn đề mà thính giả của ông gặp phải không chỉ đơn thuần là ra lệnh cho họ làm điều này hay điều kia. Đối với ông, trọng tâm của vấn đề là họ cần phải theo đuổi sự khôn ngoan từ Thiên Chúa. Sự khôn ngoan từ Thiên Chúa là chìa khóa để nhận ơn vui mừng khi họ chịu đựng nhiều thử thách.
Hãy nghe những lời nổi tiếng trong chương bốn, câu 8 đến 10, nơi Gia-cơ nói với độc giả: 'Hãy đến gần Chúa và Ngài sẽ đến gần anh em. Hãy hạ mình xuống trước Chúa và Ngài sẽ nâng anh em lên.'
Gia-cơ hướng dẫn các tín hữu hạ mình để Thiên Chúa nâng họ lên. Ông dạy rằng khiêm nhường trước mặt Chúa là con đường dẫn đến sự khôn ngoan. Và khi những người theo Chúa Cứu Thế đến gần Thiên Chúa trong sự vâng phục khiêm nhường, sự khôn ngoan họ nhận được mang lại niềm vui, ngay cả khi họ kiên trì trong hoạn nạn.
Cho đến nay trong phần giới thiệu về sách Gia-cơ, chúng ta đã xem xét bối cảnh của sách này. Giờ đây chúng ta đã sẵn sàng xem xét cấu trúc và nội dung của Thư tín. Chúng tôi vừa đề xuất, rằng sách Gia-cơ tập trung rất nhiều vào sự khôn ngoan như là con đường tìm thấy niềm vui trong những giai đoạn thử thách.
Nhưng, sự nhấn mạnh này về sự khôn ngoan giúp chúng ta hiểu được một điều gì đó hơn là chỉ có mục đích của cuốn sách này. Nhiều nhà giải kinh đã nói về sách Gia-cơ như là sách khôn ngoan của Tân Ước. Và quan điểm này cũng giúp chúng ta nắm bắt cấu trúc và nội dung đặc biệt của sách Gia-cơ.
Đến thời James viết thư, đã có một lịch sử dài về văn học khôn ngoan bắt nguồn từ Cựu Ước. Các tác phẩm khôn ngoan của Cựu Ước bao gồm sách Gióp và Truyền Đạo, cùng Sách Truyền Đạo và một số sách khôn ngoan, cùng những câu khôn ngoan được gọi là.
Sự ghi nhận của Gia-cơ đối với văn chương Cựu Ước này là rõ ràng qua nhiều cách. Ví dụ, trong chương 5 câu 11, Gia-cơ đã dùng câu chuyện Gióp - nhân vật chính trong sách Gióp - để khích lệ sự kiên trì. Ngoài ra, Gia-cơ còn đề cập đến các chủ đề như lời nói, cách đối xử với góa phụ và trẻ mồ côi, nghèo khó và sự thiên vị.
Những chủ đề này phản ánh nhiều điểm tương đồng với nội dung của Sách Khải Huyền.
Khi đọc qua sách Gia-cơ, một trong những điều chúng ta thấy là điểm chung chính là khái niệm sự khôn ngoan. Ông rõ ràng coi trọng sự khôn ngoan, sự khôn ngoan từ trên cao trái ngược với sự khôn ngoan từ dưới. Tầm quan trọng này về sự khôn ngoan và cấu trúc sách khiến chúng ta nghĩ rằng có ảnh hưởng lớn trong đời sống của Ngài đối với văn chương khôn ngoan đã đến trước ông.
Bây giờ tôi nghĩ rằng chúng ta thấy điều đó một cách rõ ràng nhất trong việc trích dẫn và sử dụng sách Châm Ngôn và cũng trong cách ông ghi lại lời của Chúa chúng ta, của Chúa Cứu Thế Giê-xu, Đấng cũng thường nói trong bối cảnh của sự khôn ngoan. Bên cạnh đó, có một sự phát triển của quan niệm về sự khôn ngoan và văn chương khôn ngoan, một thể loại thực sự của văn chương khôn ngoan.
trong thời kỳ giữa hai giao ước. Và tôi nghĩ chúng ta thấy một số chủ đề tương tự qua các sách khôn ngoan trong sách Gia-cơ. Thỉnh thoảng chúng ta thấy cùng một cấu trúc. Nhưng tôi nghĩ nhiều chủ đề cũng đã được khởi đầu từ sách Phúc Âm Mác và cũng với Chúa Giê-xu.
Và vì vậy, tôi nghĩ ảnh hưởng lớn hơn đối với Gia-cơ có lẽ đến từ Chúa Giê-xu, nhưng thể loại đó cùng tầm quan trọng của sự khôn ngoan châm biếm trong suốt thời kỳ Đền thờ thứ hai của Do Thái giáo vào thời Chúa Giê-xu cũng rất quan trọng trong sách Gia-cơ.
Thư của Gia-cơ cũng phản ánh nội dung của các sách khôn ngoan có ảnh hưởng bên ngoài Kinh Thánh, như Sách Khôn ngoan của Sirach, còn được gọi đơn giản là Sirach, và Sách Khôn ngoan của Solomon. Những cuốn sách này rất được biết đến trong thời của Gia-cơ, và có những điểm tương đồng nổi bật với cả hai trong thư của ông.
Như một ví dụ, trong chương 1 câu 26 từ Sirach, chúng ta đọc, và Gia-cơ chương 1 câu 5 dạy chúng ta,
Ngoài các loại văn học khôn ngoan này, phần lớn lời chỉ dẫn của Chúa Giê-xu được ghi lại trong các sách Phúc Âm là đặc trưng của sự dạy dỗ khôn ngoan ở Israel, và các nhà giải kinh đã lưu ý một số điểm tương đồng giữa các bài viết của Gia-cơ và lời dạy của Chúa Giê-xu.
Ví dụ, Ma-thi-ơ chương 5 câu 10, nơi Đấng Chúa Giê-xu nói rằng: "Phước cho những ai bị bách hại vì sự công chính, vì vương quốc thiên đàng thuộc về họ."
Hãy so sánh điều này với Gia-cơ chương 1 câu 12, nơi Gia-cơ viết: 'Phước cho người nào chịu thử thách, vì khi đã đứng vững trong thử thách, người ấy sẽ nhận lãnh mão triều thiên sự sống mà Thiên Chúa đã hứa với những ai yêu mến Ngài.' Văn chương khôn ngoan của Do Thái giáo vào thế kỷ thứ nhất
Một chút trước đó, đã có ảnh hưởng đáng kể đối với sách Gia-cơ, đặc biệt là trong môi trường văn hóa và văn học mà tác giả đang làm việc. Trên thực tế, có hàng chục ám chỉ và tương đồng giữa sách Gia-cơ và các văn học khác, cả trong Cựu Ước và trong các văn học Do Thái khác.
Bạn biết rằng Gia-cơ trích dẫn từ sách Châm Ngôn hai lần, ít nhất một lần và có lẽ là hai lần, và ông có nhiều ẩn dụ, đặc biệt là về sự khôn ngoan của tác phẩm Josephus, một tác phẩm được viết vào khoảng một thế kỷ trước thời Tân Ước. Nhưng có một điều đặc trưng của sách Gia-cơ về mặt trí tuệ, và đó là
Ông liên kết sự khôn ngoan của mình rất chặt chẽ, với sự dạy dỗ của Chúa Giê-xu.
Gia-cơ có lẽ là một trong những nhà minh họa phong phú nhất trong Tân Ước với những hình ảnh về con tàu được dẫn dắt bởi bánh lái nhỏ, người nông dân kiên nhẫn chờ đợi và thương gia đi lại. Có nhiều, nhiều hình ảnh. Đó là tất cả ảnh hưởng của sự khôn ngoan, nhưng nội dung của Gia-cơ
đang thực sự mang theo cách mà Chúa Giê-xu trình bày về vương quốc Thiên Chúa và cách sự hiện diện của vương quốc thay đổi cuộc sống của bạn. Vì Gia-cơ có mối quan hệ thân thiết với văn học khôn ngoan, cấu trúc của sách Truyền Đạo khá khác với những gì chúng ta có thể mong đợi. Ngay cả một cái nhìn ngắn gọn về bức thư này cũng cho chúng ta biết rằng cấu trúc của nó không đơn giản.
Trên thực tế, từ quan điểm hiện đại của chúng ta, nó có vẻ khá vô trật tự. Giống như sách Châm Ngôn, sách Gia-cơ đề cập nhiều chủ đề quan trọng, và thường chỉ dành vài câu cho một chủ đề trước khi chuyển sang chủ đề khác. Thỉnh thoảng, nó sẽ quay lại những chủ đề đó trong phần sau, nhưng không theo một quy luật nhất định nào.
Một số nhà bình luận thậm chí còn kết luận rằng không có cấu trúc nào cho sách Gia-cơ. Họ đã gợi ý rằng đó chỉ là tuyển tập những lời khôn ngoan không có trật tự hay mạch suy nghĩ thực sự. Nhưng chúng ta phải cẩn thận ở đây. Thư này không chỉ là một mớ hỗn độn của những câu không liên quan được đặt chung mà không có bất kỳ trật tự nào.
Mặc dù sách Gia-cơ giống với văn học khôn ngoan về cả hình thức lẫn nội dung, nhưng nó cũng khác với thể loại đó theo nhiều cách. Không giống như các văn học khôn ngoan khác, Gia-cơ là một bức thư được viết cho các hội thánh cụ thể. Và vì lý do này, nó phản ánh một số đặc điểm tổ chức của các tác phẩm Tân Ước khác.
Không có sự đồng thuận nào trong giới chú giải về cách tổ chức hay cấu trúc của sách Gia-cơ. Nhưng vì mục đích bài học này, chúng tôi đã chia sách thành bảy phần. Thư Gia-cơ mở đầu bằng lời chào hỏi trong Gia-cơ chương 1 câu 1.
Phần chính đầu tiên là phần giới thiệu các chủ đề chính của sách, mà chúng ta có thể gọi là sự khôn ngoan và niềm vui trong sách Gia-cơ chương 1 câu 2 đến 18. Phần chính thứ hai thể hiện mối quan tâm của Gia-cơ về sự khôn ngoan và vâng lời trong sách Gia-cơ chương 1 câu 19 đến chương 2 câu 26.
Phần chính thứ ba đề cập đến sự khôn ngoan và hòa bình trong hội thánh trong Gia-cơ 3-1, 4-12. Phần chính thứ tư tập trung vào sự khôn ngoan và tương lai trong Gia-cơ 4-13, 5-12.
Phần chính thứ năm và cuối cùng dành cho những gì chúng ta có thể mô tả là sự khôn ngoan và cầu nguyện trong sách Gia-cơ chương 5 câu 13 đến 18. Sau năm phần chính này, có một lời kết luận trong chương 5 câu 19 và 20. Hãy xem xét kỹ hơn từng phần này, bắt đầu với lời chào trong sách Gia-cơ chương 1 câu 1.
Hãy lắng nghe một lần nữa chương 1 câu 1, Gia-cơ nói ngắn gọn. Chúng ta không nên bỏ qua cách Gia-cơ mô tả chính mình ở đây. Ông tự gọi mình là tôi tớ của Thiên Chúa và của Chúa Giê-xu.
Gia-cơ có thể đã tự giới thiệu mình là người lãnh đạo hội thánh, hoặc thậm chí là anh em của Chúa Giê-xu. Thay vào đó, ông đã chọn nêu rõ mình là người hầu việc Chúa và với Chúa Cứu Thế. Điều này có thể là lời tuyên bố cá nhân của Gia-cơ về sự khiêm nhường, một chủ đề ông đề cập sau đó trong sách này.
Ở đây, ông minh họa sự khiêm nhường đó bằng cách làm cho rõ ràng rằng ông là đầy tớ của anh em mình là Chúa Giê-xu. Theo lời chào, phần chính đầu tiên tập trung vào những gì chúng ta gọi là sự khôn ngoan và niềm vui.
Gia-cơ đã viết thư của mình cho các Cơ Đốc nhân bị đuổi khỏi Giê-ru-sa-lem và bị phân tán khắp vùng Địa Trung Hải. Họ phải đối mặt với nhiều loại thử thách khác nhau mà chắc chắn đã khiến họ nản lòng. Và vì lý do này, những lời đầu tiên của Gia-cơ về tầm quan trọng của sự khôn ngoan bắt đầu bằng một lời kêu gọi hãy vui mừng.
Hãy lắng nghe một lần nữa Gia-cơ chương 1 câu 2, nơi Gia-cơ nói với thính giả của mình: Đoạn Kinh Thánh này có vẻ khác thường đối với chúng ta, đặc biệt là vì nó nói đến những người đang đối diện với nhiều loại thử thách.
Nhưng lời kêu gọi của Gia-cơ về việc xem thử thách là niềm vui trọn vẹn không phải là điều bất thường như chúng ta nghĩ. Cụm từ 'niềm vui trọn vẹn' xuất phát từ cách diễn đạt tiếng Hy Lạp, passion kharan, có thể dịch là niềm vui trọn vẹn không pha trộn. Cách khích lệ này phù hợp với văn chương khôn ngoan khác trong thời đại của Gia-cơ.
Nhiều lần, các tác phẩm khôn ngoan khích lệ những người chịu khổ hãy coi mình là người được phước. Chẳng hạn, Chúa Giê-xu kết thúc các Phúc Âm trong Ma-thi-ơ chương 5 câu 12 bằng lời kêu gọi hãy vui mừng và hân hoan trước sự bắt bớ.
Như đã nói trước đó, trong chương 1 câu 3-4, Sứ đồ Gia-cơ dạy rằng sự kiên trì qua thử thách giúp tín hữu trưởng thành và trọn vẹn. Nói cách khác, khi dân sự của Thiên Chúa chịu đựng gian khổ, họ sẽ trưởng thành trong sự trọn vẹn trong mọi điều Thiên Chúa định sẵn cho họ.
Nhưng trên thực tế, ngay cả người tin tưởng chân thành nhất cũng khó có thể thấy điều này đúng khi đang chịu đau khổ. Đây là lý do tại sao, ngay trong câu tiếp theo, thánh Gia-cơ đã bảo độc giả của mình theo đuổi sự khôn ngoan từ Thiên Chúa.
Bạn sẽ nhớ lại rằng trong Gia-cơ chương 1 câu 5 chép rằng: 'Nếu có ai trong anh em thiếu sự khôn ngoan, hãy cầu xin Thiên Chúa Hằng Hữu, là Đấng ban cho cách rộng rãi cho mọi người.'
Những ai muốn có niềm vui trọn vẹn khi trải qua thử thách phải xin Chúa ban cho sự sáng suốt. Họ cần sự khôn ngoan để hiểu được cách thử thách dẫn đến sự trưởng thành hơn cho mình. Và nếu chúng ta xin sự khôn ngoan ấy từ Chúa, Ngài sẽ ban cho chúng ta. Như Gia-cơ đã nói trong chương 1 câu 17, Chúa ban cho những ân tứ tốt lành và hoàn hảo cho dân Ngài.
Gia-cơ đã kết thúc phần này trong Chương 1, câu 18 với sự xác tín này.
Khi chúng ta nhận lãnh sự khôn ngoan để hiểu cách Thiên Chúa hành động qua những thử thách, chúng ta có thể vui mừng. Sự khôn ngoan củng cố lòng tin quyết rằng Thiên Chúa đã định cho chúng ta ơn phước là sự cứu rỗi đời đời. Sau phần thảo luận về sự khôn ngoan và niềm vui, Gia-cơ chuyển sang mối quan hệ giữa sự khôn ngoan và sự vâng lời.
Trong phần này, Gia-cơ đã thảo luận về sự khôn ngoan và sự vâng lời trong ba bước cơ bản. Để bắt đầu, Chương 1:19-27 giới thiệu tầm quan trọng của việc hành động thay vì chỉ lắng nghe hoặc nói chuyện. Trong Chương 1:22, chúng ta đọc thấy phần này.
Đừng chỉ nghe theo lời và tự lừa dối mình. Hãy làm theo lời đó. Chỉ nghe lời thôi thì chưa đủ. Lời khôn ngoan từ Thiên Chúa cũng phải dẫn đến sự vâng lời trung tín. Nếu không, chúng ta đang tự lừa dối mình.
Khi bạn đọc bức thư của James, bạn hiểu rằng ông thực sự nhấn mạnh vào việc áp dụng một cách thực tế những điều chúng ta nói rằng chúng ta tin. Đó là một chủ đề rất nổi bật xuyên suốt toàn bộ bức thư. Bạn đặt câu hỏi tại sao James nhấn mạnh điều đó và câu trả lời đầu tiên dường như là James sống trong thế giới thực.
Ông phục vụ cho con người thật và thế giới chúng ta đang sống là nơi lời nói rẻ tiền, nơi rất dễ nói rằng chúng ta tin vào Thiên Chúa nhưng lại khó làm theo những gì niềm tin đó thể hiện trong hành động.
Điều này dường như là một thử thách không chỉ đối với Gia-cơ mà cả với Chúa Giê-xu. Nói suông không giống như làm. Chúa Giê-xu biết điều đó. Gia-cơ cũng biết điều đó. Chúng ta đang cố gắng đến với những con người thực tế trong thế giới thực với một vấn đề thực tế. Gia-cơ mong đợi những người đọc của mình làm nhiều hơn chỉ nghe Lời Chúa. Ông mong đợi họ đặt đức tin vào hành động thực hành.
Chủ đề này rất quan trọng đối với Gia-cơ đến nỗi, mặc dù ông chủ yếu thảo luận về nó trong các chương 1 và 2, ông đã quay lại với nó theo định kỳ xuyên suốt bức thư của mình. Ví dụ, trong chương 3 câu 13, quan điểm cơ bản của Gia-cơ về mối quan hệ giữa sự khôn ngoan và sự vâng lời lại xuất hiện một lần nữa.
Gia-cơ viết: 'Ai là người khôn ngoan và thông hiểu giữa vòng anh em? Hãy để người ấy thể hiện điều đó qua đời sống đạo đức, bằng những việc làm xuất phát từ sự khiêm nhường đến từ sự khôn ngoan.' Như câu Kinh Thánh này chỉ ra, sự khôn ngoan và thông hiểu về mục đích của Thiên Chúa trong những thử thách và đau khổ không chỉ là vấn đề trí thức.
Những người có đức tin sẽ thể hiện qua đời sống đạo đức, bởi những việc làm xuất phát từ sự khiêm nhường đến từ khôn ngoan mà Chúa Hằng Hữu ban cho. Vì vậy, trong chương 1 câu 27, Gia-cơ kết thúc phần này về việc cần hành động bằng cách tóm tắt đạo đức đích thực như sau.
Lòng đạo mà Thiên Chúa Cha chấp nhận là trong sạch và không tì vết là thế này, chăm sóc trẻ mồ côi và góa phụ trong cảnh hoạn nạn của họ và giữ mình khỏi bị ô uế bởi thế gian. Gia-cơ nói rất thẳng thắn về lòng đạo mà ông gọi là trong sạch và không tì vết là thế này, chăm sóc trẻ mồ côi và góa phụ trong cảnh hoạn nạn của họ.
và để giữ mình khỏi bị ô nhiễm bởi thế gian. Trong nền văn hóa chúng ta, vốn rất thiên về chủ nghĩa vật chất, đó là hai mặt trong một đồng tiền, rằng một trong những cách chúng ta bị ô nhiễm bởi thế gian không phải là một cách không quan tâm đến người nghèo xung quanh, hay đổ lỗi cho sự nghèo đói của họ chỉ do lỗi của riêng họ.
và không nhìn vào những nguyên nhân hệ thống của nó. Hoặc nhìn vào chính chúng ta - những người có phương cách như có nghĩa là bằng cách nào đó chúng ta vượt trội hơn hoặc chúng ta có phước lành của Chúa, còn người nghèo thì không. Thực tế, thường thấy rằng đức tin của người nghèo mạnh mẽ hơn và chân thật hơn những người
Những người không phải chịu cùng những điều mà họ đã phải chịu. Theo sau lời kêu gọi hành động giới thiệu này, Gia-cơ đã nói sâu hơn về mối liên hệ giữa sự khôn ngoan và sự vâng lời bằng cách tập trung vào vấn đề thiên vị trong thư Gia-cơ chương 2 câu 1 đến 13.
Một số người trong vòng thính giả của sứ đồ Gia-cơ rõ ràng đã ưu ái người giàu và bỏ qua người nghèo. Trong phần này, Gia-cơ đã giải quyết vấn đề bằng cách kêu gọi họ một cách đúng đắn chú ý đến điều ông gọi là Luật Pháp Hoàng Đế.
Trong chương 2 câu 8, Gia-cơ nói: Về cơ bản, bỏ qua người nghèo để ưu tiên người giàu là sự thiếu yêu thương người lân cận. Và Gia-cơ dạy rằng họ phải tránh sự thiên vị bằng cách tuân giữ luật pháp nhà vua.
Chúng ta thấy trong lời dạy của Gia-cơ về người giàu và mối quan hệ của họ với người nghèo là một sự phản ánh thực sự của lời dạy của Chúa Cứu Thế trong Lu-ca chương 16. Trong chương 2 của Gia-cơ, ông nói về việc bạn không biết rằng Thiên Chúa đã chọn những người nghèo, những người yêu mến Ngài, để trở thành những người thừa kế vương quốc của Ngài, những người giàu có.
đang được thể hiện một phần tất cả khi họ bước vào các buổi nhóm của Cơ Đốc giáo, họ đang thể hiện sự tôn trọng, bạn có thể ngồi vào chỗ của tôi, bạn có thể ngồi vào chỗ tốt nhất trong hội thánh. Và Gia-cơ cảnh báo những người đang hành động theo cách đó phải nhớ rằng người nghèo có đầy đủ quyền thừa kế trong vương quốc của Chúa, đầy đủ quyền thừa kế và do đó họ nên được thể hiện phẩm giá và sự tôn trọng.
và tư cách thành viên đầy đủ trong cộng đồng dân Chúa. Như chúng ta đã thấy, sách Gia-cơ có trọng tâm rất tích cực vào luật pháp của Thiên Chúa. Theo quan điểm của Gia-cơ, luật pháp dạy chúng ta phải quan tâm lẫn nhau, cầu xin lòng thương xót dành cho người nghèo, tránh tính thiên vị và những điều tương tự. Nhưng quan điểm tích cực này có thể bị sử dụng sai lầm nếu chúng ta không cẩn thận.
Các Cơ Đốc nhân hiện đại thường chỉ ra sự lạm dụng luật pháp của Thiên Chúa như một cách để cố gắng biện minh cho chính mình trước mặt Thiên Chúa bởi những việc làm công bình của chúng ta. Và chúng ta đúng khi bác bỏ sự lạm dụng luật pháp này. Nhưng ngược lại, sách Gia-cơ nhấn mạnh một khía cạnh khác của luật pháp.
Gia-cơ dạy rằng mặc dù không ai có thể được xưng công bình bởi luật pháp, nhưng luật pháp của Thiên Chúa là nguồn sự khôn ngoan của chúng ta và chúng ta nên sống trong sự vâng phục nó. Tất nhiên, chúng ta không vâng theo luật pháp như thể chúng ta vẫn còn sống trong thời Cựu Ước. Chúng ta phải không ngừng áp dụng luật pháp của Thiên Chúa bởi ánh sáng của Chúa Cứu Thế và những lời dạy của Tân Ước.
Nhưng những người đã tin cậy Chúa Cứu Thế để được cứu thì vâng theo luật pháp, vì lòng biết ơn đối với Thiên Chúa, vì đó là sự bày tỏ khôn ngoan của Ngài.
Theo nghĩa này, Gia-cơ nhắc lại Thi Thiên 19:7 và 8, nơi chúng ta đọc thấy điều này,
Sau khi giới thiệu tầm quan trọng của hành động khi đáp ứng với Lời khôn ngoan và chống lại sự thiên vị bằng cách vâng theo luật pháp tối cao của Thiên Chúa, sứ đồ Gia-cơ đã đề cập đến mối quan hệ giữa đức tin và sự vâng lời trong chương 2 câu 14 đến 26.
Trong chương 2 đoạn câu 14, Gia-cơ đặt ra câu hỏi này,
Gia-cơ trả lời câu hỏi này bằng một câu trả lời dứt khoát là không. Ông đã làm điều này theo một số cách. Đầu tiên, ông chỉ ra rằng ngay cả ma quỷ cũng tin những điều đúng đắn về Thiên Chúa, nhưng không mang lại ích lợi gì cho nó. Sau đó, ông lưu ý rằng đức tin của Áp-ra-ham đã dẫn đến sự vâng lời như thế nào, và ông mô tả thể nào Ra-háp đã chứng minh đức tin của mình qua việc làm tốt.
Vì vậy, trong Chương 2 câu 26, Gia-cơ đã đưa ra kết luận được biết đến này. Theo Gia-cơ, có niềm tin đúng đắn chưa đủ. Đức tin không được bày tỏ qua sự vâng phục thì chẳng khác nào chết. Đó không phải là đức tin cứu rỗi thực sự.
Sau khi khuyên giục thính giả sống đời vâng lời, Gia-cơ tập trung vào mối quan hệ giữa sự khôn ngoan và bình an giữa vòng những người theo Chúa Cứu Thế. Hãy nghe câu hỏi của Gia-cơ trong chương 4 câu 1: 'Điều gì gây ra những tranh chiến giữa các anh em?'
Mặc dù câu này xuất hiện ở giữa phần này, theo nhiều cách khác nhau, toàn bộ phần này liên quan đến câu hỏi này. Trong phần này, Gia-cơ đã lưu ý ba vấn đề chính liên quan đến sự khôn ngoan và hòa bình giữa các tín hữu. Đầu tiên, trong Chương 3 câu 1 đến 12, Gia-cơ đang tập trung vào lưỡi, hoặc cách sử dụng từ của chúng ta.
Trong chương 3 câu 4 và 5, sứ đồ Gia-cơ so sánh lưỡi với bánh lái tàu. Ông giải thích như sau: 'Các tàu lớn lao và bị gió mạnh cuốn đi, nhưng được điều khiển bởi bánh lái rất nhỏ. Lưỡi cũng vậy, chỉ là phần nhỏ của thân thể mà lại gây ra những lời phạm thượng lớn lao.'
Sau đó, trong câu 6, ông tiếp tục, nói với độc giả, sự cảnh báo của Gia-cơ về khả năng làm điều ác của lưỡi rất giống với những gì chúng ta tìm thấy trong sách Châm ngôn.
Sách Châm Ngôn cũng đề cập đến những nguy hiểm liên quan đến lưỡi, hay lời nói, nhiều lần. Chúng ta thấy điều này trong các chỗ như Châm Ngôn 10:31, 11:12, 15:4, và nhiều câu khác. Cả Gia-cơ và Châm Ngôn đều chỉ ra rằng lời nói có thể gây ra đủ loại rắc rối giữa dân sự của Thiên Chúa.
Để tránh xung đột và sống trong hòa bình, chúng ta phải kiểm soát. Những hướng dẫn về hội thánh để chúng ta sống trong ánh sáng của sự tái lâm của Chúa Cứu Thế, trong sự trông đợi về sự trở lại tương lai của Ngài. Một trong những cách mà sách Gia-cơ đưa ra để đánh giá lòng mình là tập trung vào lời nói của chúng ta.
Nói cách khác, Gia-cơ xem lời nói của một người, lưỡi, là cách nói tắt cho những lời nói, như biểu hiện của toàn bộ con người đạo đức. Nó cho thấy nhiệt độ tấm lòng. Và như Chúa Giê-xu dạy: 'Nước mắt tuôn trào từ lòng, mới nói ra từ miệng', khi Gia-cơ nói rằng người ta phải kiềm chế lưỡi mình,
Và không nên như vậy, từ cùng một miệng lại nói ra những lời chúc phúc và lời nguyền rủa. Ngài bảo chúng ta rằng lòng chúng ta phải hoàn toàn cam kết với Thiên Chúa. Chúng ta không được là người đa lâng nhưng chúng ta phải bằng đức tin giữ vững lời dạy của Đấng Christ. Và khi chúng ta làm điều đó, lời nói của chúng ta nên chúc phúc cho anh chị em mình thay vì nguyền rủa họ.
Vấn đề thứ hai liên quan đến sự khôn ngoan và bình an là hai loại khôn ngoan. Chúng ta tìm thấy điều này trong chương 3 câu 13 đến 18. Trong Gia-cơ chương 3 câu 14 đến 17, chúng ta đọc phân đoạn này.
Nếu anh em nuôi dưỡng sự đố kỵ cay đắng và tham vọng ích kỷ trong lòng, thì sự khôn ngoan ấy chẳng đến từ trời, nhưng là trần tục, khởi phát từ lòng ham muốn, thuộc về ma quỷ. Nhưng sự khôn ngoan từ trời đến trước hết là thuần khiết, sau đó là yêu thương hòa thuận, nhân từ, nhịn nhục, đầy lòng thương xót và kết quả tốt lành, không thiên vị và chân thật.
Như chúng ta thấy ở đây, để giải thích mối quan hệ giữa sự khôn ngoan và hòa bình, Gia-cơ đã phân biệt giữa sự khôn ngoan thuộc thế gian, ngay cả sự khôn ngoan của quỷ dữ, và sự khôn ngoan đến từ thiên đàng. Sự khôn ngoan thuộc thế gian dẫn đến sự ghen tị cay đắng và tham vọng ích kỷ, nhưng sự khôn ngoan đến từ Thiên Chúa mang lại hòa bình trong cộng đồng Cơ Đốc.
Gia-cơ kêu gọi độc giả từ bỏ những cuộc chiến đấu và tranh cãi của họ. Ông giải thích rằng khi chúng ta bám chặt vào những ham muốn ích kỷ của mình thì không thể có sự bình an giữa chúng ta. Sự khôn ngoan trần tục, ông dạy, chỉ dẫn đến sự lộn xộn và mọi việc ác.
Vì vậy, Gia-cơ hướng dẫn độc giả của mình hãy dựa vào sự khôn ngoan đến từ Thiên Chúa. Khi chúng ta làm điều này, chúng ta tìm thấy bình an. Như Gia-cơ nói trong Kinh Thánh Chương 3, câu 18, những người gây dựng hòa bình trong sự bình an gieo trồng mùa gặt công chính.
Vấn đề thứ ba trong phần này, trong Chương 4 câu 1 đến 12, xem xét sự khôn ngoan và hòa bình trong mối quan hệ với xung đột bên trong mà những người theo Chúa Cứu Thế trải nghiệm. Gia-cơ đã xác định xung đột giữa các Cơ Đốc nhân đến từ những ham muốn ích kỷ, động cơ sai lầm và sự bất mãn trong chúng ta. Theo quan điểm của Gia-cơ,
Những ham muốn xấu xa trong vòng thính giả của ông đã gây tổn hại lớn cho cộng đồng Cơ Đốc. Họ bị chi phối bởi những ham muốn của mình. Và vì thế, họ đã chiến đấu, che giấu sự thật, thậm chí phá hoại lẫn nhau. Vì vậy, thánh Gia-cơ nghiêm khắc nói với họ điều họ phải làm để mang lại hòa bình.
Trong chương 4 câu 7 đến 10, Gia-cơ nói rằng chỉ có sự vâng phục khiêm nhường đối với Thiên Chúa mới chấm dứt những tranh đấu và cãi vã của họ, và ban cho họ sự bình an giữa anh em.
Bây giờ, chúng ta hãy xem xét mối quan hệ giữa sự khôn ngoan và tương lai. Phần thảo luận về sự khôn ngoan và tương lai của Gia-cơ có thể được chia thành ba phần. Phần đầu tiên được tìm thấy trong Chương 4 câu 13 đến 17, đề cập đến những người đang lập kế hoạch cho tương lai như thể Thiên Chúa Hằng Hữu không cai trị.
Những câu Kinh Thánh này cho thấy nhiều thính giả của sứ đồ Gia-cơ đang cố xác định tương lai riêng. Họ chú tâm tích lũy của cải và khoe khoang những điều sẽ làm/đi. Để đáp lại, những lời Gia-cơ nhắc họ rằng cuộc đời họ chỉ là phù du. Họ không thể biết tương lai đang chờ đợi mình.
Hãy lắng nghe chương 4 câu 15 và 16, nơi Gia-cơ nói với họ: 'Chỉ Thiên Chúa mới kiểm soát tương lai, và những người khôn ngoan sẽ nhận biết chân lý này.'
Trong phần thứ hai của phần này, James đã chú ý đến sự khôn ngoan và tương lai theo một cách hơi khác. Trong chương 5 câu 1 đến 6, ông kêu gọi đừng tích trữ của cải không cần thiết vì ngày phán xét trong tương lai.
Gia-cơ đã nói rất dài về cách đối xử với người nghèo ở nhiều nơi, và ông liên tục lên án người giàu dữ dội vì lợi dụng những người kém may mắn hơn. Trong những câu này, Gia-cơ đã cảnh báo mạnh mẽ những người giàu đã kiếm được của cải bằng cách làm tổn hại người nghèo, và ông thông báo cho họ rằng họ sẽ sớm phải chịu hình phạt.
Như Ngài nói trong chương 5 câu 3, vàng bạc của các ngươi bị hư hoại. Sự hư hoại ấy sẽ làm chứng nghịch cùng các ngươi và giết chết các ngươi như lửa. Các ngươi đã tích trữ của cải trong những ngày sau rốt. Như phân đoạn này cho thấy, việc tích lũy của cải bằng cách làm tổn hại người khác sẽ đem đến sự phán xét nghiêm trọng.
Những gì sứ đồ Gia-cơ nói cơ bản là điều sẽ gây sốc cho nhiều người Do Thái nghe ông. Về cơ bản, ông đảo ngược cách hiểu mà nhiều người Israel đã có về mối quan hệ giữa người giàu và người nghèo. Và ông thực sự kêu gọi người nghèo là những người được phước và cảnh báo những người giàu.
để thực sự sẵn sàng ăn năn và mong đợi sự phán xét. Cơ sở cho sự phán xét đó là những người này đang tích trữ của cải của họ, về cơ bản, nếu bạn được ban phước với của cải, ý muốn của Chúa là bạn sẽ chia sẻ điều này với người hàng xóm của bạn, sử dụng nó để ban phước cho người hàng xóm của bạn. Nhưng họ đang tích trữ nó cho chính họ. Họ đang lừa dối người lao động của mình bằng cách không trả cho họ một mức lương công bằng.
Sự giàu có là một món quà của Thiên Chúa mà sau đó được sử dụng theo ý muốn của Ngài, không phải cho chính bạn, mà cuối cùng là cho người lân cận. Nói cách khác, mọi doanh nghiệp nên được hướng dẫn bởi nguyên tắc: 'Hãy yêu người lân cận như chính mình'. Phần thứ ba trong phân đoạn của Gia-cơ bàn về sự khôn ngoan trong tương lai, từ chương 5 câu 7 đến 12, chuyển sang nói về việc kiên nhẫn chờ đợi kế hoạch của Thiên Chúa trong tương lai được diễn ra.
Gia-cơ đã chỉ trích những người lên kế hoạch mà không nhờ cậy Thiên Chúa ban sự khôn ngoan. Ông cảnh báo những kẻ khinh dể sự khôn ngoan của Thiên Chúa bằng cách tích trữ của cải và ngược đãi người nghèo rằng họ sẽ thấy sự phán xét của Ngài. Nhưng sau đó, Gia-cơ khích lệ những người đang chịu khổ hãy kiên nhẫn chờ đợi Thiên Chúa hoàn tất sự trọn vẹn của lịch sử.
Hãy nghe Gia-cơ chương 5 câu 7 và 8 khi ông dùng hình ảnh này: 'Vậy, thưa anh chị em, chúng ta hãy kiên nhẫn cho đến khi Chúa tái lâm. Hãy xem người nông dân kiên nhẫn chờ đợi đất sinh hoa lợi, chờ những cơn mưa mùa thu và mùa xuân. Anh chị em cũng phải kiên nhẫn và đứng vững, vì sự tái lâm của Chúa rất gần.'
Như chúng tôi vừa chỉ ra, lời của Gia-cơ trong phần này không chỉ khuyên răn người giàu mà còn khuyến khích người nghèo và bị áp bức. Lời quở trách mạnh mẽ của Gia-cơ nhắc nhở khán giả rằng ngày phán xét đã đến. Và vào thời điểm đó, những người đã trung thành tin cậy nơi Thiên Chúa sẽ được thưởng.
Theo cách này, Ngài khích lệ những người trung thành tiếp tục trên con đường sự khôn ngoan vinh quang, sống theo lời tuyên xưng đức tin, vâng phục Thiên Chúa trong ánh sáng của cuối cùng vĩ đại trong kế hoạch của Thiên Chúa cho tương lai.
Sau khi giải thích cho độc giả về sự liên quan của sự khôn ngoan với niềm vui, sự vâng lời, sự bình an và tương lai, sách Gia-cơ kết thúc bằng một áp dụng thực tiễn ngắn gọn về sự khôn ngoan và cầu nguyện. Độc giả của sách Gia-cơ đang đương đầu với nhiều vấn đề.
Họ đã bị tản lạc khỏi nhà chỗ của mình. Người giàu đang áp bức người nghèo. Họ đã tranh cãi và làm tổn thương nhau. Rất nhiều người, dường như, đang bị chi phối bởi những ham muốn ích kỷ của mình, và họ thấy khó sống theo cách phù hợp với lời tuyên xưng đức tin của họ. Vì vậy, trong phần cuối cùng này, Gia-cơ đã dạy họ phải làm gì trong cộng đồng Cơ Đốc khi họ đối mặt, với những công việc đấu tranh này.
Tương tự như những gì ông đã dạy vào lúc bắt đầu nhật thực. Ở đây, James đã hướng dẫn họ phải dâng mình cho sự cầu nguyện. Trong những lúc khó khăn hoặc vui mừng, khi đối diện với bệnh tật, thậm chí bệnh tật do tội lỗi cá nhân gây ra, những kẻ có trí tuệ sẽ cầu nguyện.
Đừng nghe chương 5 câu 13 và 14, nơi Gia-cơ nói với độc giả của mình: 'Có ai trong anh em đang gặp rắc rối không? Hãy cầu nguyện cho người ấy. Có ai trong anh em đang vui vẻ không? Hãy hát những bài ca ngợi khen. Có ai trong anh em đau ốm không? Hãy mời các trưởng lão hội thánh đến cầu nguyện cho người ấy.'
Rõ ràng, Gia-cơ mong đợi độc giả của mình đến gần Thiên Chúa để nhận sự khôn ngoan trong mọi hoàn cảnh. Lý do cho điều này là đủ rõ ràng trong câu 16, nơi Gia-cơ nói,
Sau khi hoàn thành phần chính của bức thư với lời kêu gọi kiên nhẫn và cầu nguyện, Gia-cơ kết thúc thư bằng một sự khích lệ. Trong chương 5 câu 19 và 20, Gia-cơ khuyên giục thính giả hãy coi sóc nhau và giúp những người đã lạc lối trở về với lẽ thật.
Ông nhắc nhở họ rằng, với tư cách là anh chị em trong cộng đồng đức tin, họ có nghĩa vụ và đặc ân là dẫn người khác trở lại với đức tin thật sự cứu rỗi.
Trong phần giới thiệu này về sách Gia-cơ, chúng ta đã xem xét bối cảnh của sách và lưu ý tác giả, độc giả và dịp viết. Chúng ta cũng đã khám phá cấu trúc và nội dung bức thư và thấy rằng cuốn sách này đóng vai trò là cuốn sách khôn ngoan trong Tân Ước, cho những tín hữu đang đối diện với sự nản lòng qua thử thách, qua niềm vui, sự vâng lời, bình an, tương lai và sự cầu nguyện.
Sách Gia-cơ đã thúc giục các Cơ Đốc nhân thế kỷ thứ nhất tìm kiếm Thiên Chúa để nhận sự khôn ngoan, hầu họ có thể có niềm vui khi trải qua thử thách. Dĩ nhiên, bạn và tôi sống trong hoàn cảnh rất khác với độc giả nguyên thủy của Gia-cơ, nhưng chúng ta cũng đối mặt với những thử thách và cũng cần sự khôn ngoan từ Thiên Chúa để chúng ta có thể đáp ứng những thử thách đó.
Giống như đối tượng đầu tiên của thư Gia-cơ, chúng ta cần niềm vui thánh khiết mà sự khôn ngoan của Chúa mang lại. Dù trong bài học này chúng ta chỉ mới khám phá những điều sách này trình bày, một điều cần làm rõ là Thư Gia-cơ vạch ra con đường sống khôn ngoan cho mọi thời đại.
Và càng áp dụng sách này vào đời sống, chúng ta càng nhận được phước lành niềm vui thuần khiết mà Chúa ban cho dân Ngài, bất kể chúng ta đối diện thử thách hay khó khăn nào.
"""
  
  sources = source_texts.split("\n")
  targets = target_texts.split("\n")
  for index, _ in enumerate(sources):
    if sources[index] and targets[index]:
      result = llm.predict(sources[index], targets[index], "en", "vi")
      # print("source::\n", targets[index])
      # print(f"target::\n {result}\n\n")
  
