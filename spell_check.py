
## download models from gdrive to 
## ~/miniconda3/envs/soni/lib/python3.10/site-packages/data/checkpoints/
import neuspell

class SpellCheck():
  def __init__(self):
    # self.checker = neuspell.CnnlstmChecker()
    self.checker = neuspell.SclstmChecker()
    # self.checker = neuspell.NestedlstmChecker()
    self.checker.from_pretrained()
        
  def correct(self, text):
    return self.checker.correct(text)