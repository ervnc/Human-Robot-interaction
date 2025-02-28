import ollama

class LLM:
  def __init__(self, prompt):
    self.prompt = prompt
    self.client = ollama.Client()
    self.model = "deepseek-r1:1.5b"

  def generate(self):
    response = self.client.generate(model=self.model, prompt=self.prompt)
    return response

if __name__ == "__main__":
  prompt = "hello"
  llm = LLM(prompt)
  print(llm.generate().response)