import backoff
import openai
from openai import OpenAI
import re
import copy
import os

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

openai_client = OpenAI(
    api_key=OPENAI_API_KEY,
    base_url=OPENAI_BASE_URL
)

@backoff.on_exception(backoff.expo, (openai.RateLimitError, openai.Timeout, openai.APIError, openai.APIConnectionError, openai.APIStatusError))
def get_precise_response(messages, model="gpt-4o-mini", temperature=0.2, top_p=0.1):
        message = openai_client.chat.completions.create(
            model=model,
            messages=messages,
            temperature=temperature,
            top_p=top_p,
        )
        return message.choices[0].message.content

def heuristic_moderator(context):
    if 'goodbye' in context[-1].lower() or 'good bye' in context[-1].lower():
        return True
    words1 = set(context[-1].lower().split())
    words2 = set(context[-3].lower().split())
    intersection = words1.intersection(words2)
    overlap_degree = len(intersection) / min(len(words1), len(words2))
    if overlap_degree > 0.9:
        return True  
    return False  

def moderator(context):
    user_prompt = """Your task is to assess the current state of the conversation (the most recent utterances) and determine whether the conversation has concluded.
The conversation is considered to have concluded if any of the following conditions are met:
- The Client and Counselor work out an actionable plan together.
- The Counselor decides not to pursue any changes in the Client's behavior and communicates readiness to provide support in the future.

Here are some examples to help you understand the task better:
## Example 1
Conversation Snippet:
Counselor: Hello. How are you?
Client: I am good. What about you?
Counselor: I'm doing well, thank you. How have things been going for you lately?
Client: I don't think my behavior is problematic or needs to change.

Question: Should the conversation be concluded?
Response:
Conversation State: The Client expresses reluctance to consider any changes in behavior, indicating a lack of perceived necessity for change. The Counselor has not yet attempted to propose a plan or persuade the Client to consider any changes.
End or Not: No


## Example 2
Conversation Snippet:
Counselor: Okay. Try to cut back a bit and see how that goes.
Client: Yeah. I mean, I-I don't know how it's going to go, especially with trying to cut back on smoking already, but, I mean, I can try.
Counselor: Okay. So, you can give it a go and, I think doing managing, you might just learn a little bit about how easy or difficult it's going to be, and-and also what it's like to cope with stress in other ways.
Client: Yeah.
Counselor: Okay. So, it sounds like we have two avenues to pursue, anything else?
Client: I don't think so.
Counselor: Okay. Well, that sounds good. So, why don't we meet again in a couple of weeks and you kind of update me on how cutting back on your drinking has gone, and also, what it was like when you met with a social worker? Sound good?
Client: Yeah.
Counselor: Okay, great. I look forward to talking next time.
Client: Okay. Thanks.

Question: Should the conversation be concluded?
Response:
Conversation State: The Client agrees to attempt cutting back on alcohol and smoking, expressing uncertainty but willingness to try. The Counselor suggests a follow-up meeting to discuss progress, establishing a future point of contact and support.
End or Not: Yes


## Example 3
Conversation Snippet:
Counselor: It's great to hear your willingness to consider change. What are some reasons that make you think reducing alcohol consumption could be beneficial for you?
Client: I realize that reducing alcohol consumption could improve my overall health and well-being. I think I need some time to reflect on this further. Can we end the session for now and discuss this at a later time?
Counselor: Of course, it's important to take time to reflect. When you're ready, we can revisit this conversation. Just know that I'm here to support you whenever you're ready to talk. Take care.

Question: Should the conversation be concluded?
Response:
Conversation State: The client expresses a desire to pause the conversation to reflect further on the topic of reducing alcohol consumption, requesting to end the current session and continue the discussion at a later time.
End or Not: Yes


## Example 4
Conversation Snippet:
Client: I understand that reducing my alcohol consumption could benefit my sports performance.
Counselor: It's great that you're considering this change. What are some reasons that make you think reducing alcohol could benefit your sports performance?
Client: I understand that reducing my alcohol intake could have a positive impact on my sports performance. I see how it could help me stay focused and perform better on the field.
Counselor: It sounds like you're considering the potential benefits of cutting back on alcohol to enhance your sports performance. What are your thoughts on taking some steps in that direction?
Client: Thank you for sharing your insights.
Counselor: You're welcome. It's great to hear your thoughts on this. What are some reasons you might consider cutting back on your alcohol consumption to support your sports goals?
Client: Thank you for your guidance. I understand the potential benefits of reducing alcohol consumption to improve my sports performance.

Question: Should the conversation be concluded?
Response:
Conversation State: The Client acknowledges the benefits of reducing alcohol consumption but does not explicitly agree to any plan or commit to specific actions. The Counselor continues to engage, discussing the benefits and potential steps forward.
End or Not: No


Here is a new Conversation Snippet:
[@context]

Question: Should the conversation be concluded?
"""
    user_prompt = user_prompt.replace('[@context]', "\n".join(context[-5:]))
    response = get_precise_response([{'role': 'user', 'content': user_prompt}])
    if response and 'yes' in response.lower():
        return True
    return False

class Env:
    def __init__(self,
                 client,
                 counselor,
                 max_turns=20,
                 initial_context=["Counselor: Hello. How are you?", "Client: I am good. What about you?"],
                 output_file=None):
        self.client = client
        self.counselor = counselor
        self.conversation = copy.deepcopy(initial_context)
        self.max_turns = max_turns
        if output_file:
            self.output_file = open(output_file, 'w')
            for context in self.conversation:
                self.output_file.write(context + '\n')
        else:
            self.output_file = None
            for context in self.conversation:
                print(context)
    
    def output(self, utterance):
        if self.output_file:
            self.output_file.write(utterance + '\n')
        else:
            print(utterance)
        
    def clean_utterance(self, utterance):
        utterance = re.sub(r'\[.*?\]', '', utterance)
        return utterance
    
    def interact(self):
        for _ in range(self.max_turns):
            counselor_response = self.counselor.reply()
            self.output(counselor_response)
            counselor_response = self.clean_utterance(counselor_response)
            self.client.receive(counselor_response)
            self.conversation.append(counselor_response)
            if (heuristic_moderator(self.conversation)) or (_ > 20 and moderator(self.conversation)):
                break
            client_response = self.client.reply()
            if 'Terminate' in client_response:
                self.output(client_response)
                break
            self.output(client_response)
            client_response = self.clean_utterance(client_response)
            self.counselor.receive(client_response)
            self.conversation.append(client_response)
            if (heuristic_moderator(self.conversation)) or (_ > 20 and moderator(self.conversation)):
                break
