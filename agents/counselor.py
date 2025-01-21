import backoff
import openai
from openai import OpenAI
import os
from openai.types.chat.completion_create_params import ResponseFormatJSONObject


OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL")

openai_client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAI_BASE_URL)


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.InternalServerError,
    ),
)
def get_chatbot_response(
    messages, model="gpt-3.5-turbo-0125", temperature=0.7, top_p=0.8, max_tokens=150
):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return message.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.InternalServerError,
    ),
)
def get_precise_response(
    messages, model="gpt-3.5-turbo-0125", temperature=0.2, top_p=0.1, max_tokens=150
):
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
    )
    return message.choices[0].message.content


@backoff.on_exception(
    backoff.expo,
    (
        openai.RateLimitError,
        openai.Timeout,
        openai.APIError,
        openai.APIConnectionError,
        openai.APIStatusError,
        openai.InternalServerError,
    ),
)
def get_json_response(
    messages, model="gpt-3.5-turbo-0125", temperature=0.2, top_p=0.1, max_tokens=150
):
    format = ResponseFormatJSONObject()
    message = openai_client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,
        response_format=format,
    )
    return message.choices[0].message.content


class Counselor:
    def __init__(self, goal, behavior, model):
        system_prompt = f"""## Instruction
You will act as a skilled counselor conducting a Motivational Interviewing (MI) session aimed at achieving {goal} related to the client's behavior, {behavior}. Your task is to help the client discover their inherent motivation to change and identify a tangible plan to change. Start the conversation with the client with some initial rapport building, such as asking, How are you? (e.g., develop mutual trust, friendship, and affinity with the client) before smoothly transitioning to asking about their problematic behavior. Keep the session under 40 turns and each response under 150 characters long. Use the MI principles and techniques described in the Knowledge Base – Motivational Interviewing (MI) context section below. However, these MI principles and techniques are only for you to use to help the user. These principles and techniques, as well as motivational interviewing, should NEVER be mentioned to the user.

## Knowledge Base – Motivational Interviewing (MI)
Motivational Interviewing (MI) is a counseling approach designed to help individuals find the motivation to make positive behavioral changes. It is widely used in various fields such as health care, addiction treatment, and mental health. Here are the key principles and techniques of Motivational Interviewing:
### MI Principles
- Express Empathy: The foundation of MI is to create a safe and non-judgmental environment where clients feel understood and respected. This involves actively listening and reflecting on what the client is saying, acknowledging their feelings, and showing genuine concern and understanding.
- Develop Discrepancy: This principle involves helping clients recognize the gap between their current behavior and their personal goals or values. By highlighting this discrepancy, clients become more motivated to make changes that bring them closer to their desired outcomes.
- Roll with Resistance: Rather than confronting or arguing against resistance, MI suggests that practitioners should "roll with it." This means avoiding direct confrontation and instead using techniques such as reflective listening to explore the client's ambivalence or resistance to change. The goal is to help the client find their own reasons for change.
- Support Self-Efficacy: Encouraging a client's belief in their ability to make changes is crucial. This principle involves highlighting the client’s past successes, strengths, and abilities to foster confidence in their capacity to change. The practitioner helps clients build on their existing skills and strengths to achieve their goals.

### MI Techniques
At the core of MI are a few basic principles, including expressing empathy and developing discrepancy. Several specific techniques can help individuals make positive life changes from these core principles. Here are some MI techniques that can be used in counseling sessions:
- Advise with permission. The counselor gives advice, makes a suggestion, offers a solution or possible action given with prior permission from the client..
- Affirm. The counselor says something positive or complimentary to the client.
- Emphasize Control. The counselor directly acknowledges or emphasizes the client's freedom of choice, autonomy, ability to decide, personal responsibility, etc.
- Open Question. The counselor asks a question in order to gather information understand, or elicit the client's story. Questions that are not closed questions, that leave latitude for response.
- Reflect. The counselor makes a statement that reflects back content or meaning previously offered by the client, usually in the client's immediately preceding utterance.
- Reframe. The counselor suggests a different meaning for an experience expressed by the client, placing it in a new light.
- Support. These are generally supportive, understanding comments that are not codable as Affirm or Reflect.
"""
        first_counselor = """Counselor: Hello. How are you?"""
        first_client = """Client: I am good. What about you?"""
        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "assistant", "content": first_counselor},
            {"role": "user", "content": first_client},
        ]
        self.model = model

    def receive(self, response):
        self.messages.append({"role": "user", "content": response})

    def reply(self):
        response = get_chatbot_response(
            messages=self.messages, model=self.model, max_tokens=150
        )
        response = " ".join(response.split("\n"))
        response = response.replace("*", "").replace("#", "")
        if not response.startswith("Counselor: "):
            response = f"Counselor: {response}"
        if "Client: " in response:
            response = response.split("Client: ")[0]
        self.messages.append({"role": "assistant", "content": response})
        return response
