"""VLM Helper Functions."""
import io
import base64
import numpy as np
from openai import OpenAI
from anthropic import Anthropic
from termcolor import colored
import PIL.Image
import google.generativeai as genai
import bumble.utils.utils  as U

class Gemini:
    def __init__(self, openai_api_key, model_name):
        self.model_name = model_name
        genai.configure(api_key=openai_api_key)
        assert model_name in ['gemini-1.5-pro', 'gemini-1.5-flash']
        self.model = genai.GenerativeModel(model_name=model_name)

    def create_msg_history(self, history_instruction, history_desc, history_model_analysis, history_imgs):
        messages = []
        messages.append(history_instruction)

        # Add history of descriptions, imgs, and model analysis. Model analysis is from the assistant.
        assert len(history_desc) == len(history_imgs) == len(history_model_analysis)

        for desc, img, model_analysis in zip(history_desc, history_imgs, history_model_analysis):
            user_content = []
            # user_content.append({'type': 'text', 'text': desc})
            messages.append(desc)
            if img is not None:
                pil_image = U.decode_image(img)
                pil_image = PIL.Image.fromarray(pil_image)
                messages.append(pil_image)

            messages.append(model_analysis)

        return messages

    def query(self, instruction, prompt_seq, temperature=0, max_tokens=2048, history=None, return_full_chat=False):
        """Queries GPT-4V."""
        TODO
        messages = []
        messages.append(instruction)

        if history:
            print("history is provided")
            messages.extend(history)

        # prompt_seq is a list of strings and np.ndarrays
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                # content.append({'type': 'text', 'text': elem})
                content.append(elem)
            elif isinstance(elem, np.ndarray):
                pil_image = U.decode_image(elem)
                pil_image = PIL.Image.fromarray(pil_image)
                content.append(pil_image)

        messages.extend(content)

        error = False
        retry = True
        response = None
        while retry:
            try:
                response = self.model.generate_content(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                retry = False
                error = False
            except Exception as e:
                print(f"WARNING!!! GEMINI CHAT FAILED: {str(e)}")
                # import ipdb; ipdb.set_trace()
                error = True
                pass

            if error:
                retry = U.confirm_user(True, 'Press y to retry and n to skip')
            else:
                retry = False

        if error:
            print(colored('Error in querying Anthropic.', 'red'))
            return ""

        return response.text

class Ant:
    def __init__(self, openai_api_key, model_name):
        self.model_name = model_name
        self.client = Anthropic(api_key=openai_api_key)
        assert model_name in ['claude-3-5-sonnet-20240620', 'claude-3-opus-20240229', 'claude-3-sonnet-20240229', 'claude-3-haiku-20240307']

    def create_msg_history(self, history_instruction, history_desc, history_model_analysis, history_imgs):
        messages = []
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': history_instruction}]})
        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'I will pay attention to the history provided below..'}]})

        # Add history of descriptions, imgs, and model analysis. Model analysis is from the assistant.
        assert len(history_desc) == len(history_imgs) == len(history_model_analysis)

        for desc, img, model_analysis in zip(history_desc, history_imgs, history_model_analysis):
            user_content = []
            user_content.append({'type': 'text', 'text': desc})
            if img is not None:
                base64_image_str = base64.b64encode(img).decode('utf-8')
                image_content = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64_image_str}}
                user_content.append(image_content)

            messages.append({'role': 'user', 'content': user_content})
            model_content = []
            model_content.append({'type': 'text', 'text': model_analysis})
            messages.append({'role': 'assistant', 'content': model_content})

        return messages

    def query(self, instruction, prompt_seq, temperature=0, max_tokens=2048, history=None, return_full_chat=False):
        """Queries GPT-4V."""
        messages = []
        # Add instructions as user which are in plain text
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})

        messages.append({'role': 'assistant', 'content': [{'type': 'text', 'text': 'sounds good, let me help you with that.'}]})

        if history:
            print("history is provided")
            messages.extend(history)

        # prompt_seq is a list of strings and np.ndarrays
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, np.ndarray):
                base64_image_str = base64.b64encode(elem).decode('utf-8')
                image_content = {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': base64_image_str}}
                content.append(image_content)
        messages.append({'role': 'user', 'content': content})

        error = False
        retry = True
        response = None
        while retry:
            try:
                response = self.client.messages.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=max_tokens
                )
                retry = False
                error = False
            except Exception as e:
                print(f"WARNING!!! ANTHROPIC CHAT FAILED: {str(e)}")
                error = True
                pass

            if error:
                retry = U.confirm_user(True, 'Press y to retry and n to skip')
            else:
                retry = False

        if error:
            print(colored('Error in querying Anthropic.', 'red'))
            return ""

        if return_full_chat:
            content = response.content[0]
            messages.append({'role': 'assistant', 'content': content})
            return response.content[0].text, messages

        return response.content[0].text

class GPT4V:
    """GPT4V VLM."""

    def __init__(self, openai_api_key, model_name='gpt-4o-2024-05-13'):
        self.model_name = model_name
        self.client = OpenAI(api_key=openai_api_key)

    def create_msg_history(self, history_instruction, history_desc, history_model_analysis, history_imgs):
        messages = []
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': history_instruction}]})

        # Add history of descriptions, imgs, and model analysis. Model analysis is from the assistant.
        assert len(history_desc) == len(history_imgs) == len(history_model_analysis)

        for desc, img, model_analysis in zip(history_desc, history_imgs, history_model_analysis):
            user_content = []
            user_content.append({'type': 'text', 'text': desc})
            if img is not None:
                base64_image_str = base64.b64encode(img).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                user_content.append({'type': 'image_url', 'image_url': {'url': image_url}})
            messages.append({'role': 'user', 'content': user_content})

            model_content = []
            model_content.append({'type': 'text', 'text': model_analysis})
            messages.append({'role': 'assistant', 'content': model_content})

        return messages

    def query(self, instruction, prompt_seq, temperature=0, max_tokens=2048, history=None, return_full_chat=False):
        """Queries GPT-4V."""
        messages = []
        # Add instructions as user which are in plain text
        messages.append({'role': 'user', 'content': [{'type': 'text', 'text': instruction}]})

        if history:
            print("history is provided")
            messages.extend(history)

        # prompt_seq is a list of strings and np.ndarrays
        content = []
        for elem in prompt_seq:
            if isinstance(elem, str):
                content.append({'type': 'text', 'text': elem})
            elif isinstance(elem, np.ndarray):
                base64_image_str = base64.b64encode(elem).decode('utf-8')
                image_url = f'data:image/jpeg;base64,{base64_image_str}'
                content.append({'type': 'image_url', 'image_url': {'url': image_url}})
        messages.append({'role': 'user', 'content': content})

        # DEBUG:
        # from bumble.utils.utils import plot_gpt_chats; plot_gpt_chats([messages], save_key='test', save_dir='temp')
        # import ipdb; ipdb.set_trace()
        error = False
        retry = True
        response = None
        while retry:
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens
                )
                retry = False
                error = False
            except Exception as e:
                print(f"WARNING!!! OPENAI CHAT FAILED: {str(e)}")
                # import ipdb; ipdb.set_trace()
                error = True
                pass

            if error:
                retry = U.confirm_user(True, 'Press y to retry and n to skip')
            else:
                retry = False

        if error:
            print(colored('Error in querying OpenAI.', 'red'))
            return ""

        if return_full_chat:
            messages.append({'role': 'assistant', 'content': response.choices[0].message.content})
            return response.choices[0].message.content, messages

        return response.choices[0].message.content
