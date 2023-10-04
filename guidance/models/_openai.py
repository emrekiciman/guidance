import re

import guidance.endpoints
from ._model import Model, Chat


chat_model_pattern = r'^(gpt-3\.5-turbo|gpt-4)(-\d+k)?(-\d{4})?$'

class OpenAI(Model):
    def __init__(self, model, caching=True, **endpoint_kwargs):

        # subclass to OpenAIChat if model is chat
        if re.match(chat_model_pattern, model) and self.__class__ is OpenAI:
            self.__class__ = OpenAIChat
            OpenAIChat.__init__(self, model=model, caching=caching)
            return

        # standard init
        super().__init__(model)
        self.model = model

        self.endpoint = guidance.endpoints.OpenAI(model, **endpoint_kwargs)
        self._endpoint_session = self.endpoint.session()

    # TODO add logit_bias and log_probs to get this to work with select() but that seems counter to the way the new API seems to be layered
    def __call__(self, pattern, max_tokens=100, n=1, top_p=1, temperature=0.0, ensure_bos_token=True, **endpoint_session_kwargs):
        prompt = str(self)

        # TODO bos_token not handled
        #if ensure_bos_token and not prompt.startswith(self.bos_token):
        #    prompt = self.bos_token + prompt
        
        # TODO any logic for stop or stop_regex? 
        #    from prior API - assert stop_regex is None or stream, "We can only support stop_regex for the OpenAI API when stream=True!"
        #    from prior API - assert stop_regex is None or n == 1, "We don't yet support stop_regex combined with n > 1 with the OpenAI API!"
        # TODO any logic for logprobs, echo, logit_bias, token_healing, or pattern?
        #    from prior API - assert token_healing is None or token_healing is False, "The OpenAI API does not yet support token healing! Please either switch to an endpoint that does, or don't use the `token_healing` argument to `gen`."
        # TODO should temperature default to 0/0? or to the global set when the LLM was created?
        # TODO any cache management we want to do here, or is that within the endpoint or endpoint_session?
        return self._endpoint_session(prompt, temperature=temperature, n=n, max_tokens=max_tokens, top_p=top_p, **endpoint_session_kwargs)


# async def __call__(self, prompt, stop=None, stop_regex=None, temperature=None, n=1, max_tokens=1000, logprobs=None,
#                       top_p=1.0, echo=False, logit_bias=None, token_healing=None, pattern=None, stream=None,
#                       cache_seed=0, caching=None, **completion_kwargs):

    def get_encoded(self, s):
        return self.endpoint.encode(s)
    
    def get_decoded(self, s):
        return self.endpoint.decode(s)
    
    def get_id_to_token(self, id):
        return self.endpoint.id_to_token(id)

    def get_token_to_id(self, token):
        return self.endpoint.token_to_id(token)



class OpenAIChat(OpenAI, Chat):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def tool_def(self, *args, **kwargs):
        lm = self + "<||_html:<span style='background-color: rgba(93, 63, 211, 0.15)'>_||>"
        lm = OpenAI.tool_def(lm, *args, **kwargs)
        return lm + "<||_html:</span>_||>"