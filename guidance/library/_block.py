import types

import guidance
from guidance import TextRange

@guidance
def block(lm, name=None, open_text="", close_text="", hidden=False):
    offset = len(lm._state) + len(open_text)

    def __enter__(lm):
        return lm.append(open_text)
    
    def __exit__(lm, exc_type, exc_value, traceback):
        _rec_close(lm, open_text, close_text, hidden, text_name=name, text_offset=offset)
    
    # bind the enter and exit methods
    lm.instance__enter__.append(types.MethodType(__enter__, lm))
    lm.instance__exit__.append(types.MethodType(__exit__, lm))

    return lm

# @guidance
# def hidden(lm):
#     return lm.block(hidden=True)

def _rec_close(lm, open_text, close_text, hidden, text_name=None, text_offset=0):
    if text_name is not None:
        lm[text_name] = TextRange(text_offset, len(lm), lm)
    if close_text != "":
        lm += close_text
    if hidden:
        lm._reset(text_offset, clear_variables=False)
    
    for child in lm._children:
        _rec_close(child, close_text, text_name=text_name, text_offset=text_offset)