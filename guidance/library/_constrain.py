import guidance


class Constraint:

    def __init__(self):
        pass

    def set_context(self, prefix, lm):
        pass

    def valid_next_chars(self):  ## can return nothing if generation is complete
        pass

    def set_next_chars(self, c): ## returns a new immutable Constraint object with an updated internal state, if state is needed
        return self


@guidance
def constrain(lm, name="value", constraint=None, suffix="", logprobs=None, max_tokens=1000, list_append=False):
    ''' Select a value from a list of choices.

    Parameters
    ----------
    name : str
        The name of the variable to set with the value generated under constraints.
    constraint : Constraint
        A Constraint object that .
    suffix : str
        An optional suffix to append to the selected value. Passing the next string as a suffix allows the select
        statement to better differentiate between options that depend on the final token (and hence on a token that may overlap the following text).
    list_append : bool
        Whether to append the generated value to a list stored in the variable. If set to True, the variable
        must be a list, and the generated value will be appended to the list.
    '''
    
    assert constraint is not None, "You must provide a constraint!"

    ### EMK: We will give the suffix to the constraint provider when asking for the next token
    ###      When the constraint provider reaches the end of an option/program/etc, it will append the suffix    
    def recursive_getnext_valid_token(constraint, validstr=""):
        ret = []
        # TODO - this logic assumes that, if 'abc' is a valid token, that 'ab' must also be a valid token.  Not sure this is true. If not, the answer is to just use the underlying token mapping directly.
        next_valid_chars = constraint.valid_next_chars()
        if len(next_valid_chars) > 0:
            for c in next_valid_chars:
                valid_tokens = lm.get_encoded(str(validstr + c))
                if( len(valid_tokens) > 1):
                    #print(f"valid token: {valid_tokens[0]}")
                    ret.append(valid_tokens[0])
                else:
                    # NOTE: adding the abbreviated token valid_tokens[0] when a longer one is available doesn't seem to work correctly ? TODO why
                    next_constraint = constraint.set_next_chars(c)
                    recursive_valid_tokens = recursive_getnext_valid_token(next_constraint, validstr + c)
                    if( len(recursive_valid_tokens) > 0 ):
                        ret.extend(recursive_valid_tokens)
                    else: 
                        # if there are no valid tokens after this one, then our greedy search for the next token is done
                        # TODO - we should be doing this not only when recursive_valid_tokens is empty, but whenever
                        #        not generating a next token is valid.  Ie., we could continue, or we could stop...
                        # TODO - integrate this with the suffix logic TODO below 
                        ret.append(valid_tokens[0])
            return set(ret)
        else:
            # TODO pull characters from suffix.
            return []

    def gen_next_token(current_prefix, valid_next_tokens):
        logit_bias = {}
        for token in valid_next_tokens:
            logit_bias[token] = 100

        #print(f"Current prefix = {current_prefix}")
        gen_obj = lm.get_endpoint_session()(
            current_prefix, # TODO: perhaps we should allow passing of token ids directly? (this could allow us to avoid retokenizing the whole prefix many times)
            max_tokens=1,
            logit_bias=logit_bias,
            logprobs=len(logit_bias),
            cache_seed=0,
            stream=False,
            token_healing=False # we manage token boundary healing ourselves for this function
        )
        gen_obj = gen_obj["choices"][0] # get the first choice (we only asked for one)

        # TODO - is setting logit_biases sufficient? Or do we still need to go through the gen_obj and ensure that returned tokens meet our constraints?

        return gen_obj 

    prefix = str(lm)
    gen = prefix
    current_constraint = constraint

    # TODO extend this to do a beam search, or if we have a small cheap model, we could use it for an A* search?
    for i in range(max_tokens):
        next_valid_tokens = recursive_getnext_valid_token(current_constraint)
        if( len(next_valid_tokens) == 0):
            # the only way to get here is if the constraint provider has reached the end of an option/program/etc
            # TODO add ways for the LLM to end early, generate the suffix, etc.
            break

        next_gen = gen_next_token(gen, next_valid_tokens) 
        next_token = next_gen["text"]
        next_chars = next_token # the token is returned as characters, not as a token id
        gen += next_chars
        current_constraint = current_constraint.set_next_chars(next_chars)

    gen = gen[len(prefix):]
    lm[name] = gen

    # see if we are appending to a list or not
    if list_append:
        value_list = lm.get(name, [])
        value_list.append(gen)
        lm[name] =  value_list
    else:
        lm[name] = gen

    return lm.append("<||_html:<span style='background-color: rgba(0, 165, 0, 0.25)'>_||>" + gen + "<||_html:</span>_||>" + suffix)
    
    