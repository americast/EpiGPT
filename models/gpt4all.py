from data.serialize import serialize_arr, SerializerSettings
from gpt4all import GPT4All
import numpy as np
import tiktoken
from jax import grad,vmap

def tokenize_fn(str, model):
    """
    Retrieve the token IDs for a string for a specific GPT model.

    Args:
        str (list of str): str to be tokenized.
        model (str): Name of the LLM model.

    Returns:
        list of int: List of corresponding token IDs.
    """
    encoding = tiktoken.encoding_for_model(model)
    return encoding.encode(str)

def gpt4all_completion_fn(model, input_str, steps, settings, num_samples, temp):
    """
    Generate text completions from GPT4All.

    Args:
        model (str): Name of the GPT-3 model to use.
        input_str (str): Serialized input time series data.
        steps (int): Number of time steps to predict.
        settings (SerializerSettings): Serialization settings.
        num_samples (int): Number of completions to generate.
        temp (float): Temperature for sampling.

    Returns:
        list of str: List of generated samples.
    """
    
    model = GPT4All(model)

    avg_tokens_per_step = np.max([len(x) for x in input_str.split(settings.time_sep)]) + 1
    max_tokens = int(avg_tokens_per_step*(steps+1))
    # define logit bias to prevent GPT-3 from producing unwanted tokens
    logit_bias = {}
    allowed_tokens = [settings.bit_sep + str(i) for i in range(settings.base)] 
    allowed_tokens += [settings.time_sep, settings.plus_sign, settings.minus_sign]
    allowed_tokens = [t for t in allowed_tokens if len(t) > 0] # remove empty tokens like an implicit plus sign
    # if (model not in ['gpt-3.5-turbo','gpt-4']): # logit bias not supported for chat models
    #     logit_bias = {id: 30 for id in get_allowed_ids(allowed_tokens, model)}
    # if model in ['gpt-3.5-turbo','gpt-4']:
    outputs = []
    for i in range(num_samples):
        output = model.generate("You are a helpful assistant that performs time series predictions. The user will provide a sequence and you will predict the remaining sequence. The sequence is represented by decimal strings separated by commas. Please continue the following sequence without producing any additional text. Do not say anything like 'the next terms in the sequence are', just return the numbers. Sequence:\n "+input_str, max_tokens=max_tokens)
        outputs.append(output)


    # output = model.generate("The capital of France is ", max_tokens=3)

    # response = openai.ChatCompletion.create(
    #     model=model,
    #     messages=[
    #             {"role": "system", "content": chatgpt_sys_message},
    #             {"role": "user", "content": extra_input+input_str+settings.time_sep}
    #         ],
    #     max_tokens=int(avg_tokens_per_step*steps), 
    #     temperature=temp,
    #     logit_bias=logit_bias,
    #     n=num_samples,
    # )
    try:
        for i, output in enumerate(outputs):
            idx_here = [n for (n, e) in enumerate(output) if e == ','][int(steps)-1]
            outputs[i] = output[:idx_here]
    except:
        import pudb; pu.db

    return outputs
    # else:
    #     response = openai.Completion.create(
    #         model=model,
    #         prompt=input_str, 
    #         max_tokens=int(avg_tokens_per_step*steps), 
    #         temperature=temp,
    #         logit_bias=logit_bias,
    #         n=num_samples
    #     )
    #     return [choice.text for choice in response.choices]
    