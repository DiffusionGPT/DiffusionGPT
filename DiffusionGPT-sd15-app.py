# coding: utf-8
import os
import gradio as gr
import random
import torch
import cv2
import re
import uuid
import json
import pickle
from PIL import Image, ImageDraw, ImageOps, ImageFont
import math
import numpy as np
import argparse
import inspect
import tempfile
# import subprocess

from diffusers import StableDiffusionPipeline, StableDiffusionInpaintPipeline, StableDiffusionInstructPix2PixPipeline
from diffusers import EulerAncestralDiscreteScheduler, PNDMScheduler
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, DiffusionPipeline, UniPCMultistepScheduler

from langchain.agents.initialize import initialize_agent
from langchain.agents.tools import Tool
from langchain.chains.conversation.memory import ConversationBufferMemory
from langchain.llms.openai import OpenAI
from langchain.llms import AzureOpenAI


from sentence_transformers import SentenceTransformer
from compel import Compel


PREFIX = """DiffusionGPT is designed to be able to assist users in generating high-quality images.

Human may provide some text prompts to DiffusionGPT. The input prompts will be analyzed by DiffusionGPT to select the most suitable generative model for generating images.

Overall, DiffusionGPT is a powerful image generation system that can assist in processing various forms of textual input and match them with the most suitable generative model to accomplish the generation task.

TOOLS:
------

DiffusionGPT  has access to the following tools:"""

FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
{ai_prefix}: [your response here]
```
"""

SUFFIX = """You are very strict to the filename correctness and will never fake a file name if it does not exist.
You will remember to provide the image file name loyally if it's provided in the last tool observation.

Begin!

Previous conversation history:
{chat_history}

New input: {input}
DiffusionGPT must use tools to observe images rather than imagination.
The thoughts and observations are only visible for DiffusionGPT, DiffusionGPT should remember to repeat important information in the final response for Human. 
Thought: Do I need to use a tool? {agent_scratchpad} Let's think step by step.
"""


TOT_PROMPTS = """Identify and behave as five different experts that are appropriate to select one element from the input list that best matches the input prompt. 
All experts will write down the selection result, then share it with the group. 

You then analyze all 5 analyses and output the consensus selected element or your best guess matched element. 
The final selection output MUST be the same as the TEMPLATE:
TEMPLATE:
```
Selected: [the selected word]
```

Input list: {search_list}

Input prompt: {input}
"""


PROMPT_PARSE_PROMPTS = """Given the user input text.
Please judge the paradigm of the input text, and then recognize the main string of text prompts according to the corresponding form. 
The output must be same as the TEMPlATE:

TEMPLATE:
```
Prompts: [the output prompts]
```

For instance:
1. Input: A dog
   Prompts: A dog 
2. Input: generate an image of a dog
   Prompts: an image of a dog
3. Input: I want to see a beach
   Prompts: a beach
4. Input: If you give me a toy, I will laugh very happily
   Prompts: a toy and a laugh face

Input: {inputs}

"""


TREE_OF_MODEL_PROMPT_SUBJECT = """ You are an information analyst who can analyze and abstract a set of words to abstract some representation categories.
Below is a template that can represent the abstracted categories in Subject Dimension belonging to concrete noun:

TEMPLATE:
```
Categories:
- [Subject]
- [Subject]
- ...
```

You MUST abstract the categories in a highly abstract manner only from Subject Dimension and ensure the whole number of categories are fewer than 5.
Then, You MUST remove the Style-related categories.

Please output the categories following the format of TEMPLATE. 

Input: {input}

"""

TREE_OF_MODEL_PROMPT_STYLE = """ You are an information analyst who can analyze and summarize a set of words to abstract some representation categories.
Below is a template that can represent the the abstracted categories in Style Dimension:

TEMPLATE:
```
Categories:
- [Style]
- [Style]
- ...
```

You MUST abstract the categories in a highly abstract manner from only Style dimension and ensure the whole number of categories are fewer than 8.

Please output the Categories following the format of TEMPLATE.

Input: {input}

"""

TREE_OF_MODEL_PROMPT_ = """ You are an information analyst who can create a Knowledge Tree according to the input categories.
Below is a knowledge tree template:

TEMPLATE:
```
Knowledge Tree:
- [Subject]
  - [Style]
  - ...
- [Subject]
- ...
```

You MUST place the each Style category as subcategory under the Subject categories based on whether it can be well matched with a specific subject category to form a reasonable scene.

Please output the categories following the format of TEMPLATE. 

Subject Input: {subject}

Style Input: {style}

"""


TREE_OF_MODEL_PROMPT_ADD_MODELS = """ You are an information analyst who can add some input models to an input knowledge tree according to the similarity of the model tags and the categories of the knowledge tree.

You need to place each input model into the appropriate subcategory on the tree, one by one.
You MUST keep the original content of the knowledge tree.  


Please output the final knowledge tree.

Knowledge Tree Input: {tree}

Models Input: {models}

Model Tags Input: {model_tags}

"""


os.makedirs('image', exist_ok=True)


from langchain.llms.base import LLM

from langchain import PromptTemplate, HuggingFaceHub
from langchain.llms import HuggingFacePipeline

from transformers import AutoTokenizer, AutoModelForCausalLM


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    return seed


def prompts(name, description):
    def decorator(func):
        func.name = name
        func.description = description
        return func

    return decorator


def cut_dialogue_history(history_memory, keep_last_n_words=500):
    if history_memory is None or len(history_memory) == 0:
        return history_memory
    tokens = history_memory.split()
    n_tokens = len(tokens)
    
    if n_tokens < keep_last_n_words:
        return history_memory
    paragraphs = history_memory.split('\n')
    last_n_tokens = n_tokens
    while last_n_tokens >= keep_last_n_words:
        last_n_tokens -= len(paragraphs[0].split(' '))
        paragraphs = paragraphs[1:]
    return '\n' + '\n'.join(paragraphs)


class Text2Image:
    def __init__(self, device):
        print(f"Initializing Text2Image to {device}")
        self.device = device
        self.torch_dtype = torch.float16 if 'cuda' in device else torch.float32

        self.llm = OpenAI(temperature=0)

        if not os.path.exists('model_tree_tot_sd15.json'):
            with open('model_data_sd15.json', 'r') as f:
                self.model_data_all = json.load(f)
                
            model_tags = {model["model_name"]: model["tag"] for model in self.model_data_all}

            model_tree = self.build_tree(model_tags)

            model_all_data = {model["model_name"].split(".")[0]: model for model in self.model_data_all}

            save_model_tree = {}
            for cate_name, sub_category in model_tree.items():
                cate_name = cate_name.lower()
                temp_category = {}

                if "Universal" not in sub_category:
                    temp_category["universal"] = [model_all_data["majicmixRealistic_v6"], model_all_data["FilmVelvia2"]]

                for sec_cate_name, sub_sub_cates in sub_category.items():
                    sec_cate_name = sec_cate_name.lower()
                    temp_model_list = []
                    
                    for model_name in sub_sub_cates:
                        model_name = model_name.strip()
                        lower_name = model_name[0].lower() + model_name[1:]
                        if model_name in model_all_data:
                            temp_model_list.append(model_all_data[model_name])
                        elif lower_name in model_all_data:
                            temp_model_list.append(model_all_data[lower_name])

                        
                    temp_category[sec_cate_name] = temp_model_list

                save_model_tree[cate_name] = temp_category

            # write in json
            json_data = json.dumps(save_model_tree, indent=2)
            with open('model_tree_tot_sd15.json', 'w') as f:
                f.write(json_data)
                f.close()
                

        with open('model_tree_tot_sd15.json', 'r') as f:
            self.model_data = json.load(f)

        with open('model_data_sd15.json', 'r') as f:
            self.model_all_data = json.load(f)
            self.model_all_data = {model["model_name"]:model for model in self.model_all_data}

        # Advantage databases with human feedback
        with open('./VectorDB_HF/prompt_embed_st.pickle', 'rb') as f:
            self.pt_pairs = pickle.load(f)

        with open('./VectorDB_HF/prompt2scores.json', 'r') as f:
            self.prompt2scores = json.load(f)

        self.st_model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')


    def build_tree(self, model_tags):
        tags_only = list(model_tags.values()) 
        model_names = list(model_tags.keys())

        prompts = TREE_OF_MODEL_PROMPT.format(input=tags_only)

        prompt1 = TREE_OF_MODEL_PROMPT_SUBJECT.format(input=tags_only)
        response1 = self.llm(prompt1)
        
        prompt2 = TREE_OF_MODEL_PROMPT_STYLE.format(input=tags_only)
        response2 = self.llm(prompt2)

        prompt_tree = TREE_OF_MODEL_PROMPT_.format(style=response2, subject=response1)
        response = self.llm(prompt_tree)
        
        tree = response.split("Knowledge Tree:")[1]
        
        model_names = [name.split(".")[0] for name in list(model_tags.keys())]
        
        prompts = TREE_OF_MODEL_PROMPT_ADD_MODELS.format(model_tags=model_tags, tree=tree, models=model_names)
        
        tree = self.llm(prompts)

        output = {}
        tree_list = tree.split("\n")
        for category in tree_list:
            if category == '':
                continue
            
            if category.startswith("- "):
                current_key = category[2:]
                output[current_key] = {}
            elif category.startswith("  - "):
                next_key = category[4:]
                output[current_key][next_key] = []
            elif category.startswith("    - "):
                output[current_key][next_key].append(category[6:])
        
        return output

    def prompt_parse(self, inputs):
        
        prompts = PROMPT_PARSE_PROMPTS.format(inputs=inputs)
        output = self.llm(prompts)
        output = output.split("Prompts:")[1]
        
        return output.strip() 

    def get_property(self, model_data):
        properties = []
        for model in model_data:
            name = "model_name:" + model["model_name"] + ", "
            tag = "tag:" + ",".join(model["tag"])
           
            prop = name + tag + "\n\n"
            properties.append(prop)
        return properties

    def search_one_matched(self, inputs, search_list):
        
        tot_prompts = TOT_PROMPTS.format(search_list=search_list, input=inputs)

        model_name = self.llm(tot_prompts)
        print(model_name)
        
        if "Selected:" in model_name:
            model_name = model_name.split("Selected:")[-1]
        
        for ch in [",", ";", "."]:
            if ch in model_name:
                model_name = model_name.split(ch)[0]
        model_name = model_name.strip().lower()

        return model_name


    def select_best_model_with_HF(self, inputs, model_space):

        text_embed = torch.Tensor(self.st_model.encode([inputs]) )
        text_embed /= text_embed.norm(dim=1, keepdim=True)

        similarity = text_embed @ self.pt_pairs['text_embeds'].T

        topk_idxs = similarity.topk(5).indices[0,:]
        
        topk_model_list = []
        model_names_of_tree = [model["model_name"].split(".")[0] for model in model_space]

        for idx, p in enumerate(topk_idxs):

            save_prompt_name = self.pt_pairs['prompts'][int(p)][:100].replace('\n','') 

            model_scores = self.prompt2scores[save_prompt_name]

            model_names = list(model_scores.keys())
            reward_scores = []
            for name, values in model_scores.items():
                reward_scores.append(values['image_reward'])

            reward_scores = torch.Tensor(reward_scores)
            topk_model_idx = reward_scores.topk(5).indices.tolist()
            topk_models = [model_names[i] for i in topk_model_idx]

            topk_model_list.append(topk_models)

        prompt1 = f"Please judge whether each name in this list {model_names_of_tree} has highly similar name in the list {topk_model_list}, if yes, output the similar model name, the output MUST be Template: Model: [model name, ...]"
        intersection_model = self.llm(prompt1)
        
        prompts = f"Please select one model name from the following model list {intersection_model} that has the highest frequency and top ranking according to the list {topk_model_list}.\n\n The output MUST be Template: Model: [model name]"
        selected_model = self.llm(prompts)
        selected_model = selected_model.split("Model:")[1]

        return selected_model.strip()

    def search_model_tree(self, inputs):
        search_space = self.model_data
        search_path = []
        
        while not isinstance(search_space, list):
            search_list = list(search_space.keys())
            name = self.search_one_matched(inputs, search_list)
            search_path.append(name)
            search_space = search_space[name]

        candidate_model_data = {}
        for model in search_space:
            candidate_model_data[model["model_name"]] = model
        
        model_properties = self.get_property(search_space)
        
        model_name_pre = self.select_best_model_with_HF(inputs, search_space)
        all_names = list(self.model_all_data.keys())
        all_names = [name + "\n" for name in all_names]
        
        prompts = f"Please according to the name of {model_name_pre} and select one element from the list bellow, and ensure the selected element MUST be the same as one of the list {all_names}."
        model_name = self.llm(prompts).strip('\n')

        
        if model_name not in self.model_all_data:
            model_name = model_name[0].lower() + model_name[1:]
        selected_model = self.model_all_data[model_name]
        
        search_path.append(model_name)
        
        return search_path, selected_model


    def prompt_entension(self, inputs, model):
        example_prompt = model["example_prompts"][0]
        example_n_prompt = model["negtive_prompts"][0]
        
        prompts = f"Here is a paragraph describing an image. " \
                  f"{inputs}. " \
                  f"Please follow the sentence pattern of the example to expand the description of the input paragraph. The output MUST preserve the contents of the input paragraph. Example: {example_prompt}."

        extended_prompt = self.llm(prompts)
        
        return extended_prompt, example_n_prompt

    def match_id(self, model):

        model_names = list(self.model_all_data.keys())
        
        prompts = f"Here is a model. " \
                  f"{model}. " \
                  f"Please select the model name that best matches the given model from the model name list {model_names}. " \
                  f"The output must be the same as the word in the list. "

        matched_name = self.llm(prompts)
        matched_name = matched_name[2:]

        return matched_name


    @prompts(name="Generate Image From User Input Text", 
             description="always useful to generate an image from a user input text and save it to a file. "
                         "The input to this tool MUST be the whole user input text.")
                       

    def inference(self, inputs):
        # Prompt Parse
        original_input = inputs
        inputs = self.prompt_parse(inputs)

        # search model tree
        model_select_path, selected_model_data = self.search_model_tree(inputs)
        print("Selected model path:", model_select_path)
        print("Selected model name:", selected_model_data["model_name"])
        
        model_name = selected_model_data["model_name"]
        model_type = selected_model_data["model_type"]
        
        
        # load model ckpt
        self.pipe_prior = None
        if "checkpoint" in model_type:
            
            if model_name in list(self.model_all_data.keys()):
                model_id = "./checkpoints/" + model_name
                self.pipe = StableDiffusionPipeline.from_single_file(model_id,  torch_dtype=self.torch_dtype)

            self.pipe.to(self.device)

        # load model lora
        elif model_type == "lora":
            
            base_model = selected_model_data["resources_used"][0]
            base_model_name = self.match_id(base_model)
            base_model_name = "./checkpoints/" + base_model_name
            print(base_model_name)
            self.pipe = StableDiffusionPipeline.from_single_file(base_model_name, torch_dtype=self.torch_dtype)
            
            self.pipe.to(self.device)
            self.pipe.load_lora_weights("./checkpoints", weight_name=model_name)

        # Prompt Extension
        if selected_model_data["example_prompts"][0] != "None":
            prompt, n_prompt = self.prompt_entension(inputs, selected_model_data)
        else:
            prompt = inputs
            n_prompt = selected_model_data["negtive_prompts"][0]

        if n_prompt == "None":
            n_prompt = ""

        prompt_embeds = None
        
           
        ## compel for long text
        compel = Compel(tokenizer=self.pipe.tokenizer, text_encoder=self.pipe.text_encoder, truncate_long_prompts=False)

        prompt_list = list(prompt.strip().split("."))
        n_prompt_list = list(n_prompt.strip().split("."))
        prompt = f'{prompt_list}.and()'
        n_prompt = f'{n_prompt_list}.and()'
        
        prompt_embeds = compel.build_conditioning_tensor(prompt)
        negative_conditioning = compel.build_conditioning_tensor(n_prompt)
        [prompt_embeds, negative_conditioning] = compel.pad_conditioning_tensors_to_same_length([prompt_embeds, negative_conditioning])
        negative_prompt_embeds = negative_conditioning


        if prompt_embeds is not None:
            
            output_latents = self.pipe(prompt_embeds=prompt_embeds, negative_prompt_embeds=negative_prompt_embeds, num_images_per_prompt=1, output_type='latent').images
            with torch.no_grad():
                images = self.pipe.decode_latents(output_latents)
            images = self.pipe.numpy_to_pil(images)
        else:
            output_latents = self.pipe(prompt, negative_prompt=n_prompt, height=512, width=512, num_images_per_prompt=1, output_type='latent').images
            with torch.no_grad():
                images = self.pipe.decode_latents(output_latents)
            images = self.pipe.numpy_to_pil(images)

        image_filename = os.path.join('image', f"{str(uuid.uuid4())[:8]}.png")
        
        output = images[0]
        output.save(image_filename)

        print(
            f"\nProcessed Text2Image, Input Text: {inputs}, Output Image: {image_filename}")
        return image_filename



class ConversationBot:
    def __init__(self, load_dict):
        print(f"Initializing DiffusionGPT, load_dict={load_dict}")
        
        self.models = {}
        # Load Basic Foundation Models
        for class_name, device in load_dict.items():
            self.models[class_name] = globals()[class_name](device=device)

        # Load Template Foundation Models
        for class_name, module in globals().items():
            if getattr(module, 'template_model', False):
                template_required_names = {k for k in inspect.signature(module.__init__).parameters.keys() if k!='self'}
                loaded_names = set([type(e).__name__ for e in self.models.values()])
                if template_required_names.issubset(loaded_names):
                    self.models[class_name] = globals()[class_name](
                        **{name: self.models[name] for name in template_required_names})
        
        print(f"All the Available Functions: {self.models}")

        self.tools = []
        for instance in self.models.values():
            for e in dir(instance):
                if e.startswith('inference'):
                    func = getattr(instance, e)
                    self.tools.append(Tool(name=func.name, description=func.description, func=func))
        self.llm = OpenAI(temperature=0)
        
        self.memory = ConversationBufferMemory(memory_key="chat_history", output_key='output')

    def init_agent(self, lang):
        self.memory.clear() #clear previous history
        
        place = "Enter text and press enter, or upload an image"
        label_clear = "Clear"
        
        self.agent = initialize_agent(
            self.tools,
            self.llm,
            agent="conversational-react-description",
            verbose=True,
            memory=self.memory,
            return_intermediate_steps=True,
            agent_kwargs={'prefix': PREFIX, 'format_instructions': FORMAT_INSTRUCTIONS,
                          'suffix': SUFFIX},
            handle_parsing_errors="Check your output and make sure it conforms!" )
        return gr.update(visible = True), gr.update(visible = False), gr.update(placeholder=place), gr.update(value=label_clear)

    def run_text(self, text, state):
        
        # self.agent.memory.buffer = cut_dialogue_history(self.agent.memory.buffer, keep_last_n_words=500)
        res = self.agent({"input": text.strip()})
        res['output'] = res['output'].replace("\\", "/")
        response = re.sub('(image/[-\w]*.png)', lambda m: f'![](file={m.group(0)})*{m.group(0)}*', res['output'])
        state = state + [(text, response)]
        print(f"\nProcessed run_text, Input text: {text}\n")
        return state, state


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--load', type=str, default="Text2Image_cuda:0")
    args = parser.parse_args()
    load_dict = {e.split('_')[0].strip(): e.split('_')[1].strip() for e in args.load.split(',')}

    def init_api(apikey):
        os.environ['OPENAI_API_KEY'] = apikey
        global bot
        bot = ConversationBot(load_dict=load_dict)
        bot.init_agent("English")
        print('set new api key:', apikey)
        return None

    init_api(apikey="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")
    def inference_warp(prompt):
        prompt = prompt.strip()
        global bot
        state = []
        _, state = bot.run_text(prompt, state)
        
        print('========>', str(state))
        
        pattern = r"\(file=(.*?)\)"
        matches = re.findall(pattern,  str(state))

        
        if matches:
            file_path = matches[0]
            print(file_path)
        

        image = Image.open(file_path)
        return image

    with gr.Blocks(css="#chatbot .overflow-y-auto{height:1000px}") as demo:
        state = gr.State([])
        with gr.Row():
            with gr.Column():
                apikey = gr.Textbox(label='apikey', value="")
                prompt = gr.Textbox(label='Prompt')
                run_button = gr.Button('Generate Image')

            result = gr.Image(label="Generated Image", height=512,width=512)

        apikey.change(fn=init_api, inputs=[apikey])
        run_button.click(fn=inference_warp,
                    inputs=prompt,
                    outputs=result,)

                    

        examples = [
                ["create an illustration of a romantic couple sharing a tender moment under a starry sky."],
                ["generate an image of a laughing woman, fashion magazine cover."],
                ["a robot cooking in the kitchen."],
                ["The man who whistles tunes pianos, watercolor."],
                

        ]

        gr.Examples(examples=examples,
                    inputs=prompt,
                    outputs=result,
                    fn=inference_warp,
                    cache_examples=True,
                    run_on_click=True
                    )

    demo.launch(server_name="0.0.0.0", server_port=7862)
