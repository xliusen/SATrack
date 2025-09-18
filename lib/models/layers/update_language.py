import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration

import torch.nn as nn


class updateLanguage(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained("/data/data6T/xls/download/blip2-opt-2.7b")
        self.model = Blip2ForConditionalGeneration.from_pretrained("/data/data6T/xls/download/blip2-opt-2.7b", torch_dtype=torch.float16).to("cuda")


    def forward(self, images, language = None):
        # prompt = 'Question: Please describe directly what the image shows.  Answer:'
        # prompt = (
        #     f"Question: This is a grid of images showing the same object from different time steps in a video: '{language}'. "
        #     "Please describe the common visual characteristics of the object. Answer:"
        # )
        # prompt = (f"The following is a description of a target object in a video： ‘{language}’."
        #           "Please assess whether the description accurately reflects the object’s category, color, shape, size, or distinctive features shown in the image:"
        #           "If the description is accurate and clear, output the original description as is."
        #           "If the description is inaccurate, vague, or incomplete, revise it into a more precise, specific, and concise one-sentence description."
        #           )
        # ========== 预处理输入 ==========
        # if language is not None:
        #     inputs = self.processor(images, language, return_tensors="pt").to("cuda", torch.float16)
        # else:
        #     inputs = self.processor(images, return_tensors="pt").to("cuda", torch.float16)
        # prompt = (f'Question: The previous description of the target was: {language}. '
        #           f'Given the template images showing the target\'s stable appearance and the current frame showing the updated scene,'
        #           f' generate an updated and consistent description of the target in the current frame.  Answer:')
        # prompt = f"""Previous description:
        # "{language}"
        # Based on the updated scene and the stable appearance from the templates, generate a consistent and updated description of the target."""
        # ====== 优化后的 Prompt ======
        # prompt = (
        #     "The first few images are templates showing the same object from earlier frames. "
        #     "The last image is the current frame. "
        #     "Describe the object in the current frame, keeping the identity consistent with the templates."
        # )
        prompt = "Describe the object shown in the image."
        # print(prompt)
        inputs = self.processor(images=images, text=prompt, return_tensors="pt").to("cuda", torch.float16)

        # ========== 生成描述 ==========
        generated_ids = self.model.generate(**inputs, max_new_tokens=100)
        generated_text = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
        return generated_text

