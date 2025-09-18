import torch
from PIL import Image
from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration

import torch.nn as nn


class updateLanguage(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""

    def __init__(self):
        super().__init__()
        self.model_name = "Salesforce/instructblip-flan-t5-xl"

        # ---------- 加载模型 ----------
        self.processor = InstructBlipProcessor.from_pretrained(self.model_name)
        self.model = InstructBlipForConditionalGeneration.from_pretrained(self.model_name)


    def forward(self, template_images, search_images, language = None):
        assert len(template_images) == len(search_images), "模板图和搜索图数量不一致"

        # ---------- 拼接每帧的模板图和搜索图 ----------
        paired_images = []
        for t_img, s_img in zip(template_images, search_images):
            t_img_pil = Image.fromarray(t_img[..., ::-1])
            s_img_pil = Image.fromarray(s_img[..., ::-1])
            paired = self.concat_images([t_img_pil, s_img_pil], padding=5)
            paired_images.append(paired)

        # ---------- 拼接所有帧图像 ----------
        final_image = self.concat_images(paired_images, padding=10)

        # ---------- Prompt 编写 ----------
        prompt = (
            "Ignoring the structure of the image pairs, describe the object's appearance, clothing, and motion pattern across the frames."
        )

        # ---------- 模型推理 ----------
        inputs = self.processor(images=final_image, text=prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=40)
        description = self.processor.decode(outputs[0], skip_special_tokens=True)
        return description

    # ---------- 图像拼接函数 ----------
    def concat_images(self, images, padding=10):
        widths, heights = zip(*(img.size for img in images))
        total_width = sum(widths) + padding * (len(images) - 1)
        max_height = max(heights)
        new_img = Image.new("RGB", (total_width, max_height), (255, 255, 255))

        x_offset = 0
        for img in images:
            new_img.paste(img, (x_offset, 0))
            x_offset += img.width + padding
        return new_img
