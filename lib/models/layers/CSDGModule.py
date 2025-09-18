import torch
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import torch.nn as nn
from sentence_transformers import SentenceTransformer, util

class CSDGModule(nn.Module):
    """
    Cross-frame Semantic Description Generator (CSDG)
    输入多帧目标模板图像，输出统一、稳定、抽象的语义描述
    """

    def __init__(self, model_path="/data/data6T/xls/download/blip2-opt-2.7b", use_fp16=True):
        super().__init__()
        self.processor = Blip2Processor.from_pretrained(model_path)
        dtype = torch.float16 if use_fp16 else torch.float32
        self.model = Blip2ForConditionalGeneration.from_pretrained(model_path, torch_dtype=dtype).to("cuda")
        self.sentence_model= SentenceTransformer('/data/data6T/xls/download/all-MiniLM-L6-v2')
        self.dtype = dtype

    def forward(self, image_list, language, object_class, use_mosaic=False, alpha=0.6):
        """
        Args:
            image_list: List[Image] (多帧模板图像)
            use_mosaic: bool (是否将图片拼成一张图像)
        Returns:
            A unified language description of the object across frames
        """
        if use_mosaic:
            mosaic_image = self._make_mosaic(image_list)
            prompt = (
                "This is a grid of images showing the same object from different time steps in a video. "
                "Please describe the common visual characteristics of the object. Answer:"
            )
            inputs = self.processor(images=mosaic_image, text=prompt, return_tensors="pt").to("cuda", self.dtype)
            generated_ids = self.model.generate(**inputs, max_new_tokens=50)
            return self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        else:
            prompt = (
                "Question: This picture is composed of two pictures spliced together. The target in the first picture is extracted from the second one. Please describe this target. Answer: The target is"
            )
            descriptions = []
            for img in image_list:
                # if object_class:
                #     inputs = self.processor(images=img, text=object_class, return_tensors="pt").to("cuda", self.dtype)
                # else:
                #     inputs = self.processor(images=img, text='', return_tensors="pt").to("cuda", self.dtype)
                inputs = self.processor(images=img, text=prompt, return_tensors="pt").to("cuda", self.dtype)
                generated_ids = self.model.generate(**inputs, max_new_tokens=40)
                description = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                # print(description)
                descriptions.append(description)

            # 合并多个描述
            # final_prompt = (
            #     "These are descriptions of the same object observed at different times:\n"
            #     + "\n".join(f"- {desc}" for desc in descriptions) +
            #     "\nBased on the above, summarize a single, unified description of the object. Answer:"
            # )
            # # 用一张代表性图像辅助引导摘要（如第一帧）
            # inputs = self.processor(images=image_list[0], text=final_prompt, return_tensors="pt").to("cuda", self.dtype)
            # generated_ids = self.model.generate(**inputs, max_new_tokens=60)
            # final_description = self.processor.tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            # model = SentenceTransformer('all-MiniLM-L6-v2')
            # 所有句子 + 引导描述编码
            all_sentences = descriptions + [language]
            embeddings = self.sentence_model.encode(all_sentences, convert_to_tensor=True)

            gen_embeddings = embeddings[:-1]  # 模型生成的句子
            ref_embedding = embeddings[-1]  # 原始参考句子

            # 计算生成句子的“中心”
            center_embedding = torch.mean(gen_embeddings, dim=0)

            # 到“中心”的距离（表示共性）
            center_sims = util.cos_sim(gen_embeddings, center_embedding)

            # 到参考句子的相似度（表示语义对齐）
            ref_sims = util.cos_sim(gen_embeddings, ref_embedding)

            # 综合评分
            total_scores = alpha * ref_sims + (1 - alpha) * center_sims  # shape: [N,1]

            best_idx = torch.argmax(total_scores).item()
            best_description = descriptions[best_idx]

            return best_description

    def _make_mosaic(self, image_list, grid_size=(2, 2)):
        """
        将图像列表拼接为一张 mosaic 图像
        """
        from PIL import ImageOps
        assert len(image_list) > 0
        img_w, img_h = image_list[0].size
        num_images = len(image_list)
        rows, cols = grid_size
        new_im = Image.new('RGB', (cols * img_w, rows * img_h))

        for i, img in enumerate(image_list):
            if i >= rows * cols:
                break
            row = i // cols
            col = i % cols
            new_im.paste(ImageOps.fit(img, (img_w, img_h)), (col * img_w, row * img_h))
        return new_im
