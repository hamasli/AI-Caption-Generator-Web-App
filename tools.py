from langchain.tools import BaseTool;
from transformers import BlipForConditionalGeneration, BlipProcessor;
from PIL import Image;
from transformers import DetrImageProcessor, DetrForObjectDetection
from PIL import Image
import torch;
class imageCaptionTool(BaseTool):
    name="Image Captioner";
    description="Use this tool when given the path to an image that you like to be  described" \
                "It will return a simple caption describing the image ";

    def _run(self,img_path):
        image = Image.open(img_path).convert('RGB');
        model_name = "Salesforce/blip-image-captioning-large";
        device = "cpu";
        processor = BlipProcessor.from_pretrained(model_name);
        model = BlipForConditionalGeneration.from_pretrained(model_name).to(device);

        inputs = processor(image, return_tensors='pt').to(device);
        output = model.generate(**inputs, max_new_tokens=20);
        # caption = processor().decode(output[0], skip_spcial_tokens=True)
        # Decode the generated tokens to get the caption
        caption = processor.tokenizer.decode(output[0], skip_special_tokens=True);
        return caption;

    def _arun(self,query):
        raise NotImplementedError('This tool does not support async')


class ObjectDetectionTool(BaseTool):
    name="Object detector";
    description = "Use this tool when given the path to an image that you like to detect objects." \
                  "It will return a list of all detected objects. Each element in the list in the format  " \
                  "[x1,y1,x2,y2] class_name confidence_score";


    def _run(self,img_path):

        image = Image.open(img_path).convert('RGB');
        # you can specify the revision tag if you don't want the timm dependency
        processor = DetrImageProcessor.from_pretrained("facebook/detr-resnet-50", revision="no_timm")
        model = DetrForObjectDetection.from_pretrained("facebook/detr-resnet-50", revision="no_timm")

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs)

        # convert outputs (bounding boxes and class logits) to COCO API
        # let's only keep detections with score > 0.9
        target_sizes = torch.tensor([image.size[::-1]])
        results = processor.post_process_object_detection(outputs, target_sizes=target_sizes, threshold=0.9)[0]

        detections = " ";
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            detections += '[ {},{},{},{}]'.format(int(box[0]), int(box[1]), int(box[2]), int(box[3]));
            detections += '{}'.format(model.config.id2label[int(label)]);
            detections += '{}\n'.format(float(score));

        return detections;

    def _arun(self,query):
        raise NotImplementedError('This tool does not support async')

