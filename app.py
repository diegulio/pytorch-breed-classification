import gradio as gr
import torch
from torchvision import transforms

from app.backbone import Backbone
from app.config import CFG
from app.model import PetClassificationModel

# Load model
backbone = Backbone(CFG.MODEL, len(CFG.idx_to_class), pretrained=CFG.PRETRAINED)
model = PetClassificationModel(base_model=backbone.model, config=CFG)
model.load_state_dict(torch.load("models/best_model.pt"))
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Eval mode
model.eval()

model.to(device)


pred_transforms = transforms.Compose(
    [
        transforms.Resize(CFG.IMG_SIZE),
        transforms.ToTensor(),
    ]
)


def predict(x):
    x = pred_transforms(x).unsqueeze(0)  # transform and batched
    x = x.to(device)

    with torch.no_grad():
        prediction = torch.nn.functional.softmax(model(x)[0], dim=0)
        confidences = {
            CFG.idx_to_class[i]: float(prediction[i])
            for i in range(len(CFG.idx_to_class))
        }

    return confidences


gr.Interface(
    fn=predict,
    title="Breed Classifier 🐶🧡🐱",
    description="Clasifica una imagen entre: 120 razas, gato o ninguno!",
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=5),
    examples=[
        "statics/pug.jpg",
        "statics/poodle.jpg",
        "statics/cat.jpg",
        "statics/no.jpg",
    ],
).launch()
