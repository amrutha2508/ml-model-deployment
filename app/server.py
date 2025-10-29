from fastapi import FastAPI
import joblib
import numpy as np
from fastapi.middleware.cors import CORSMiddleware
import re
import nltk
import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizer
# uncomment below nltk.download if you are testing it locally
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)



#  ------------------ Load IRIS Model ------------------
models = {
    "model1": joblib.load("app/models/model.joblib")
}

class_names = {
    "model1": np.array(['setosa', 'versicolor', 'virginica'])
}


# ----------------------------
# Define MultiLabelClassifier
# ----------------------------
class MultiLabelClassifier(nn.Module):
    def __init__(self, num_labels):
        super(MultiLabelClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.dropout = nn.Dropout(0.3)
        self.batch_norm = nn.BatchNorm1d(self.bert.config.hidden_size)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(pooled_output)
        x = self.batch_norm(x)
        logits = self.classifier(x)
        return torch.sigmoid(logits)

# ----------------------------
# Load models with state_dict
# ----------------------------
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
broad_categories = ['{Education}', '{Careers & Workforce}', '{Society}', '{Other}']
num_broad_labels = len(broad_categories)
broad_model = MultiLabelClassifier(num_broad_labels)
broad_model.load_state_dict(
    torch.load("app/models/broad_model.pth", map_location=torch.device("cpu"))
)
broad_model.eval()  # important for inference

society_categories = [
    '{AI in various Industries}', '{AI in companies & Enterprises}',
    '{AI Investments & Market Trends}', '{AI Ethics, Law & Policy}',
    '{AI Governance & Geopolitics}', '{AI overview, risks & impact}'
]
num_society_labels = len(society_categories)
society_model = MultiLabelClassifier(num_society_labels)
society_model.load_state_dict(
    torch.load("app/models/society_model.pth", map_location=torch.device("cpu"))
)
society_model.eval()


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get('/')
def read_root():
    return {'message':'Iris Model API'}


@app.post("/predict/model1")
def predict_model1(data: dict):
    features = np.array(data['features']).reshape(1,-1)
    prediction = models["model1"].predict(features)
    class_name = class_names["model1"][prediction][0]
    return {"predicted_class": class_name}

# ------------------ Multi-label prediction ------------------
def preprocess_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'(?i)\s*artificial\s+intelligence\s*', ' ai ', text)
    text = re.sub(r'\bA\.I\.*\b', ' ai ', text, flags=re.IGNORECASE)
    tokens = nltk.word_tokenize(text)
    tokens = [re.sub(r'[^a-z]', ' ', token) for token in tokens]
    tokens = [token for token in tokens if token.strip()]
    text = ' '.join(tokens)
    text = re.sub(r'\b(ai\s+)+', 'ai ', text, flags=re.IGNORECASE)
    return text.strip()

def get_initial_predictions(text):
    device = torch.device("cpu")
    encoding = tokenizer(text, truncation=True, padding="max_length", max_length=128, return_tensors="pt")
    input_ids, attention_mask = encoding["input_ids"].to(device), encoding["attention_mask"].to(device)

    with torch.no_grad():
        broad_probs = broad_model(input_ids, attention_mask).cpu().numpy().flatten()
        society_probs = society_model(input_ids, attention_mask).cpu().numpy().flatten()
    return broad_probs, society_probs

def combine_and_filter_predictions(broad_probs, society_probs, broad_threshold=0.3, society_threshold=0.6):
    broad_pred_set = [(broad_categories[i], broad_probs[i]) for i in range(len(broad_probs)) if broad_probs[i] > broad_threshold]

    society_pred_set = []
    if "{Society}" in [pred[0] for pred in broad_pred_set]:
        society_pred_set = [(society_categories[i], society_probs[i]) for i in range(len(society_probs)) if society_probs[i] > society_threshold]

    combined_pred_set = []
    combined_probs = []

    for category, prob in broad_pred_set:
        if category == "{Society}":
            for society_category, society_prob in society_pred_set:
                combined_pred_set.append(society_category)
                combined_probs.append(society_prob)
        else:
            combined_pred_set.append(category)
            combined_probs.append(prob)

    sorted_indices = sorted(range(len(combined_probs)), key=lambda k: combined_probs[k], reverse=True)
    sorted_combined_pred_set = [combined_pred_set[i] for i in sorted_indices]
    sorted_combined_probs = [combined_probs[i] for i in sorted_indices]

    return sorted_combined_pred_set, sorted_combined_probs

@app.post("/predict/model2")
def predict_model2(data: dict):
    try:
        text = data.get("text", "")
        if not text.strip():
            return {"predictions": [], "probabilities": []}
        text = preprocess_text(text)
        broad_probs, society_probs = get_initial_predictions(text)
        predictions, probabilities = combine_and_filter_predictions(broad_probs, society_probs)

        # Convert numpy.float32 to Python float
        probabilities = [float(p) for p in probabilities]
        return {"predictions": predictions, "probabilities": probabilities}
    except Exception as e:
        import traceback
        return {"error": str(e), "trace": traceback.format_exc()}