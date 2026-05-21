import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import plotly.graph_objects as go
import os

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Intel Scene Classifier",
    page_icon="🏔️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ── Styling ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background: #0f0f0f;
    color: #f0ede8;
}

h1, h2, h3 {
    font-family: 'DM Serif Display', serif !important;
}

.hero-title {
    font-family: 'DM Serif Display', serif;
    font-size: 3.2rem;
    font-weight: 400;
    color: #f0ede8;
    line-height: 1.1;
    margin-bottom: 0.2rem;
}

.hero-subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 1rem;
    color: #888;
    font-weight: 300;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-bottom: 2.5rem;
}

.model-card {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: 0.5rem;
    cursor: pointer;
    transition: all 0.2s;
}

.model-card:hover {
    border-color: #c8b89a;
}

.result-box {
    background: linear-gradient(135deg, #1a1a1a 0%, #141414 100%);
    border: 1px solid #2a2a2a;
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    margin: 1.5rem 0;
}

.predicted-class {
    font-family: 'DM Serif Display', serif;
    font-size: 2.8rem;
    color: #c8b89a;
    margin: 0.3rem 0;
}

.confidence-value {
    font-size: 1.1rem;
    color: #888;
    font-weight: 300;
}

.confidence-high   { color: #7ec98f; }
.confidence-medium { color: #e8c47a; }
.confidence-low    { color: #e87a7a; }

.class-info {
    background: #161616;
    border-left: 3px solid #c8b89a;
    border-radius: 0 8px 8px 0;
    padding: 0.8rem 1.2rem;
    margin-top: 1rem;
    font-size: 0.9rem;
    color: #aaa;
    text-align: left;
}

.divider {
    border: none;
    border-top: 1px solid #222;
    margin: 2rem 0;
}

/* Upload zone */
[data-testid="stFileUploadDropzone"] {
    background: #161616 !important;
    border: 2px dashed #333 !important;
    border-radius: 12px !important;
    color: #888 !important;
}

[data-testid="stFileUploadDropzone"]:hover {
    border-color: #c8b89a !important;
}

/* Radio buttons */
[data-testid="stRadio"] > div {
    flex-direction: column;
    gap: 0.5rem;
}

[data-testid="stRadio"] label {
    background: #1a1a1a;
    border: 1px solid #2a2a2a;
    border-radius: 10px;
    padding: 0.8rem 1rem;
    cursor: pointer;
    transition: all 0.2s;
    color: #ccc !important;
}

[data-testid="stRadio"] label:hover {
    border-color: #c8b89a;
    color: #f0ede8 !important;
}

/* Selectbox */
[data-testid="stSelectbox"] > div > div {
    background: #1a1a1a !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 10px !important;
    color: #f0ede8 !important;
}

/* Button */
.stButton > button {
    background: #c8b89a !important;
    color: #0f0f0f !important;
    border: none !important;
    border-radius: 8px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    letter-spacing: 0.05em !important;
    padding: 0.6rem 2rem !important;
    width: 100%;
    transition: all 0.2s !important;
}

.stButton > button:hover {
    background: #e0d0b8 !important;
    transform: translateY(-1px);
}

/* Image */
[data-testid="stImage"] img {
    border-radius: 12px;
    border: 1px solid #2a2a2a;
}

/* Spinner */
.stSpinner > div {
    border-top-color: #c8b89a !important;
}

footer { visibility: hidden; }
#MainMenu { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Constants ──────────────────────────────────────────────────────────────
CLASSES = ['buildings', 'forest', 'glacier', 'mountain', 'sea', 'street']

CLASS_INFO = {
    'buildings': ('🏙️', 'Urban structures, cityscapes, and architectural scenes'),
    'forest':    ('🌲', 'Dense woodland, trees, and natural vegetation'),
    'glacier':   ('🧊', 'Ice formations, polar landscapes, and frozen terrain'),
    'mountain':  ('⛰️', 'Rocky peaks, highland terrain, and elevated landforms'),
    'sea':       ('🌊', 'Ocean, open water, coastal and marine environments'),
    'street':    ('🛣️', 'Roads, sidewalks, urban pathways, and street-level scenes'),
}

IMG_SIZE = 150
CKPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Model definitions (must match the notebook exactly) ───────────────────
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1), nn.BatchNorm2d(32),  nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.BatchNorm2d(256), nn.ReLU(), nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)), nn.Flatten(),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    def forward(self, x): return self.classifier(self.features(x))


class ResNetFineTune(nn.Module):
    def __init__(self, num_classes=6):
        super().__init__()
        base = models.resnet18(weights=None)
        self.backbone = nn.Sequential(*list(base.children())[:-1])
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(base.fc.in_features, 256), nn.ReLU(), nn.Dropout(0.4),
            nn.Linear(256, num_classes)
        )
    def forward(self, x): return self.head(self.backbone(x))


class LitIntelClassifier(nn.Module):
    """Thin wrapper that mirrors the Lightning module's state_dict key prefix 'model.*'"""
    def __init__(self, architecture):
        super().__init__()
        self.model = architecture
    def forward(self, x): return self.model(x)


# ── Model loader (cached) ──────────────────────────────────────────────────
@st.cache_resource
def load_model(model_name: str):
    if model_name == "ImprovedCNN":
        arch = LitIntelClassifier(ImprovedCNN(num_classes=6))
        ckpt_path = os.path.join(CKPT_DIR, "cnn_best.ckpt")
    else:
        arch = LitIntelClassifier(ResNetFineTune(num_classes=6))
        ckpt_path = os.path.join(CKPT_DIR, "resnet_best.ckpt")

    ckpt = torch.load(ckpt_path, map_location="cpu")
    arch.load_state_dict(ckpt["state_dict"])
    arch.eval()
    return arch


# ── Inference transform ────────────────────────────────────────────────────
inference_transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def predict(model, image: Image.Image):
    tensor = inference_transform(image).unsqueeze(0)
    with torch.no_grad():
        logits = model(tensor)
        probs  = F.softmax(logits, dim=1).squeeze().numpy()
    top_idx  = int(np.argmax(probs))
    return CLASSES[top_idx], float(probs[top_idx]), probs


# ── Confidence color helper ────────────────────────────────────────────────
def conf_class(conf):
    if conf >= 0.75: return "confidence-high"
    if conf >= 0.45: return "confidence-medium"
    return "confidence-low"


# ── Probability bar chart ──────────────────────────────────────────────────
def prob_chart(probs, predicted_class):
    colors = ["#c8b89a" if c == predicted_class else "#2a2a2a" for c in CLASSES]
    text_colors = ["#0f0f0f" if c == predicted_class else "#888" for c in CLASSES]

    fig = go.Figure(go.Bar(
        x=probs,
        y=CLASSES,
        orientation='h',
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{p*100:.1f}%" for p in probs],
        textposition='inside',
        textfont=dict(color=text_colors, size=12, family="DM Sans"),
        hovertemplate="%{y}: %{x:.1%}<extra></extra>",
    ))
    fig.update_layout(
        height=240,
        margin=dict(l=0, r=0, t=0, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(showgrid=False, showticklabels=False, range=[0, 1],
                   zeroline=False, fixedrange=True),
        yaxis=dict(showgrid=False, tickfont=dict(color="#aaa", size=12,
                   family="DM Sans"), fixedrange=True, autorange="reversed"),
        showlegend=False,
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════
# UI
# ══════════════════════════════════════════════════════════════════════════

st.markdown('<p class="hero-title">Scene<br><i>Classifier</i></p>', unsafe_allow_html=True)
st.markdown('<p class="hero-subtitle">Intel Image Classification · Deep Learning</p>', unsafe_allow_html=True)

# ── Model selector ─────────────────────────────────────────────────────────
st.markdown("**Choose a model**")
model_choice = st.radio(
    label="model",
    options=["ImprovedCNN", "ResNet18 Fine-Tuning"],
    label_visibility="collapsed",
    horizontal=False,
)
model_key = "ImprovedCNN" if model_choice == "ImprovedCNN" else "ResNet18"

model_desc = {
    "ImprovedCNN":  "4-block CNN trained from scratch · ~680K parameters",
    "ResNet18":     "ResNet18 fully fine-tuned · 11M parameters · ImageNet pre-trained",
}
st.markdown(f'<p style="color:#666;font-size:0.85rem;margin-top:-0.3rem;margin-bottom:1.5rem">↳ {model_desc[model_key]}</p>', unsafe_allow_html=True)

# ── File uploader ──────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("**Upload an image**")
uploaded = st.file_uploader(
    label="upload",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

if uploaded:
    image = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns([1, 1], gap="medium")
    with col1:
        st.image(image, use_container_width=True)

    with col2:
        with st.spinner("Analysing…"):
            try:
                model = load_model(model_key)
                pred_class, confidence, all_probs = predict(model, image)
                emoji, desc = CLASS_INFO[pred_class]

                st.markdown(f"""
                <div class="result-box">
                    <div style="font-size:2.5rem">{emoji}</div>
                    <div class="predicted-class">{pred_class.capitalize()}</div>
                    <div class="confidence-value {conf_class(confidence)}">
                        {confidence*100:.1f}% confidence
                    </div>
                </div>
                <div class="class-info">{desc}</div>
                """, unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error loading model: {e}")
                st.stop()

    # ── Probability chart ──────────────────────────────────────────────────
    st.markdown('<hr class="divider">', unsafe_allow_html=True)
    st.markdown("**Probability distribution across all classes**")
    st.plotly_chart(prob_chart(all_probs, pred_class),
                    use_container_width=True, config={"displayModeBar": False})

else:
    st.markdown("""
    <div style="text-align:center;padding:3rem 0;color:#444;font-size:0.95rem">
        Upload a <strong style="color:#666">JPG or PNG</strong> image to classify it<br>
        <span style="font-size:0.8rem">Buildings · Forest · Glacier · Mountain · Sea · Street</span>
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────
st.markdown('<hr class="divider">', unsafe_allow_html=True)
st.markdown("""
<p style="color:#444;font-size:0.8rem;text-align:center">
    Intel Image Classification · PyTorch Lightning · Streamlit
</p>
""", unsafe_allow_html=True)
