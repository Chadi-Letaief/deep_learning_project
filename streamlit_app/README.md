# Intel Scene Classifier — Streamlit App

Inference web app for the Intel Image Classification deep learning project.  
Classifies natural scene images into 6 categories: **buildings, forest, glacier, mountain, sea, street**.

## Files

```
app.py                  ← main Streamlit application
requirements.txt        ← Python dependencies
cnn_best.ckpt           ← ImprovedCNN checkpoint (from Colab training)
resnet_best.ckpt        ← ResNet18 Fine-Tuning checkpoint (from Colab training)
```

## Local run

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deployment on Streamlit Community Cloud (free)

1. Create a GitHub repository and push all files (app.py, requirements.txt, both .ckpt files)
   - Note: .ckpt files must be under 100MB each for GitHub. If larger, see note below.

2. Go to https://share.streamlit.io and sign in with GitHub

3. Click **New app** → select your repo → set main file as `app.py` → Deploy

4. Your app will be live at `https://your-username-repo-name.streamlit.app`

### If checkpoints exceed 100MB (GitHub limit)

Upload the .ckpt files to Google Drive and make them publicly accessible,
then add this to the top of app.py to download them at startup:

```python
import gdown, os

if not os.path.exists("cnn_best.ckpt"):
    gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", "cnn_best.ckpt")
if not os.path.exists("resnet_best.ckpt"):
    gdown.download("https://drive.google.com/uc?id=YOUR_FILE_ID", "resnet_best.ckpt")
```

Add `gdown` to requirements.txt if you use this approach.
