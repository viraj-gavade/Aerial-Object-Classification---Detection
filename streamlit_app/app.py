import streamlit as st
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import plotly.graph_objects as go


@st.cache_resource
def load_model():
    from torchvision.models import mobilenet_v2
    model = mobilenet_v2(weights=None)   
    
    model.classifier = nn.Sequential(
        nn.Dropout(0.2),
        nn.Linear(model.last_channel, 2)
    )
    
    # Handle both local development and deployment paths
    import os
    if os.path.exists("models/best_classifier.pt"):
        model_path = "models/best_classifier.pt"
    else:
        model_path = "../models/best_classifier.pt"
    
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.eval()
    return model

model = load_model()
classes = ["bird", "drone"]

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225])
])

st.set_page_config(
    page_title="Aerial Object Classifier", 
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        text-align: center;
        background: linear-gradient(120deg, #2193b0, #6dd5ed);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #666;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .prediction-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">🦅 Aerial Object Classifier 🚁</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Advanced AI-powered detection system for aerial objects</p>', unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    uploaded_file = st.file_uploader(
        "Upload an image", 
        type=["jpg", "jpeg", "png"],
        help="Supported formats: JPG, JPEG, PNG"
    )

if uploaded_file is not None:
    left_col, right_col = st.columns([1, 1])
    
    with left_col:
        st.markdown("### 📸 Uploaded Image")
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, use_column_width=True)
        
        with st.expander("ℹ️ Image Details"):
            st.write(f"**Filename:** {uploaded_file.name}")
            st.write(f"**Size:** {uploaded_file.size / 1024:.2f} KB")
            st.write(f"**Dimensions:** {image.size[0]} x {image.size[1]} px")
    
    with right_col:
        st.markdown("### 🎯 Analysis Results")
        
        with st.spinner("Analyzing image..."):
            img_tensor = transform(image).unsqueeze(0)
            
            with torch.no_grad():
                outputs = model(img_tensor)
                probs = torch.softmax(outputs, dim=1)
                confidence, pred_class = torch.max(probs, 1)
            
            predicted_label = classes[pred_class.item()]
            confidence_val = confidence.item() * 100
            bird_prob = probs[0][0].item() * 100
            drone_prob = probs[0][1].item() * 100
        
        st.markdown(f"""
            <div class="prediction-card">
                <h2 style="margin:0;">{"🦅" if predicted_label == "bird" else "🚁"}</h2>
                <h1 style="margin:0.5rem 0; font-size:2.5rem;">{predicted_label.upper()}</h1>
                <p style="font-size:1.5rem; margin:0;">{confidence_val:.1f}% confident</p>
            </div>
        """, unsafe_allow_html=True)
        
        st.markdown("#### 📊 Confidence Distribution")
        
        fig = go.Figure(data=[
            go.Bar(
                x=['Bird 🦅', 'Drone 🚁'],
                y=[bird_prob, drone_prob],
                marker=dict(
                    color=['#4CAF50' if predicted_label == 'bird' else '#90CAF9',
                           '#2196F3' if predicted_label == 'drone' else '#C8E6C9'],
                    line=dict(color='#333', width=2)
                ),
                text=[f'{bird_prob:.1f}%', f'{drone_prob:.1f}%'],
                textposition='auto',
            )
        ])
        
        fig.update_layout(
            height=300,
            margin=dict(l=20, r=20, t=20, b=20),
            yaxis=dict(title='Confidence (%)', range=[0, 100]),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        col_a, col_b = st.columns(2)
        with col_a:
            st.metric("🦅 Bird", f"{bird_prob:.1f}%", 
                     delta=f"{bird_prob - drone_prob:.1f}%" if predicted_label == "bird" else None)
        with col_b:
            st.metric("🚁 Drone", f"{drone_prob:.1f}%",
                     delta=f"{drone_prob - bird_prob:.1f}%" if predicted_label == "drone" else None)

else:
    st.markdown("---")
    info_col1, info_col2 = st.columns(2)
    
    with info_col1:
        st.markdown("""
        ### ✨ Features
        - 🎯 Real-time classification
        - 📊 Confidence visualization
        - 🖼️ Support for multiple image formats
        - ⚡ Fast inference with MobileNetV2
        """)
    
    with info_col2:
        st.markdown("""
        ### 🚀 How to Use
        1. Click "Upload an image" above
        2. Select an aerial image
        3. Get instant classification results
        4. View detailed confidence scores
        """)

st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #999; font-size: 0.9rem;'>Powered by PyTorch & MobileNetV2 | Built with Streamlit</p>",
    unsafe_allow_html=True
)