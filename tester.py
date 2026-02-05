import torch
import torch.nn as nn
from transformers import Wav2Vec2Model

class DeepFakeDetector(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        print("Loading Wav2Vec2 model from Hugging Face...")
        # Load SOTA pre-trained model (Meta's Wav2Vec2-XLSR)
        self.frontend = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        # Freeze frontend to save compute (optional, unfreeze for best accuracy)
        if frozen:
            print("Freezing frontend parameters...")
            for param in self.frontend.parameters():
                param.requires_grad = False
                
        # Classification Head (Lightweight)
        # This takes the 1024 features from Wav2Vec2 and decides Real vs Fake
        self.backend = nn.Sequential(
            nn.Linear(1024, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1) # Output: 1 score (0 to 1)
        )

    def forward(self, input_values):
        # input_values shape: (batch_size, time_steps)
        
        # 1. Extract features using Wav2Vec2
        outputs = self.frontend(input_values)
        
        # 2. Mean Pooling: Average the features across time
        # outputs.last_hidden_state shape: (batch, time, 1024) -> (batch, 1024)
        features = outputs.last_hidden_state.mean(dim=1) 
        
        # 3. Classify
        logits = self.backend(features)
        
        # 4. Convert to probability (0 to 1)
        return torch.sigmoid(logits)

# --- Main Execution Block ---
if __name__ == "__main__":
    print("--- Starting DeepFake Detector Setup ---")
    
    # 1. Initialize the model
    try:
        model = DeepFakeDetector()
        print("✅ Model created successfully.")
    except Exception as e:
        print(f"❌ Error creating model: {e}")
        exit()

    # 2. Create Dummy Audio
    # Real audio is usually 16kHz. Let's simulate 5 seconds of audio.
    # 5 seconds * 16,000 samples/sec = 80,000 samples
    print("Generating dummy audio signal (5 seconds)...")
    dummy_audio = torch.randn(1, 80000) 

    # 3. Run Inference
    print("Running prediction...")
    with torch.no_grad():
        prediction = model(dummy_audio)
    
    # 4. Interpret Result
    confidence = prediction.item()
    print("------------------------------------------------")
    print(f"Raw Confidence Score: {confidence:.4f}")
    
    # Threshold is usually 0.5
    if confidence > 0.5:
        print("Result: 🚨 FAKE Audio Detected")
    else:
        print("Result: 🟢 REAL Audio Detected")
    print("------------------------------------------------")

