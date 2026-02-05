import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import librosa
import os
from transformers import Wav2Vec2Model

# ==========================================
# 1. THE MODEL ARCHITECTURE
# ==========================================
class DeepFakeDetector(nn.Module):
    def __init__(self, frozen=True):
        super().__init__()
        print("Initializing Wav2Vec2-XLSR Brain...")
        self.frontend = Wav2Vec2Model.from_pretrained("facebook/wav2vec2-large-xlsr-53")
        
        if frozen:
            for param in self.frontend.parameters():
                param.requires_grad = False
                
        self.backend = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 1)
        )

    def forward(self, input_values):
        outputs = self.frontend(input_values)
        # Max-pooling: catches the "fakest" artifact in the clip
        features, _ = torch.max(outputs.last_hidden_state, dim=1) 
        logits = self.backend(features)
        return torch.sigmoid(logits)

# ==========================================
# 2. AUDIO PROCESSING & DATASET
# ==========================================
def load_audio(file_path):
    """Loads and standardizes audio for the model."""
    file_path = file_path.strip().replace("'", "").replace('"', "")
    audio, _ = librosa.load(file_path, sr=16000, duration=5)
    # Ensure fixed length (80,000 samples for 5 seconds)
    if len(audio) < 80000:
        audio = torch.nn.functional.pad(torch.tensor(audio), (0, 80000 - len(audio)))
    else:
        audio = torch.tensor(audio[:80000])
    return audio.unsqueeze(0)

class SimpleDataset(Dataset):
    def __init__(self, data_folder):
        self.samples = []
        for label, sub in enumerate(['real', 'fake']):
            p = os.path.join(data_folder, sub)
            if os.path.exists(p):
                for f in os.listdir(p):
                    if f.endswith(('.wav', '.mp3')):
                        self.samples.append((os.path.join(p, f), label))

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        path, label = self.samples[idx]
        return load_audio(path).squeeze(0), torch.tensor(label, dtype=torch.float32)

# ==========================================
# 3. MAIN INTERFACE
# ==========================================
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DeepFakeDetector().to(device)

    # Load previously trained brain if it exists
    if os.path.exists("deepfake_model.pth"):
        print("✅ Found trained weights. Loading...")
        model.load_state_dict(torch.load("deepfake_model.pth", map_location=device))
    
    while True:
        print("\n" + "="*40)
        print("OPTIONS: [1] Train  [2] Detect File  [3] Exit")
        choice = input("Select: ")

        if choice == '1':
            # Training Mode
            if not os.path.exists("data/real") or not os.path.exists("data/fake"):
                print("❌ Setup Error: Please create 'data/real' and 'data/fake' folders first!")
                continue
            
            ds = SimpleDataset("data")
            dl = DataLoader(ds, batch_size=2, shuffle=True)
            optimizer = optim.Adam(model.backend.parameters(), lr=0.0005)
            criterion = nn.BCELoss()

            print(f"Training on {len(ds)} samples...")
            model.train()
            for epoch in range(10):
                for aud, lab in dl:
                    aud, lab = aud.to(device), lab.to(device)
                    optimizer.zero_grad()
                    pred = model(aud).squeeze()
                    loss = criterion(pred, lab)
                    loss.backward()
                    optimizer.step()
                print(f"Epoch {epoch+1}/10 Complete.")
            
            torch.save(model.state_dict(), "deepfake_model.pth")
            print("✅ Brain Saved as 'deepfake_model.pth'!")

        elif choice == '2':
            # Detection Mode
            model.eval()
            path = input("Drag & Drop File Here: ")
            try:
                audio_input = load_audio(path).to(device)
                with torch.no_grad():
                    score = model(audio_input).item()
                
                print(f"\nRESULT for {os.path.basename(path)}")
                print(f"Confidence Level: {score:.4f}")
                print(f"STATUS: {'🚨 FAKE' if score > 0.5 else '🟢 REAL'}")
            except Exception as e:
                print(f"❌ Error: {e}")

        elif choice == '3':
            break