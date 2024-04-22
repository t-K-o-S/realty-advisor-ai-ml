from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from joblib import load

# Define device to use for processing
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pre-trained scaler
scaler = load('scaler.joblib')

# Neural network model
class HousePriceModel(nn.Module):
    def __init__(self, input_size):
        super(HousePriceModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, 128)
        self.fc3 = nn.Linear(128, 128)
        self.fc4 = nn.Linear(128, 128)
        self.fc5 = nn.Linear(128, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.relu(self.fc4(x))
        x = self.fc5(x)
        return x

# Define request body schema
class HouseFeatures(BaseModel):
    area: int
    bedrooms: int
    bathrooms: int
    furnishing: int
    parking: int

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow cross-origin requests
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize PyTorch model
input_size = 5  # Update this with the actual number of features
model = HousePriceModel(input_size)
model.load_state_dict(torch.load("models/house_price_model.pth", map_location=device))
model.to(device)
model.eval()

# Define route to accept input data and perform inference
@app.post("/predict/")
async def predict(features: HouseFeatures):
    try:
        # Convert input data to numpy array
        input_data = np.array([[features.area, features.bedrooms, features.bathrooms, features.furnishing, features.parking]])
        
        # Scale the input data using the loaded scaler
        scaled_data = scaler.transform(input_data)
        
        # Convert scaled data to PyTorch tensor
        input_tensor = torch.tensor(scaled_data, dtype=torch.float32)
        
        # Perform inference using the model
        with torch.no_grad():
            output = model(input_tensor).to(device)
        
        # You can return the output as needed
        return {"prediction": output.item()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the FastAPI app
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)