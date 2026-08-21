from fastapi import FastAPI, APIRouter, UploadFile, File, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
from motor.motor_asyncio import AsyncIOMotorClient
import os
import logging
from pathlib import Path
from pydantic import BaseModel, Field, ConfigDict
from typing import List, Dict
import uuid
from datetime import datetime, timezone

import forensics
import model_service


ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

MAX_UPLOAD_BYTES = 25 * 1024 * 1024  # 25 MB, matches frontend limit
ALLOWED_CONTENT_TYPES = {"image/png", "image/jpeg", "image/jpg", "image/webp"}

# MongoDB connection
mongo_url = os.environ['MONGO_URL']
client = AsyncIOMotorClient(mongo_url)
db = client[os.environ['DB_NAME']]

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")


# Define Models
class StatusCheck(BaseModel):
    model_config = ConfigDict(extra="ignore")  # Ignore MongoDB's _id field
    
    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    client_name: str
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

class StatusCheckCreate(BaseModel):
    client_name: str


class Probabilities(BaseModel):
    fake: float
    real: float


class AnalysisResult(BaseModel):
    model_config = ConfigDict(extra="ignore")

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    filename: str
    prediction: str
    confidence: float
    probabilities: Probabilities
    verificationId: str
    timestamp: str
    summary: str
    features: Dict[str, float]


# Add your routes to the router instead of directly to app
@api_router.get("/")
async def root():
    return {"message": "Hello World"}

@api_router.post("/status", response_model=StatusCheck)
async def create_status_check(input: StatusCheckCreate):
    status_dict = input.model_dump()
    status_obj = StatusCheck(**status_dict)
    
    # Convert to dict and serialize datetime to ISO string for MongoDB
    doc = status_obj.model_dump()
    doc['timestamp'] = doc['timestamp'].isoformat()
    
    _ = await db.status_checks.insert_one(doc)
    return status_obj

@api_router.get("/status", response_model=List[StatusCheck])
async def get_status_checks():
    # Exclude MongoDB's _id field from the query results
    status_checks = await db.status_checks.find({}, {"_id": 0}).to_list(1000)
    
    # Convert ISO string timestamps back to datetime objects
    for check in status_checks:
        if isinstance(check['timestamp'], str):
            check['timestamp'] = datetime.fromisoformat(check['timestamp'])
    
    return status_checks


@api_router.post("/analyze", response_model=AnalysisResult)
async def analyze_image(file: UploadFile = File(...)):
    """
    Accepts one medical image, runs ELA + forensic feature extraction, and
    returns an authenticity prediction. Image bytes are processed in memory
    only and are never written to disk or stored — only the resulting
    metadata is persisted, matching the "images are not retained" promise
    shown in the UI.
    """
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(status_code=400, detail="Unsupported file type. Please upload PNG, JPG or JPEG.")

    file_bytes = await file.read()
    if len(file_bytes) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=400, detail="File exceeds the maximum supported size of 25 MB.")

    try:
        image = forensics.load_image(file_bytes)
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read this file as an image.")

    ela_image = forensics.generate_ela_image(image)
    features = forensics.compute_forensic_features(image, ela_image)
    prediction = model_service.predict(image, ela_image, features)

    result = AnalysisResult(
        filename=file.filename or "upload",
        prediction=prediction["prediction"],
        confidence=prediction["confidence"],
        probabilities=Probabilities(**prediction["probabilities"]),
        verificationId=f"MT-VER-{uuid.uuid4().hex[:8].upper()}",
        timestamp=datetime.now(timezone.utc).isoformat(),
        summary=prediction["summary"],
        features=features,
    )

    # MongoDB persistence disabled temporarily so the API can be
    # tested without a MongoDB server running on localhost:27017.
    #
    # doc = result.model_dump()
    # await db.analyses.insert_one(doc)

    return result


@api_router.get("/analyze/{verification_id}", response_model=AnalysisResult)
async def get_analysis(verification_id: str):
    doc = await db.analyses.find_one({"verificationId": verification_id}, {"_id": 0})
    if not doc:
        raise HTTPException(status_code=404, detail="No analysis found for that verification ID.")
    return doc

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@app.on_event("shutdown")
async def shutdown_db_client():
    client.close()

