from fastapi import FastAPI, HTTPException

router = FastAPI()

@router.get("/")
def read_root():
    return {"Hello": "World"}

@router.post("/send")
def send_message(message: str):
    if not message:
        raise HTTPException(status_code=400, detail="Message cannot be empty")
    return {"Message received": message}