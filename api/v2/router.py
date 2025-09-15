from fastapi import APIRouter

router = APIRouter(prefix="/v2", tags=["v2"])

@router.get("/health")
async def health():
    return {"ok": True}