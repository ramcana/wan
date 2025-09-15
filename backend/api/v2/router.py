from fastapi import APIRouter

router = APIRouter(prefix="/v2", tags=["Generation v2"])

@router.get("/health")
async def v2_health():
    return {"ok": True, "service": "v2", "status": "ready"}