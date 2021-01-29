"""Data visualization functions
   """

from fastapi import APIRouter

router = APIRouter()

@router.post('/visualization')
async def navigation():
   return {'to be done':f'vis'}