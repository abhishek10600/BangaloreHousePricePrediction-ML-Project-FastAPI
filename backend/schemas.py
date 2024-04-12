from pydantic import BaseModel


class HouseDataInput(BaseModel):
    location: str
    bhk: float
    no_bathrooms: float
    total_sqft: int
