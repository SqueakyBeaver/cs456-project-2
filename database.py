from typing import List
from typing import Optional
from sqlalchemy import ForeignKey
from sqlalchemy import String
from sqlalchemy.orm import DeclarativeBase
from sqlalchemy.orm import Mapped
from sqlalchemy.orm import mapped_column
from sqlalchemy.orm import relationship
from enum import Enum

class SourceType(Enum):
    FILE = 1
    WEBPAGE = 2

class Base(DeclarativeBase):
    pass

class SourceItem(Base):
    __tablename__ = "source_item"

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[Optional[str]]
    title: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))
    type: Mapped[SourceType]

