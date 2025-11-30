from enum import Enum
from typing import List, Optional

from sqlalchemy import ForeignKey, String
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


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


class User(Base):
    __tablename__ = "user"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(255))
    password_hash: Mapped[str] = mapped_column(String(1024))


class Message(Base):
    __tablename__ = "message"

    id: Mapped[int] = mapped_column(primary_key=True)
    chat_id: Mapped[int] = mapped_column(ForeignKey("chat.id"))
    author: Mapped[str] = mapped_column(String(255))
    text: Mapped[str] = mapped_column(String(262144))
    attachments: Mapped[Optional[SourceItem]]
    sources: Mapped[Optional[SourceItem]]


class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(1024))
    messages: Mapped[List[Message]]
    model: Mapped[str] = mapped_column(String(255))
