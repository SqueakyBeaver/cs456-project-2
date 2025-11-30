from enum import Enum
from typing import List, Optional

from sqlalchemy import Enum as SAEnum
from sqlalchemy import ForeignKey, String, select
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column, relationship


class SourceType(Enum):
    FILE = 1
    WEBPAGE = 2


class Base(DeclarativeBase):
    pass


class FileItem(Base):
    __tablename__ = "source_item"

    id: Mapped[int] = mapped_column(primary_key=True)
    hash: Mapped[Optional[str]]
    title: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))
    type: Mapped[SourceType] = mapped_column(SAEnum(SourceType))


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

    # Foreign Keys to SourceItem
    attachment_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("source_item.id"), nullable=True
    )
    source_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("source_item.id"), nullable=True
    )

    # Relationships
    attachments: Mapped[Optional["FileItem"]] = relationship(
        foreign_keys=[attachment_id]
    )
    sources: Mapped[Optional["FileItem"]] = relationship(foreign_keys=[source_id])


class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(1024))
    model: Mapped[str] = mapped_column(String(255))

    # One-to-many relationship
    messages: Mapped[List["Message"]] = relationship(
        backref="chat",
        cascade="all, delete-orphan",
        passive_deletes=True,
    )

    def add_message(
        self,
        session: Session,
        author: str,
        text: str,
        attachment_id: Optional[int] = None,
        source_id: Optional[int] = None,
    ) -> int:
        """Create and persist a new Message attached to this Chat.

        Args:
            session: Active SQLAlchemy Session.
            author: Author name for the message (e.g., 'user' or 'assistant').
            text: Message text content.
            attachment_id: Optional FileItem.id for an attachment.
            source_id: Optional FileItem.id for an associated source.

        Returns:
            The created Message.id.
        """
        msg = Message(
            chat_id=self.id,
            author=author,
            text=text,
            attachment_id=attachment_id,
            source_id=source_id,
        )
        session.add(msg)
        session.commit()
        # Ensure relationship is updated in-memory
        if self.messages is None:
            self.messages = []
        self.messages.append(msg)
        return msg.id


def new_chat(session: Session, user_id=0, title=None, model="gemini-2.5-.pro"):
    chat = Chat(title=title, model=model)
    session.add(chat)
    session.commit()
    return chat.id


def get_chats(session: Session, user_id: int = 0):
    return list(session.scalars(select(Chat)).all())


def delete_chat(session: Session, chat_id: int) -> bool:
    """Delete a chat and its messages by id.

    Returns True if a row was deleted, False if not found.
    """
    chat = session.get(Chat, chat_id)
    if chat is None:
        return False
    session.delete(chat)
    session.commit()
    return True
