import json
from enum import Enum
from typing import IO, List, Optional

from sqlalchemy import (
    Column,
    ForeignKey,
    Integer,
    LargeBinary,
    String,
    Table,
    create_engine,
    select,
)
from sqlalchemy import Enum as SAEnum
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    scoped_session,
    sessionmaker,
)


class Base(DeclarativeBase):
    pass


class SourceType(Enum):
    FILE = 1
    WEBPAGE = 2


# association table for enabled sources per chat
chat_enabled_source = Table(
    "chat_enabled_source",
    Base.metadata,
    Column("chat_id", Integer, ForeignKey("chat.id"), primary_key=True),
    Column("source_item_id", Integer, ForeignKey("source_item.id"), primary_key=True),
)


class FileItem(Base):
    __tablename__ = "source_item"

    id: Mapped[int] = mapped_column(primary_key=True)
    raw_bytes: Mapped[bytes] = mapped_column(LargeBinary())
    title: Mapped[str] = mapped_column(String(255))
    path: Mapped[str] = mapped_column(String(1024))
    type: Mapped[SourceType] = mapped_column(SAEnum(SourceType))
    is_source: Mapped[bool] = mapped_column(default=True)


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

    # Store serialized JSON strings for attachment_ids and source_ids
    attachment_ids: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    source_ids: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    @property
    def attachments(self) -> List[FileItem]:
        """Retrieve attachment FileItems for this message."""
        if not self.attachment_ids:
            return []
        attachment_id_list = json.loads(self.attachment_ids)
        return list(
            db_session.scalars(
                select(FileItem).where(FileItem.id.in_(attachment_id_list))
            ).all()
        )

    @attachments.setter
    def attachments(self, attachments: list[FileItem]):
        """Set attachments for this message by storing their IDs."""
        self.attachment_ids = json.dumps([attachment.id for attachment in attachments])

    @property
    def sources(self) -> List[FileItem]:
        """Retrieve source FileItems for this message."""
        if not self.source_ids:
            return []
        source_id_list = json.loads(self.source_ids)
        return list(
            db_session.scalars(
                select(FileItem).where(FileItem.id.in_(source_id_list))
            ).all()
        )

    @sources.setter
    def sources(self, sources: list[FileItem]):
        """Set sources for this message by storing their IDs."""
        self.source_ids = json.dumps([source.id for source in sources])

    def add_sources(self, sources: list[FileItem]):
        """Add multiple FileItems to sources and update the database."""
        current_sources = self.sources  # Retrieve the current list
        current_sources.extend(sources)  # Modify the list
        self.sources = current_sources  # Trigger the setter
        db_session.merge(self)  # Persist changes
        db_session.commit()

    def add_attachments(self, attachments: list[FileItem]):
        """Add multiple FileItems to attachments and update the database."""
        current_attachments = self.attachments  # Retrieve the current list
        current_attachments.extend(attachments)  # Modify the list
        self.attachments = current_attachments  # Trigger the setter
        db_session.merge(self)  # Persist changes
        db_session.commit()


class Chat(Base):
    __tablename__ = "chat"

    id: Mapped[int] = mapped_column(primary_key=True)
    title: Mapped[str] = mapped_column(String(1024))
    model: Mapped[str] = mapped_column(String(255))

    # Store serialized JSON strings for messages and enabled_sources
    message_ids: Mapped[Optional[str]] = mapped_column(String, nullable=True)
    enabled_source_ids: Mapped[Optional[str]] = mapped_column(String, nullable=True)

    @property
    def messages(self) -> List[Message]:
        """Retrieve Message objects for this chat."""
        if not self.message_ids:
            return []
        message_id_list = json.loads(self.message_ids)
        return list(
            db_session.scalars(select(Message).where(Message.id.in_(message_id_list)))
        )

    @messages.setter
    def messages(self, messages: list[Message]):
        """Set messages for this chat by storing their IDs."""
        self.message_ids = json.dumps([message.id for message in messages])

    @property
    def enabled_sources(self) -> List[FileItem]:
        """Retrieve FileItem objects for enabled sources in this chat."""
        if not self.enabled_source_ids:
            return []
        source_id_list = json.loads(self.enabled_source_ids)
        return list(
            db_session.scalars(select(FileItem).where(FileItem.id.in_(source_id_list)))
        )

    @enabled_sources.setter
    def enabled_sources(self, sources: list[FileItem]):
        """Set enabled sources for this chat by storing their IDs."""
        self.enabled_source_ids = json.dumps([source.id for source in sources])

    def add_message(
        self,
        author: str,
        text: str,
        attachment_ids: list[int] = [],
        source_ids: list[int] = [],
        files: list[IO[bytes]] = [],
    ) -> Message:
        """Create and persist a new Message attached to this Chat.

        Args:
            session: Active SQLAlchemy Session.
            author: Author name for the message (e.g., 'user' or 'assistant').
            text: Message text content.
            attachment_ids: List of FileItem IDs for attachments.
            source_ids: List of FileItem IDs for associated sources.

        Returns:
            The created Message.
        """
        if files:
            for i in files:
                i.seek(0)
                new_file = FileItem(
                    raw_bytes=i.read(),
                    title=i.name,
                    path=i.name,
                    is_source=False,
                    type=SourceType.FILE,
                )
                db_session.add(new_file)
                db_session.commit()
                attachment_ids.append(new_file.id)

        msg = Message(
            chat_id=self.id,
            author=author,
            text=text,
            attachment_ids=json.dumps(attachment_ids),  # Serialize as JSON
            source_ids=json.dumps(source_ids),  # Serialize as JSON
        )
        db_session.add(msg)
        db_session.commit()

        current_messages = self.messages
        current_messages.append(msg)
        self.messages = current_messages

        db_session.merge(self)
        db_session.commit()

        return msg

    def add_enabled_sources(self, sources: list[FileItem]):
        """Add multiple FileItems to enabled_sources and update the database."""
        current_sources = self.enabled_sources  # Retrieve the current list
        current_sources.extend(sources)  # Modify the list
        self.enabled_sources = current_sources  # Trigger the setter
        db_session.merge(self)  # Merge the object into the current session
        db_session.commit()

    def remove_enabled_sources(self, sources: list[FileItem]):
        """Remove multiple FileItems from enabled_sources and update the database."""
        current_sources = self.enabled_sources  # Retrieve the current list
        for source in sources:
            if source in current_sources:
                current_sources.remove(source)  # Modify the list
        self.enabled_sources = current_sources  # Trigger the setter
        db_session.merge(self)  # Merge the object into the current session
        db_session.commit()

    def add_messages(self, messages: list[Message]):
        """Add multiple Messages to messages and update the database."""
        current_messages = self.messages  # Retrieve the current list
        current_messages.extend(messages)  # Modify the list
        self.messages = current_messages  # Trigger the setter
        db_session.merge(self)  # Merge the object into the current session
        db_session.commit()

    def remove_messages(self, messages: list[Message]):
        """Remove multiple Messages from messages and update the database."""
        current_messages = self.messages  # Retrieve the current list
        for message in messages:
            if message in current_messages:
                current_messages.remove(message)  # Modify the list
        self.messages = current_messages  # Trigger the setter
        db_session.merge(self)  # Merge the object into the current session
        db_session.commit()


def new_chat(user_id=0, title=None, model="gemini-2.5-pro"):
    chat = Chat(title=title, model=model)
    db_session.add(chat)
    db_session.commit()
    return chat


def get_chats(user_id: int = 0):
    return list(db_session.scalars(select(Chat)).all())


def get_sources(user_id: int = 0):
    return list(db_session.scalars(select(FileItem).where(FileItem.is_source)))


def delete_chat(chat_id: int) -> bool:
    """Delete a chat and its messages by id.

    Returns True if a row was deleted, False if not found.
    """
    print(f"Deleting {chat_id}")
    chat = db_session.get(Chat, chat_id)
    print(chat)
    if chat is None:
        return False
    db_session.delete(chat)
    db_session.commit()
    return True


engine = create_engine("sqlite:///app_data.sqlite")
Base.metadata.create_all(engine)
db_session = scoped_session(sessionmaker(bind=engine, expire_on_commit=False))
