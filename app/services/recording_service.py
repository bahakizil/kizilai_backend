"""
Recording Service
Audio recording storage and management for call recordings.
"""
import os
import io
import wave
import tempfile
from datetime import datetime, timedelta
from typing import Optional, List, BinaryIO
from uuid import UUID, uuid4
from pathlib import Path

from sqlalchemy import select, and_
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import settings
from app.models.analytics import CallRecording


class RecordingBuffer:
    """Buffer for accumulating audio chunks during a call."""

    def __init__(
        self,
        call_id: str,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ):
        self.call_id = call_id
        self.sample_rate = sample_rate
        self.channels = channels
        self.sample_width = sample_width

        self.user_chunks: List[bytes] = []
        self.agent_chunks: List[bytes] = []
        self.mixed_chunks: List[bytes] = []

        self.started_at = datetime.utcnow()

    def append_user_audio(self, chunk: bytes) -> None:
        """Append audio chunk from user."""
        self.user_chunks.append(chunk)
        self.mixed_chunks.append(chunk)

    def append_agent_audio(self, chunk: bytes) -> None:
        """Append audio chunk from agent."""
        self.agent_chunks.append(chunk)
        self.mixed_chunks.append(chunk)

    def get_user_audio(self) -> bytes:
        """Get all user audio as a single buffer."""
        return b"".join(self.user_chunks)

    def get_agent_audio(self) -> bytes:
        """Get all agent audio as a single buffer."""
        return b"".join(self.agent_chunks)

    def get_mixed_audio(self) -> bytes:
        """Get all audio (mixed) as a single buffer."""
        return b"".join(self.mixed_chunks)

    def get_duration_seconds(self) -> float:
        """Calculate total duration in seconds."""
        total_bytes = sum(len(c) for c in self.mixed_chunks)
        bytes_per_second = self.sample_rate * self.channels * self.sample_width
        return total_bytes / bytes_per_second if bytes_per_second > 0 else 0


class RecordingService:
    """Audio recording storage and management."""

    def __init__(self, db: AsyncSession):
        self.db = db
        self.config = settings.recording
        self._buffers: dict[str, RecordingBuffer] = {}

    def _create_wav_file(
        self,
        audio_data: bytes,
        sample_rate: int = 16000,
        channels: int = 1,
        sample_width: int = 2,
    ) -> bytes:
        """Create a WAV file from raw audio data."""
        buffer = io.BytesIO()

        with wave.open(buffer, "wb") as wav_file:
            wav_file.setnchannels(channels)
            wav_file.setsampwidth(sample_width)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_data)

        buffer.seek(0)
        return buffer.read()

    async def start_recording(
        self,
        call_id: UUID,
        recording_type: str = "full",
    ) -> Optional[CallRecording]:
        """
        Initialize a new recording session.

        Args:
            call_id: The call ID to record
            recording_type: Type of recording (full, user_only, agent_only)

        Returns:
            CallRecording object or None if recording is disabled
        """
        if not self.config.enabled:
            return None

        buffer = RecordingBuffer(
            call_id=str(call_id),
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )
        self._buffers[str(call_id)] = buffer

        # Create recording record
        recording = CallRecording(
            call_id=str(call_id),
            recording_type=recording_type,
            storage_provider=self.config.storage_provider,
            storage_path="",  # Will be set when finalized
            storage_bucket=self.config.bucket,
            format=self.config.format,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
            status="pending",
            retention_days=self.config.retention_days,
            expires_at=datetime.utcnow() + timedelta(days=self.config.retention_days),
        )

        self.db.add(recording)
        await self.db.flush()

        return recording

    def append_audio_chunk(
        self,
        call_id: UUID,
        chunk: bytes,
        is_user: bool,
    ) -> None:
        """
        Append audio chunk to recording buffer.

        Args:
            call_id: The call ID
            chunk: Raw audio bytes
            is_user: True if from user, False if from agent
        """
        buffer = self._buffers.get(str(call_id))
        if not buffer:
            return

        if is_user:
            buffer.append_user_audio(chunk)
        else:
            buffer.append_agent_audio(chunk)

    async def finalize_recording(
        self,
        recording_id: UUID,
    ) -> Optional[str]:
        """
        Finalize and upload recording.

        Args:
            recording_id: The recording ID to finalize

        Returns:
            Storage URL or None if failed
        """
        recording = await self.db.get(CallRecording, str(recording_id))
        if not recording:
            return None

        buffer = self._buffers.get(recording.call_id)
        if not buffer:
            recording.status = "failed"
            return None

        # Get audio data based on recording type
        if recording.recording_type == "user_only":
            audio_data = buffer.get_user_audio()
        elif recording.recording_type == "agent_only":
            audio_data = buffer.get_agent_audio()
        else:
            audio_data = buffer.get_mixed_audio()

        if not audio_data:
            recording.status = "failed"
            return None

        # Create WAV file
        wav_data = self._create_wav_file(
            audio_data,
            sample_rate=self.config.sample_rate,
            channels=self.config.channels,
        )

        # Generate storage path
        date_prefix = datetime.utcnow().strftime("%Y/%m/%d")
        filename = f"{recording.call_id}_{recording.recording_type}.{self.config.format}"
        storage_path = f"{date_prefix}/{filename}"

        # Upload based on storage provider
        storage_url = await self._upload_to_storage(
            wav_data,
            storage_path,
            recording.storage_provider,
            recording.storage_bucket,
        )

        if storage_url:
            recording.storage_path = storage_path
            recording.duration_seconds = buffer.get_duration_seconds()
            recording.file_size_bytes = len(wav_data)
            recording.status = "uploaded"
        else:
            recording.status = "failed"

        # Clean up buffer
        del self._buffers[recording.call_id]

        await self.db.flush()
        return storage_url

    async def _upload_to_storage(
        self,
        data: bytes,
        path: str,
        provider: str,
        bucket: Optional[str],
    ) -> Optional[str]:
        """Upload data to storage provider."""
        if provider == "local":
            return await self._upload_local(data, path)
        elif provider == "supabase":
            return await self._upload_supabase(data, path, bucket)
        elif provider == "s3":
            return await self._upload_s3(data, path, bucket)
        else:
            return None

    async def _upload_local(self, data: bytes, path: str) -> Optional[str]:
        """Upload to local filesystem."""
        try:
            local_path = Path(self.config.local_path) / path
            local_path.parent.mkdir(parents=True, exist_ok=True)

            with open(local_path, "wb") as f:
                f.write(data)

            return str(local_path)
        except Exception as e:
            print(f"Local upload error: {e}")
            return None

    async def _upload_supabase(
        self,
        data: bytes,
        path: str,
        bucket: Optional[str],
    ) -> Optional[str]:
        """Upload to Supabase Storage."""
        try:
            from app.core.supabase import get_supabase_admin

            supabase = get_supabase_admin()
            bucket_name = bucket or self.config.bucket

            # Upload file
            result = supabase.storage.from_(bucket_name).upload(
                path,
                data,
                file_options={"content-type": "audio/wav"},
            )

            # Get public URL
            url = supabase.storage.from_(bucket_name).get_public_url(path)
            return url
        except Exception as e:
            print(f"Supabase upload error: {e}")
            return None

    async def _upload_s3(
        self,
        data: bytes,
        path: str,
        bucket: Optional[str],
    ) -> Optional[str]:
        """Upload to AWS S3."""
        try:
            import boto3

            s3_client = boto3.client(
                "s3",
                aws_access_key_id=self.config.s3_access_key_id,
                aws_secret_access_key=self.config.s3_secret_access_key,
                region_name=self.config.s3_region,
            )

            bucket_name = bucket or self.config.s3_bucket

            s3_client.put_object(
                Bucket=bucket_name,
                Key=path,
                Body=data,
                ContentType="audio/wav",
            )

            return f"s3://{bucket_name}/{path}"
        except Exception as e:
            print(f"S3 upload error: {e}")
            return None

    async def get_recording_url(
        self,
        recording_id: UUID,
    ) -> Optional[str]:
        """Get the storage URL for a recording."""
        recording = await self.db.get(CallRecording, str(recording_id))
        if not recording or recording.status != "uploaded":
            return None

        if recording.storage_provider == "local":
            return recording.storage_path
        elif recording.storage_provider == "supabase":
            try:
                from app.core.supabase import get_supabase_admin

                supabase = get_supabase_admin()
                url = supabase.storage.from_(recording.storage_bucket).get_public_url(
                    recording.storage_path
                )
                return url
            except:
                return None
        elif recording.storage_provider == "s3":
            return f"https://{recording.storage_bucket}.s3.{self.config.s3_region}.amazonaws.com/{recording.storage_path}"

        return None

    async def delete_recording(
        self,
        recording_id: UUID,
    ) -> bool:
        """Delete a recording from storage."""
        recording = await self.db.get(CallRecording, str(recording_id))
        if not recording:
            return False

        # Delete from storage
        deleted = await self._delete_from_storage(
            recording.storage_path,
            recording.storage_provider,
            recording.storage_bucket,
        )

        if deleted:
            recording.status = "deleted"
            recording.deleted_at = datetime.utcnow()
            await self.db.flush()

        return deleted

    async def _delete_from_storage(
        self,
        path: str,
        provider: str,
        bucket: Optional[str],
    ) -> bool:
        """Delete file from storage provider."""
        try:
            if provider == "local":
                local_path = Path(self.config.local_path) / path
                if local_path.exists():
                    local_path.unlink()
                return True

            elif provider == "supabase":
                from app.core.supabase import get_supabase_admin

                supabase = get_supabase_admin()
                supabase.storage.from_(bucket).remove([path])
                return True

            elif provider == "s3":
                import boto3

                s3_client = boto3.client(
                    "s3",
                    aws_access_key_id=self.config.s3_access_key_id,
                    aws_secret_access_key=self.config.s3_secret_access_key,
                    region_name=self.config.s3_region,
                )
                s3_client.delete_object(Bucket=bucket, Key=path)
                return True

        except Exception as e:
            print(f"Storage delete error: {e}")
            return False

        return False

    async def delete_expired_recordings(self) -> int:
        """
        Delete recordings that have exceeded their retention period.

        Returns:
            Number of recordings deleted
        """
        now = datetime.utcnow()

        stmt = select(CallRecording).where(
            and_(
                CallRecording.expires_at <= now,
                CallRecording.status != "deleted",
                CallRecording.deleted_at.is_(None),
            )
        )

        result = await self.db.execute(stmt)
        expired_recordings = result.scalars().all()

        deleted_count = 0
        for recording in expired_recordings:
            if await self.delete_recording(UUID(recording.id)):
                deleted_count += 1

        return deleted_count

    async def get_call_recordings(
        self,
        call_id: UUID,
    ) -> List[CallRecording]:
        """Get all recordings for a call."""
        stmt = select(CallRecording).where(
            and_(
                CallRecording.call_id == str(call_id),
                CallRecording.status != "deleted",
            )
        )

        result = await self.db.execute(stmt)
        return result.scalars().all()
