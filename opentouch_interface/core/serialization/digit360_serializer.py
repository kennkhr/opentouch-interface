import struct
import math
from enum import Enum

import numpy as np

from opentouch_interface.core.registries.class_registries import SerializerClassRegistry
from opentouch_interface.core.sensors.interfaces.digit360_interface import PressureApData, PressureData, GasHtData
from opentouch_interface.core.serialization.base_serializer import BaseSerializer, serialize, deserialize


@SerializerClassRegistry.register('Digit360')
class DigitSensorSerializer(BaseSerializer):
    """
    Serializer for the Digit360 sensor.

    Supports two data streams:
      - 'camera': Raw image serialization.
      - 'serial': Serial data that may contain one of several message types
                  (pressure_ap, pressure, imu, gas).

    For the 'serial' stream:
      - We prepend a 3-byte prefix (e.g., b'PAP', b'PRS', b'IMU', b'GAS') to identify
        which message type we're dealing with.
      - Then we call .SerializeToString() or .FromString() on the appropriate betterproto message.
    """

    def __init__(self) -> None:
        super().__init__()

    @staticmethod
    def _coerce_uint(value, bits: int, default: int = 0) -> int:
        max_value = (1 << bits) - 1
        if value is None:
            return default
        if isinstance(value, Enum):
            value = value.value
        try:
            if isinstance(value, str):
                value = value.strip()
                if value == "":
                    return default
                # Accept both "123" and "123.0"
                parsed = float(value) if "." in value else int(value)
                value = int(parsed)
            else:
                value = int(value)
        except (TypeError, ValueError, OverflowError):
            return default

        if value < 0:
            return 0
        if value > max_value:
            return max_value
        return value

    @staticmethod
    def _coerce_float(value, default: float = 0.0) -> float:
        if value is None:
            return default
        if isinstance(value, Enum):
            value = value.value
        try:
            parsed = float(value)
        except (TypeError, ValueError, OverflowError):
            return default
        if not math.isfinite(parsed):
            return default
        f32_max = np.finfo(np.float32).max
        if parsed > f32_max:
            return float(f32_max)
        if parsed < -f32_max:
            return float(-f32_max)
        return parsed

    # --------------------------------------------------
    # 1) Camera Stream
    # --------------------------------------------------
    @serialize('camera')
    def serialize_camera(self, frame: np.ndarray) -> bytes:
        """
        Serializes an OpenCV image as raw bytes with shape metadata.

        The first 12 bytes store (height, width, channels) as three unsigned ints.
        The rest is the pixel data in row-major order.
        """
        height, width, channels = frame.shape
        metadata = struct.pack('III', height, width, channels)
        return metadata + frame.tobytes()

    @deserialize('camera')
    def deserialize_camera(self, binary_data: bytes) -> np.ndarray:
        """
        Deserializes raw OpenCV image data.

        Reads the first 12 bytes (height, width, channels) and reconstructs
        the image as a numpy array of shape (height, width, channels).
        """
        height, width, channels = struct.unpack('III', binary_data[:12])
        image_data = binary_data[12:]
        return np.frombuffer(image_data, dtype=np.uint8).reshape((height, width, channels))

    # --------------------------------------------------
    # 2) Serial Stream
    # --------------------------------------------------
    @serialize('serial')
    def serialize_serial(self, data: dict) -> bytes:
        """
        Serializes the 'serial' stream for Digit360.

        For non-IMU types, this method falls back on the previous implementation
        (using betterproto messages). For the "imu" key, we flatten the data manually.
        """

        if "pressure_ap" in data:
            msg = PressureApData(**data["pressure_ap"])
            return b'PAP' + bytes(msg)
        elif "pressure" in data:
            msg = PressureData(**data["pressure"])
            return b'PRS' + bytes(msg)
        elif "gas" in data:
            msg = GasHtData(**data["gas"])
            return b'GAS' + bytes(msg)
        elif "imu" in data:
            # Custom serialization for IMU.
            imu_dict = data["imu"]
            # Top-level timestamp (ts)
            ts = self._coerce_uint(imu_dict.get("ts", 0), bits=32)
            # Raw sub-message (24 bytes: I Q f f f)
            raw = imu_dict.get("raw", {})
            sensor_raw = self._coerce_uint(raw.get("sensor_", 0), bits=32)
            ts_ght_raw = self._coerce_uint(raw.get("ts_ght", 0), bits=64)
            x_raw = self._coerce_float(raw.get("x", 0.0))
            y_raw = self._coerce_float(raw.get("y", 0.0))
            z_raw = self._coerce_float(raw.get("z", 0.0))
            # Euler sub-message (20 bytes: Q f f f)
            euler = imu_dict.get("euler", {})
            ts_ght_euler = self._coerce_uint(euler.get("ts_ght", 0), bits=64)
            heading = self._coerce_float(euler.get("heading", 0.0))
            pitch = self._coerce_float(euler.get("pitch", 0.0))
            roll = self._coerce_float(euler.get("roll", 0.0))
            # Quat sub-message (28 bytes: Q f f f f f)
            quat = imu_dict.get("quat", {})
            ts_ght_quat = self._coerce_uint(quat.get("ts_ght", 0), bits=64)
            x_quat = self._coerce_float(quat.get("x", 0.0))
            y_quat = self._coerce_float(quat.get("y", 0.0))
            z_quat = self._coerce_float(quat.get("z", 0.0))
            w_quat = self._coerce_float(quat.get("w", 0.0))
            accuracy = self._coerce_float(quat.get("accuracy", 0.0))
            # Define the format string:
            # "I" for ts, then raw: "I Q f f f", euler: "Q f f f", quat: "Q f f f f f"
            fmt = "I" + "I Q f f f" + "Q f f f" + "Q f f f f f"
            try:
                packed = struct.pack(fmt, ts,
                                     sensor_raw, ts_ght_raw, x_raw, y_raw, z_raw,
                                     ts_ght_euler, heading, pitch, roll,
                                     ts_ght_quat, x_quat, y_quat, z_quat, w_quat, accuracy)
            except struct.error as e:
                raise ValueError(f"Error packing IMU data: {e}")
            # Prepend the 3-byte prefix for IMU.
            return b'IMU' + packed
        else:
            raise ValueError("Unknown serial data type for Digit360.")

    @deserialize('serial')
    def deserialize_serial(self, binary_data: bytes) -> dict:
        """
        Deserializes binary data for the 'serial' stream.

        For non-IMU types, uses the previous approach. For IMU, unpacks a fixed 76-byte payload.
        Returns a dict with key "imu" mapping to a dictionary containing ts, raw, euler, and quat.
        """
        prefix = binary_data[:3]
        payload = binary_data[3:]

        if prefix == b'PAP':
            msg = PressureApData.FromString(payload)
            return {"pressure_ap": msg}
        elif prefix == b'PRS':
            msg = PressureData.FromString(payload)
            return {"pressure": msg}
        elif prefix == b'GAS':
            msg = GasHtData.FromString(payload)
            return {"gas": msg}
        elif prefix == b'IMU':
            fmt = "I" + "I Q f f f" + "Q f f f" + "Q f f f f f"
            expected_size = struct.calcsize(fmt)
            if len(payload) != expected_size:
                raise ValueError(f"Unexpected IMU payload size: {len(payload)}, expected {expected_size}")
            unpacked = struct.unpack(fmt, payload)
            # Unpacked order:
            # Index 0: ts
            # 1-5: raw: sensor_raw, ts_ght_raw, x_raw, y_raw, z_raw
            # 6-9: euler: ts_ght_euler, heading, pitch, roll
            # 10-15: quat: ts_ght_quat, x_quat, y_quat, z_quat, w_quat, accuracy
            ts = unpacked[0]
            raw = {
                "sensor_": unpacked[1],
                "ts_ght": unpacked[2],
                "x": unpacked[3],
                "y": unpacked[4],
                "z": unpacked[5],
            }
            euler = {
                "ts_ght": unpacked[6],
                "heading": unpacked[7],
                "pitch": unpacked[8],
                "roll": unpacked[9],
            }
            quat = {
                "ts_ght": unpacked[10],
                "x": unpacked[11],
                "y": unpacked[12],
                "z": unpacked[13],
                "w": unpacked[14],
                "accuracy": unpacked[15],
            }
            return {"imu": {"ts": ts, "raw": raw, "euler": euler, "quat": quat}}
        else:
            raise ValueError("Unknown serial data prefix for Digit360.")

    @serialize('audio')
    def serialize_audio(self, audio: list[np.ndarray]) -> bytes:
        """
        Serializes a list of chunks, where each chunk is a list of [ch1, ch2] samples.
        Preserves chunk boundaries.
        """
        num_chunks = len(audio)
        chunk_sizes = [len(chunk) for chunk in audio]
        flat_samples = [sample for chunk in audio for sample in chunk]
        arr = np.array(flat_samples, dtype=np.int16)

        # Header: num_chunks + chunk_sizes
        header = struct.pack('i', num_chunks)
        header += struct.pack(f'{num_chunks}i', *chunk_sizes)

        return header + arr.tobytes()

    @deserialize('audio')
    def deserialize_audio(self, binary_data: bytes) -> list[np.ndarray]:
        """
        Deserializes audio data into its original 3D form:
        list of chunks, each being a numpy array with shape (n, 2) containing [ch1, ch2] pairs.
        """
        offset = 0

        # Read number of chunks
        num_chunks = struct.unpack_from('i', binary_data, offset)[0]
        offset += 4

        # Read chunk sizes
        chunk_sizes = struct.unpack_from(f'{num_chunks}i', binary_data, offset)
        offset += 4 * num_chunks

        # Remaining bytes are audio samples
        arr = np.frombuffer(binary_data[offset:], dtype=np.int16)

        # Reshape to (N, 2)
        total_samples = sum(chunk_sizes)
        arr = arr.reshape((total_samples, 2))

        # Reconstruct chunks
        audio = []
        idx = 0
        for size in chunk_sizes:
            # Keep as numpy array instead of converting to list
            chunk = arr[idx:idx + size].copy()  # Create a copy to ensure it's a separate array
            audio.append(chunk)
            idx += size

        return audio
