"""Frameo ADB Backend Server.

This Quart application serves as a bridge between Home Assistant and Frameo
devices via ADB (Android Debug Bridge). It maintains persistent connections
and exposes a REST API for device control.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Any

import usb1
from adb_shell.adb_device import AdbDeviceUsb
from adb_shell.adb_device_async import AdbDeviceTcpAsync
from adb_shell.auth.keygen import keygen
from adb_shell.auth.sign_pythonrsa import PythonRSASigner
from adb_shell.exceptions import (
    AdbConnectionError,
    AdbTimeoutError,
    UsbDeviceNotFoundError,
    UsbReadFailedError,
    UsbWriteFailedError,
)
from adb_shell.transport.usb_transport import UsbTransport
from quart import Quart, jsonify, request

# --- Configuration ---

ADB_KEY_PATH = "/data/adbkey"
ADDON_OPTIONS_PATH = "/data/options.json"
SERVER_HOST = "0.0.0.0"
DEFAULT_SERVER_PORT = 5000
DEFAULT_TRANSPORT_TIMEOUT = 9.0
USB_AUTH_TIMEOUT = 120.0
TCP_AUTH_TIMEOUT = 20.0
DEFAULT_TCP_PORT = 5555

# Home Assistant brightness values are always on a 0-255 scale.
HASS_BRIGHTNESS_SCALE = 255
BACKLIGHT_SYSFS_ROOT = "/sys/class/backlight"


def _load_addon_options() -> dict[str, Any]:
    """Load addon configuration options.

    Returns:
        Dictionary of addon options.

    """
    options_path = Path(ADDON_OPTIONS_PATH)
    if options_path.exists():
        try:
            return json.loads(options_path.read_text())
        except (json.JSONDecodeError, OSError) as err:
            logging.warning("Failed to load addon options: %s", err)
    return {}


def _get_server_port() -> int:
    """Get the configured server port.

    Returns:
        Server port number.

    """
    # First check environment variable (useful for testing)
    env_port = os.environ.get("SERVER_PORT")
    if env_port:
        try:
            return int(env_port)
        except ValueError:
            pass

    # Then check addon options
    options = _load_addon_options()
    return options.get("server_port", DEFAULT_SERVER_PORT)


# --- Logging Setup ---

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
_LOGGER = logging.getLogger(__name__)


class ConnectionType(StrEnum):
    """Connection type for ADB devices."""

    USB = "USB"
    NETWORK = "NETWORK"


@dataclass
class DeviceConnection:
    """Holds the current device connection state."""

    client: AdbDeviceUsb | AdbDeviceTcpAsync | None = None
    is_usb: bool = False
    backlight_path: str | None = None
    backlight_max: int = HASS_BRIGHTNESS_SCALE

    @property
    def is_connected(self) -> bool:
        """Check if a device is connected and available."""
        return self.client is not None and self.client.available

    @property
    def supports_brightness(self) -> bool:
        """Return whether direct backlight brightness control is available.

        Only true for rooted devices where a backlight sysfs node was found.
        Non-rooted devices have no reliable way to change the screen
        brightness, so brightness control is intentionally left unavailable
        rather than silently doing nothing.
        """
        return self.backlight_path is not None

    async def close(self) -> None:
        """Close the current connection if any."""
        if self.client is not None:
            try:
                if self.is_usb:
                    await _run_sync(self.client.close)
                else:
                    await self.client.close()
            except Exception as err:
                _LOGGER.warning("Error closing connection: %s", err)
            finally:
                self.client = None
        self.backlight_path = None
        self.backlight_max = HASS_BRIGHTNESS_SCALE

    async def shell(self, command: str) -> str:
        """Execute a shell command on the connected device.

        Args:
            command: ADB shell command to execute.

        Returns:
            Command output.

        Raises:
            ConnectionError: If no device is connected.

        """
        if not self.is_connected:
            raise ConnectionError("Device is not connected or available")

        _LOGGER.info("Executing shell command: '%s'", command)

        if self.is_usb:
            return await _run_sync(self.client.shell, command)
        return await self.client.shell(command)


# --- Global State ---

_signer: PythonRSASigner | None = None
_connection = DeviceConnection()


# --- Helper Functions ---


async def _run_sync(func: Any, *args: Any, **kwargs: Any) -> Any:
    """Run a synchronous (blocking) function in an executor.

    Args:
        func: Synchronous function to run.
        *args: Positional arguments for the function.
        **kwargs: Keyword arguments for the function.

    Returns:
        Function result.

    """
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))


def _load_or_generate_keys() -> PythonRSASigner:
    """Load ADB keys from disk, or generate them if they don't exist.

    Returns:
        RSA signer for ADB authentication.

    """
    if not os.path.exists(ADB_KEY_PATH):
        _LOGGER.info("No ADB key found, generating a new one at %s", ADB_KEY_PATH)
        keygen(ADB_KEY_PATH)

    _LOGGER.info("Loading ADB key from %s", ADB_KEY_PATH)

    with open(ADB_KEY_PATH) as f:
        private_key = f.read()
    with open(f"{ADB_KEY_PATH}.pub") as f:
        public_key = f.read()

    return PythonRSASigner(public_key, private_key)


def _usb_auth_callback(device_client: Any) -> None:
    """Log a message when USB auth is needed.

    Args:
        device_client: USB device client (unused).

    """
    _LOGGER.info(
        "!!! ACTION REQUIRED !!! "
        "Please check your device's screen to 'Allow USB Debugging'."
    )


def _parse_power_state(dumpsys_output: str) -> dict[str, Any]:
    """Parse power state from dumpsys output.

    Args:
        dumpsys_output: Output from 'dumpsys power' command.

    Returns:
        Dictionary with is_on and brightness values.

    """
    is_on = "mWakefulness=Awake" in dumpsys_output
    brightness = 0

    for line in dumpsys_output.splitlines():
        if "mScreenBrightnessSetting=" in line:
            try:
                brightness = int(line.split("=")[1])
                break
            except (ValueError, IndexError):
                pass

    return {"is_on": is_on, "brightness": brightness}


async def _detect_backlight_control(connection: DeviceConnection) -> None:
    """Detect whether the device is rooted and has a usable backlight node.

    Brightness control on Frameo devices does not work through the normal
    Android `settings put system screen_brightness` path - it has no effect
    on the actual panel. The only way found to reliably change it is to
    write directly to the kernel backlight sysfs node, which requires root.

    This is therefore only enabled when:
    - `su` is available and grants root (checked via `su -c id`).
    - At least one entry exists under /sys/class/backlight/.

    The backlight node name (e.g. "rk28_bl") is SoC-specific, so it is
    auto-detected per device rather than hardcoded. If either check fails,
    brightness control is left disabled for this connection - non-rooted
    devices only ever get on/off control.

    Args:
        connection: The device connection to probe and update in place.

    """
    connection.backlight_path = None
    connection.backlight_max = HASS_BRIGHTNESS_SCALE

    try:
        root_check = await connection.shell("su -c id")
    except Exception as err:
        _LOGGER.info("Device is not rooted (su check failed: %s); "
                     "brightness control will be unavailable", err)
        return

    if "uid=0" not in root_check:
        _LOGGER.info("Device is not rooted; brightness control will be unavailable")
        return

    try:
        listing = await connection.shell(f"su -c 'ls {BACKLIGHT_SYSFS_ROOT}'")
    except Exception as err:
        _LOGGER.warning("Root available but backlight listing failed: %s", err)
        return

    names = [name.strip() for name in listing.split() if name.strip()]
    if not names:
        _LOGGER.info(
            "Root available but no backlight device found under %s; "
            "brightness control will be unavailable",
            BACKLIGHT_SYSFS_ROOT,
        )
        return

    backlight_name = names[0]
    backlight_dir = f"{BACKLIGHT_SYSFS_ROOT}/{backlight_name}"

    try:
        max_raw = await connection.shell(f"su -c 'cat {backlight_dir}/max_brightness'")
        connection.backlight_max = int(max_raw.strip())
    except Exception as err:
        _LOGGER.warning(
            "Could not read max_brightness for %s (%s); assuming %d",
            backlight_name, err, HASS_BRIGHTNESS_SCALE,
        )
        connection.backlight_max = HASS_BRIGHTNESS_SCALE

    connection.backlight_path = f"{backlight_dir}/brightness"
    _LOGGER.info(
        "Detected rooted backlight control at %s (native max=%d)",
        connection.backlight_path,
        connection.backlight_max,
    )


def _hass_to_native_brightness(hass_brightness: int, native_max: int) -> int:
    """Scale a 0-255 Home Assistant brightness value to the device's native range."""
    return round(hass_brightness * native_max / HASS_BRIGHTNESS_SCALE)


def _native_to_hass_brightness(native_brightness: int, native_max: int) -> int:
    """Scale a device-native brightness value to Home Assistant's 0-255 range."""
    if native_max <= 0:
        return 0
    return round(native_brightness * HASS_BRIGHTNESS_SCALE / native_max)


# --- Quart Web Application ---

app = Quart(__name__)


@app.before_serving
async def startup() -> None:
    """Initialize the ADB signer before starting the server."""
    global _signer
    _signer = await _run_sync(_load_or_generate_keys)
    _LOGGER.info("Frameo ADB Server initialized and ready for connection requests")


# --- API Endpoints ---


@app.route("/devices/usb", methods=["GET"])
async def get_usb_devices() -> tuple[Any, int]:
    """Scan for and return connected USB ADB devices.

    Returns:
        JSON list of device serial numbers.

    """
    _LOGGER.info("Request received: GET /devices/usb")

    try:
        devices = await _run_sync(UsbTransport.find_all_adb_devices)
        serials = [dev.serial_number for dev in devices]
        _LOGGER.info("Discovered USB devices: %s", serials)
        return jsonify(serials), 200

    except UsbDeviceNotFoundError:
        _LOGGER.warning("No USB devices found during scan")
        return jsonify([]), 200

    except Exception as err:
        _LOGGER.exception("Error finding USB devices")
        return jsonify({"error": str(err)}), 500


@app.route("/connect", methods=["POST"])
async def connect_device() -> tuple[Any, int]:
    """Establish a connection to a Frameo device.

    Expects JSON body with connection details:
    - connection_type: "USB" or "Network"
    - serial: Device serial (for USB)
    - host: Device IP (for Network)
    - port: Device port (for Network, default 5555)

    Returns:
        JSON status response.

    """
    global _connection

    conn_details = await request.get_json()
    if not conn_details:
        return jsonify({"error": "Connection details not provided"}), 400

    conn_type = conn_details.get("connection_type", "USB").upper()
    _LOGGER.info("Connect request via %s: %s", conn_type, conn_details)

    try:
        # Close any existing connection
        await _connection.close()

        if conn_type == ConnectionType.USB:
            serial = conn_details.get("serial")
            if not serial:
                return jsonify({"error": "USB connection requires a serial number"}), 400

            client = AdbDeviceUsb(
                serial=serial,
                default_transport_timeout_s=DEFAULT_TRANSPORT_TIMEOUT,
            )
            await _run_sync(
                client.connect,
                rsa_keys=[_signer],
                auth_timeout_s=USB_AUTH_TIMEOUT,
                auth_callback=_usb_auth_callback,
            )
            _connection = DeviceConnection(client=client, is_usb=True)

        else:  # NETWORK
            host = conn_details.get("host")
            if not host:
                return jsonify({"error": "Network connection requires a host"}), 400

            port = int(conn_details.get("port", DEFAULT_TCP_PORT))

            client = AdbDeviceTcpAsync(
                host=host,
                port=port,
                default_transport_timeout_s=DEFAULT_TRANSPORT_TIMEOUT,
            )
            await client.connect(rsa_keys=[_signer], auth_timeout_s=TCP_AUTH_TIMEOUT)
            _connection = DeviceConnection(client=client, is_usb=False)

        identifier = conn_details.get("serial") or conn_details.get("host")
        _LOGGER.info("Successfully connected to device: %s", identifier)

        await _detect_backlight_control(_connection)

        return jsonify({"status": "connected", "rooted": _connection.supports_brightness}), 200

    except (
        AdbConnectionError,
        AdbTimeoutError,
        UsbDeviceNotFoundError,
        usb1.USBError,
        ConnectionResetError,
    ) as err:
        _LOGGER.error("Failed to connect to device: %s", err)
        await _connection.close()
        return jsonify({"error": f"Connection failed: {err}"}), 500

    except Exception as err:
        _LOGGER.exception("Unexpected error during connection")
        await _connection.close()
        return jsonify({"error": f"Unexpected error: {err}"}), 500


@app.route("/state", methods=["POST"])
async def get_state() -> tuple[Any, int]:
    """Get the current device state (screen on/off, brightness, root status).

    Returns:
        JSON with is_on, brightness and rooted values.

    """
    _LOGGER.info("Request received: POST /state")

    try:
        output = await _connection.shell("dumpsys power")
        state = _parse_power_state(output)
        state["rooted"] = _connection.supports_brightness

        if _connection.supports_brightness:
            try:
                raw = await _connection.shell(f"su -c 'cat {_connection.backlight_path}'")
                native_value = int(raw.strip())
                state["brightness"] = _native_to_hass_brightness(
                    native_value, _connection.backlight_max
                )
            except Exception as err:
                _LOGGER.warning("Failed to read backlight brightness: %s", err)

        return jsonify(state), 200

    except ConnectionError as err:
        return jsonify({"error": str(err)}), 503

    except (
        AdbConnectionError,
        AdbTimeoutError,
        ConnectionResetError,
        usb1.USBError,
        UsbReadFailedError,
        UsbWriteFailedError,
    ) as err:
        _LOGGER.error("Device disconnected or connection failed: %s", err)
        await _connection.close()
        return jsonify({"error": "Device disconnected", "details": str(err)}), 503


@app.route("/brightness", methods=["POST"])
async def set_brightness() -> tuple[Any, int]:
    """Set the screen brightness via direct backlight sysfs access.

    Only available for rooted devices where a backlight node was detected
    during /connect - see `_detect_backlight_control`. Non-rooted devices
    have no working brightness control, so this returns 409 rather than
    silently doing nothing.

    Expects JSON body with:
    - brightness: Target brightness on a 0-255 scale (Home Assistant convention).

    Returns:
        JSON with result, or an error if brightness control is unavailable.

    """
    data = await request.get_json()
    brightness = data.get("brightness") if data else None

    if brightness is None:
        return jsonify({"error": "brightness not provided"}), 400

    if not _connection.supports_brightness:
        return jsonify({
            "error": "Device does not support brightness control "
                     "(root and a backlight sysfs node are required)"
        }), 409

    try:
        native_value = _hass_to_native_brightness(int(brightness), _connection.backlight_max)
        await _connection.shell(
            f"su -c 'echo {native_value} > {_connection.backlight_path}'"
        )
        _LOGGER.info(
            "Set brightness to %s (native=%d)", brightness, native_value
        )
        return jsonify({"result": "ok", "brightness": brightness}), 200

    except ConnectionError as err:
        return jsonify({"error": str(err)}), 503

    except (
        AdbConnectionError,
        AdbTimeoutError,
        ConnectionResetError,
        usb1.USBError,
        UsbReadFailedError,
        UsbWriteFailedError,
    ) as err:
        _LOGGER.error("Device disconnected or brightness command failed: %s", err)
        await _connection.close()
        return jsonify({"error": "Device disconnected", "details": str(err)}), 503


@app.route("/shell", methods=["POST"])
async def run_shell_command() -> tuple[Any, int]:
    """Execute an arbitrary shell command on the device.

    Expects JSON body with:
    - command: Shell command to execute

    Returns:
        JSON with command result.

    """
    data = await request.get_json()
    command = data.get("command") if data else None

    if not command:
        return jsonify({"error": "Command not provided"}), 400

    _LOGGER.info("Executing shell command: '%s'", command)

    try:
        result = await _connection.shell(command)
        _LOGGER.info("Shell command result: '%s'", result.replace('\n', ' | '))
        return jsonify({"result": result}), 200

    except ConnectionError as err:
        return jsonify({"error": str(err)}), 503

    except (
        AdbConnectionError,
        AdbTimeoutError,
        ConnectionResetError,
        usb1.USBError,
        UsbReadFailedError,
        UsbWriteFailedError,
    ) as err:
        _LOGGER.error("Device disconnected or command failed: %s", err)
        await _connection.close()
        return jsonify({"error": "Device disconnected", "details": str(err)}), 503


@app.route("/tcpip", methods=["POST"])
async def enable_tcpip() -> tuple[Any, int]:
    """Enable wireless ADB debugging on the device.

    Requires an active USB connection.

    Returns:
        JSON with result.

    """
    _LOGGER.info("Request received: POST /tcpip")

    if not _connection.is_usb or not _connection.is_connected:
        return jsonify({"error": "A USB connection is required for this action"}), 400

    try:
        port = DEFAULT_TCP_PORT
        await _run_sync(
            _connection.client._open,
            destination=f"tcpip:{port}".encode("utf-8"),
            transport_timeout_s=None,
            read_timeout_s=10.0,
            timeout_s=None,
        )
        return jsonify({"result": f"TCP/IP enabled on port {port}"}), 200

    except (
        AdbConnectionError,
        AdbTimeoutError,
        ConnectionResetError,
        usb1.USBError,
        UsbReadFailedError,
        UsbWriteFailedError,
    ) as err:
        _LOGGER.error("Device disconnected during tcpip command: %s", err)
        await _connection.close()
        return jsonify({"error": "Device disconnected", "details": str(err)}), 503

    except Exception as err:
        _LOGGER.exception("ADB error on tcpip command")
        return jsonify({"error": str(err)}), 500


if __name__ == "__main__":
    server_port = _get_server_port()
    _LOGGER.info("Starting Frameo ADB Server on port %d", server_port)
    app.run(host=SERVER_HOST, port=server_port, debug=False)
