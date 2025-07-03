import base64
import struct

from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

from src.common.conf.settings import settings

S_KEY = settings.SECRET_KEY


def encrypt_db(value) -> str:
    CRC24_INIT = 0xB704CE
    CRC24_POLY = 0x1864CFB

    def ord_safe(ch):
        if isinstance(ch, int):
            return ch
        return ord(ch)

    def crc24(data):
        crc = CRC24_INIT
        for byte in data:
            crc ^= ord_safe(byte) << 16
            for _i in range(8):
                crc <<= 1
                if crc & 0x1000000:
                    crc ^= CRC24_POLY
        return crc & 0xFFFFFF

    def pad(text, block_size, zero=False):
        num = block_size - (len(text) % block_size)
        ch = b"\0" if zero else chr(num).encode("latin-1")
        return text + (ch * num)

    def encrypt(data):
        aes_key = S_KEY[:32].encode('utf-8')
        iv = b"\0" * 16
        cipher = Cipher(algorithms.AES(aes_key), modes.CBC(iv), backend=default_backend())
        context = cipher.encryptor()
        return context.update(data) + context.finalize()
    def armor(data):
        body = base64.b64encode(data)
        crc = base64.b64encode(struct.pack(">L", crc24(data))[1:])
        body = body.decode('ascii')
        crc = crc.decode('ascii')
        return f"-----BEGIN PGP MESSAGE-----\n\n{body}\n={crc}\n-----END PGP MESSAGE-----"

    utf8_encoded_value = value.encode('utf-8')
    padded = pad(utf8_encoded_value,16)
    encrypted = encrypt(padded)
    armored = armor(encrypted)
    return armored


def decrypt_db(value) -> str:
    return f"convert_from(decrypt(dearmor({value}), '{S_KEY}'::bytea, 'aes'), 'utf-8')"
