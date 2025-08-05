#!/usr/bin/env python3
"""
AES-256-GCM 文件加密工具
🔐 确保塔罗牌解读数据的隐私安全
"""
import os
import hashlib
import base64
import getpass
import argparse
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
import secrets

class FileEncryptor:
    """AES-256-GCM 文件加密器"""
    
    def __init__(self, password: str = None):
        """
        初始化加密器
        Args:
            password: 加密密码，如果为None则提示用户输入
        """
        if password is None:
            password = getpass.getpass("🔐 请输入加密密码: ")
        
        self.password = password.encode()
        # 使用加密安全的随机盐值
        self.salt = secrets.token_bytes(16)
        
    def derive_key(self):
        """派生加密密钥"""
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=self.salt,
            iterations=600000,
        )
        return kdf.derive(self.password)
    
    def encrypt_file(self, input_file: str, output_file: str = None) -> str:
        """
        加密单个文件
        Args:
            input_file: 输入文件路径
            output_file: 输出文件路径，默认为输入文件名 + .encrypted
        Returns:
            加密后的文件路径
        """
        if output_file is None:
            output_file = f"{input_file}.encrypted"
        
        print(f"🔐 加密文件: {input_file}")
        
        # 生成随机 nonce（12 字节）并派生密钥
        nonce = secrets.token_bytes(12)
        key = self.derive_key()
        aesgcm = AESGCM(key)
        
        # 计算原文件哈希（加密前）
        file_hash = self._calculate_file_hash(input_file)
        print(f"🔍 文件哈希(SHA-256): {file_hash}")
        
        # 一次性读取整个文件进行加密（内存友好，适用于 <1-2 GB 文件）
        with open(input_file, 'rb') as fin:
            plaintext = fin.read()
        ciphertext = aesgcm.encrypt(nonce, plaintext, None)
        
        # 写入: [16B salt] + [12B nonce] + ciphertext
        with open(output_file, 'wb') as fout:
            fout.write(self.salt)
            fout.write(nonce)
            fout.write(ciphertext)
        
        # 验证加密文件
        orig_size = os.path.getsize(input_file)
        enc_size = os.path.getsize(output_file)
        
        print(f"✅ 加密完成: {output_file}")
        print(f"📊 原文件大小: {orig_size} bytes")
        print(f"📊 加密文件大小: {enc_size} bytes")
        print(f"📈 大小增加: {enc_size - orig_size} bytes (头信息)")
        
        return output_file
    
    def decrypt_file(self, input_file: str, output_file: str = None) -> str:
        """
        解密文件
        Args:
            input_file: 加密文件路径
            output_file: 输出文件路径，默认去除.encrypted后缀
        Returns:
            解密后的文件路径
        """
        if output_file is None:
            if input_file.endswith('.encrypted'):
                output_file = input_file[:-10]  # 去除.encrypted
            else:
                output_file = f"{input_file}.decrypted"
        
        print(f"🔓 解密文件: {input_file}")
        
        try:
            with open(input_file, 'rb') as fin:
                # 读取盐值(16字节)和nonce(12字节)
                salt = fin.read(16)
                nonce = fin.read(12)
                
                # 重新派生密钥
                self.salt = salt
                key = self.derive_key()
                aesgcm = AESGCM(key)
                
                # 读取剩余全部密文并解密
                ciphertext = fin.read()
                plaintext = aesgcm.decrypt(nonce, ciphertext, None)
                
                with open(output_file, 'wb') as fout:
                    fout.write(plaintext)
            
            # 验证解密文件
            dec_hash = self._calculate_file_hash(output_file)
            print(f"✅ 解密完成: {output_file}")
            print(f"🔍 解密文件哈希: {dec_hash}")
            return output_file
            
        except Exception as e:
            print(f"❌ 解密失败: {str(e)}")
            # 清理部分解密文件
            if os.path.exists(output_file):
                os.remove(output_file)
            raise
    
    def _calculate_file_hash(self, file_path: str) -> str:
        """计算文件的SHA-256哈希值"""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            while True:
                data = f.read(65536)
                if not data:
                    break
                sha256.update(data)
        return sha256.hexdigest()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="塔罗牌数据隐私加密工具")
    parser.add_argument("--encrypt", metavar="FILE", help="加密指定文件")
    parser.add_argument("--decrypt", metavar="FILE", help="解密指定文件")
    
    args = parser.parse_args()
    
    if args.encrypt and args.decrypt:
        print("❌ 错误：不能同时使用 --encrypt 和 --decrypt")
        return
    
    if args.encrypt:
        if not os.path.exists(args.encrypt):
            print(f"❌ 文件不存在: {args.encrypt}")
            return
            
        encryptor = FileEncryptor()
        encryptor.encrypt_file(args.encrypt)
        
        print("\n🔐 加密成功！")
        print("💡 安全提示：")
        print("- 请将密码保存在安全的地方")
        print("- 删除原始未加密文件: "
              f"shred -u {args.encrypt} (Linux/Mac)")
        
    elif args.decrypt:
        if not os.path.exists(args.decrypt):
            print(f"❌ 文件不存在: {args.decrypt}")
            return
            
        encryptor = FileEncryptor()
        encryptor.decrypt_file(args.decrypt)
        
        print("\n🔓 解密成功！")
    else:
        parser.print_help()

if __name__ == "__main__":
    main()